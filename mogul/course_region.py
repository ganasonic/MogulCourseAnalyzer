"""Course observation ROI, lane acquisition and independent boundary guards."""
import cv2
import numpy as np
from .geometry import contains


def in_course(calibration,point):
    return any(contains(calibration.lane_polygon(i),point) for i in range(len(calibration.lanes)))


def visible_track(calibration,track):
    return in_course(calibration,track['point']) or (track.get('course_tracking',False) and contains(calibration.middle,track['point']))


class CourseRegion:
    def __init__(self,calibration):
        self.polygons=[calibration.lane_polygon(i) for i in range(len(calibration.lanes))]
        self.course_polygon=calibration.middle
        self.transforms=[]
        self.guards=[]
        self.project_matrix=cv2.getPerspectiveTransform(np.asarray(calibration.middle,np.float32),
            np.array([[0,0],[1,0],[1,1],[0,1]],np.float32))
        for polygon in self.polygons:
            matrix=cv2.getPerspectiveTransform(np.asarray(polygon,np.float32),
                np.array([[0,0],[1,0],[1,1],[0,1]],np.float32))
            self.transforms.append(matrix)
            # Lane-relative guard, not a percentage of the entire video. Side
            # arrivals start inside a lane; the bottom guard identifies jumps.
            guard=np.array([[[-.12,-.10],[1.12,-.10],[1.12,1.10],[-.12,1.10]]],np.float32)
            expanded=cv2.perspectiveTransform(guard,np.linalg.inv(matrix))[0]
            base=np.asarray(polygon)
            # Perspective extrapolation near the vanishing point can otherwise
            # create a huge guard. Cap it in image space to a narrow boundary strip.
            limit=(base.max(axis=0)-base.min(axis=0))*[.12,.04]
            self.guards.append((base+np.clip(expanded-base,-limit,limit)).tolist())
        self.observation_polygons=[self.course_polygon]+self.guards

    def contains(self,point):
        return any(contains(polygon,point) for polygon in self.polygons)

    def project(self,point):
        value=self.project_matrix@np.array([point[0],point[1],1.])
        return value[:2]/value[2]

    def can_start(self,point):
        if self.contains(point):
            return True
        if not self.can_observe(point):
            return False
        # Entrance evidence is needed before the foot crosses the top edge.
        for matrix in self.transforms:
            x,y=cv2.perspectiveTransform(np.asarray([[point]],np.float32),matrix)[0,0]
            if 0<=x<=1 and -.10<=y<=0:
                return True
            if self.tracking_contains(point) and 0<=x<=1 and 1<=y<=1.10:
                return True  # bottom arrivals are tracked but never counted
        return False

    def can_observe(self,point):
        return any(contains(polygon,point) for polygon in self.observation_polygons)

    def tracking_contains(self,point):
        return contains(self.course_polygon,point)

    def can_enter(self,point):
        """Motion-only tracking starts at the entrance, never at mid-course noise."""
        if not self.can_observe(point):
            return False
        for matrix in self.transforms:
            x,y=cv2.perspectiveTransform(np.asarray([[point]],np.float32),matrix)[0,0]
            if 0<=x<=1 and -.10<=y<=.12:
                return True
        return False

    def crop(self,image):
        """Observe the enclosing course so existing targets remain visible in gaps."""
        height,width=image.shape[:2]
        points=np.concatenate([np.asarray(p) for p in self.observation_polygons])*[width,height]
        low=np.maximum(0,np.floor(points.min(axis=0))).astype(int)
        high=np.minimum([width,height],np.ceil(points.max(axis=0))+1).astype(int)
        x0,y0=low; x1,y1=high
        if x1<=x0 or y1<=y0:
            raise ValueError('コース内の解析画像領域がありません')
        crop=image[y0:y1,x0:x1].copy()
        mask=np.zeros(crop.shape[:2],np.uint8)
        for polygon in self.observation_polygons:
            contour=np.rint(np.asarray(polygon)*[width,height]-low).astype(np.int32)
            cv2.fillPoly(mask,[contour],255)
        crop[mask==0]=114
        return crop,(int(x0),int(y0))
