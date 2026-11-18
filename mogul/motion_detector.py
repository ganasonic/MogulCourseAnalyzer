"""Class-independent motion proposals in calibrated lanes; no learned model."""
import hashlib
import cv2
import numpy as np
from .athlete_detector import Detection


class MotionDetector:
    sha256=hashlib.sha256(b'lane-motion-background-v1').hexdigest()

    def __init__(self,min_area=3,delta=12):
        self.min_area=min_area
        self.delta=delta
        self.background=None
        self.previous=None
        self.previous_small=None
        self.status='背景初期化中'
        self.previous_boxes=[]

    def camera_transform(self,image,region):
        h,w=image.shape[:2]; scale=min(1,640/w)
        gray=cv2.cvtColor(cv2.resize(image,(round(w*scale),round(h*scale))),cv2.COLOR_BGR2GRAY)
        matrix=np.eye(2,3,dtype=np.float32)
        reset=False
        if self.previous_small is not None and gray.shape==self.previous_small.shape:
            mask=np.full_like(gray,255)
            for polygon in region.observation_polygons:
                cv2.fillPoly(mask,[np.rint(np.asarray(polygon)*[gray.shape[1],gray.shape[0]]).astype(np.int32)],0)
            points=cv2.goodFeaturesToTrack(self.previous_small,180,.03,8,mask=mask)
            if points is not None and len(points)>=12:
                moved,valid,_=cv2.calcOpticalFlowPyrLK(self.previous_small,gray,points,None)
                if moved is not None:
                    good=valid.ravel()>0
                    if good.sum()>=12:
                        estimated,inliers=cv2.estimateAffinePartial2D(points[good],moved[good],method=cv2.RANSAC,ransacReprojThreshold=1.5)
                        if estimated is not None and inliers.sum()>=12 and inliers.mean()>=.5:
                            angle=abs(np.arctan2(estimated[1,0],estimated[0,0]))
                            zoom=np.linalg.norm(estimated[:,0])
                            if np.linalg.norm(estimated[:,2])>6 or angle>.015 or abs(zoom-1)>.01:
                                reset=True
                            else:
                                matrix=estimated.astype(np.float32)
                                matrix[:,2]/=scale
        self.previous_small=gray
        return matrix,reset

    def detect_course(self,image,region,tiled=False):
        matrix,reset=self.camera_transform(image,region)
        masked,(x0,y0)=region.crop(image)
        h,w=masked.shape[:2]; ih,iw=image.shape[:2]
        gray=cv2.cvtColor(image[y0:y0+h,x0:x0+w],cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(3,3),.5)
        if self.background is None or self.background.shape!=gray.shape or reset:
            self.background=gray.astype(np.float32); self.previous=gray
            self.previous_boxes=[]
            self.status='カメラ移動により背景を再初期化' if reset else '背景初期化（最初の1枚は集計しません）'
            return []
        local=matrix.copy()
        local[:,2]+=matrix[:,:2]@np.array([x0,y0])-np.array([x0,y0])
        background=cv2.warpAffine(self.background,local,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        previous=cv2.warpAffine(self.previous,local,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        valid=cv2.warpAffine(np.ones((h,w),np.uint8),local,(w,h),flags=cv2.INTER_NEAREST)
        valid=cv2.erode(valid,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        mask=np.zeros((h,w),np.uint8)
        for polygon in region.observation_polygons:
            cv2.fillPoly(mask,[np.rint(np.asarray(polygon)*[iw,ih]-[x0,y0]).astype(np.int32)],255)
        allowed=(mask>0)&(valid>0)
        if not allowed.any():
            self.background=gray.astype(np.float32); self.previous=gray
            return []
        brightness=float(np.median((gray.astype(np.float32)-background)[allowed]))
        background=np.clip(background+brightness,0,255)
        difference=np.abs(gray.astype(np.float32)-background)
        noise=float(np.percentile(difference[allowed],90))
        changed=(difference>=max(self.delta,noise*1.5))&allowed
        current_change=np.abs(gray.astype(np.float32)-previous.astype(np.float32))>=max(4,self.delta*.5)
        count,labels,stats,centers=cv2.connectedComponentsWithStats(changed.astype(np.uint8),8)
        boxes=[]
        maximum=max(30,int(allowed.sum()*.12))
        for label in range(1,count):
            x,y,bw,bh,area=stats[label]
            if area<self.min_area or area>maximum:
                continue
            support=(labels[y:y+bh,x:x+bw]==label)
            overlaps=any(min(x+bw,b[2])>max(x,b[0]) and min(y+bh,b[3])>max(y,b[1]) for b in self.previous_boxes)
            if not np.any(current_change[y:y+bh,x:x+bw]&support) and not overlaps:
                continue  # stationary shadows / old background ghosts are not new motion
            boxes.append([int(x),int(y),int(x+bw),int(y+bh)])
        # Join nearby moving fragments (body, skis) without joining adjacent lanes.
        merged=[]
        for box in sorted(boxes,key=lambda b:(b[1],b[0])):
            point=[(x0+(box[0]+box[2])/2)/iw,(y0+box[3])/ih]
            lane=next((i for i,p in enumerate(region.polygons) if cv2.pointPolygonTest(np.asarray(p,np.float32),tuple(point),False)>=0),None)
            if lane is None:
                lane=next((i for i,p in enumerate(region.guards) if cv2.pointPolygonTest(np.asarray(p,np.float32),tuple(point),False)>=0),None)
            if lane is None and region.tracking_contains(point):
                lane=min(range(len(region.polygons)),key=lambda i:np.linalg.norm(np.mean(region.polygons[i],axis=0)-point))
            if lane is None:
                continue
            lane_width=np.ptp(np.asarray(region.polygons[lane])[:,0])*iw
            distance=max(2,min(6,lane_width*.2))
            match=next((entry for entry in merged if entry[0]==lane and
                max(entry[1][0],box[0])-min(entry[1][2],box[2])<=distance and
                max(entry[1][1],box[1])-min(entry[1][3],box[3])<=distance),None)
            if match:
                a=match[1]; a[:]=[min(a[0],box[0]),min(a[1],box[1]),max(a[2],box[2]),max(a[3],box[3])]
            else:
                merged.append([lane,box])
        learning=np.where(changed,.003,.04).astype(np.float32)
        self.background=background*(1-learning)+gray*learning
        self.previous=gray
        self.previous_boxes=[box for _,box in merged]
        self.status='移動対象を検出中（人間の見た目は判定しません）'
        return [Detection([(x0+b[0])/iw,(y0+b[1])/ih,(x0+b[2])/iw,(y0+b[3])/ih],1.) for _,b in merged]
