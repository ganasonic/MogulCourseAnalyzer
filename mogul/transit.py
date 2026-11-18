"""Observed lane entry/exit evidence, independent of legacy partial-passage totals."""
import cv2
import numpy as np
from uuid import uuid4


class LaneTransit:
    def __init__(self,c):
        self.c=c
        self.transforms={lane.id:cv2.getPerspectiveTransform(np.asarray(c.lane_polygon(i),np.float32),
                         np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)) for i,lane in enumerate(c.lanes)}
        for i,lane in enumerate(c.lanes):
            if not cv2.isContourConvex(np.asarray(c.lane_polygon(i),np.float32)) or abs(np.linalg.det(self.transforms[lane.id]))<1e-10:
                raise ValueError(f'レーン{lane.id}を完走判定できません。凸四角形に補正してください')

    def local(self,lane,point):
        return cv2.perspectiveTransform(np.asarray([[point]],np.float32),self.transforms[lane])[0,0]

    def event(self,run,kind,point,timestamp,lane):
        if any(e['kind']==kind for e in run.events):
            return
        bump=next((b for b in self.c.bumps if b.confirmed and b.lane_id==lane and
                   cv2.pointPolygonTest(np.asarray(b.polygon,np.float32),tuple(map(float,point)),False)>=0),None)
        run.events.append({'id':str(uuid4()),'kind':kind,'point':list(map(float,point)),
            'timestamp_ms':timestamp,'lane_id':lane,'review':'unreviewed',
            'bump_id':bump.id if bump else None,'bump_label':bump.label if bump else None})

    def update(self,run,track,previous,previous_ms,timestamp):
        point=track.foot
        gap=timestamp-previous_ms
        if run.entry_lane is None and previous and 0<gap<=1500:
            for lane in self.transforms:
                a,b=self.local(lane,previous),self.local(lane,point)
                # Require an observed crossing of the actual top edge, not first sight mid-lane.
                if a[1]<=0<b[1] and b[1]-a[1]<=.35:
                    f=-a[1]/(b[1]-a[1]); x=a[0]+f*(b[0]-a[0])
                    if 0<=x<=1:
                        run.entry_lane=lane; run.transit_state='in_lane'
                        break
                # Motion blobs do not have anatomical feet. Use a short entrance
                # zone as well, so one-pixel fragmentation at the exact top line
                # does not lose an otherwise observed downstream entrance.
                if self.c.analysis_config.get('detection_backend')=='motion' and 0<=a[1]<=.12<b[1] and b[1]-a[1]<=.35:
                    f=(.12-a[1])/(b[1]-a[1]);x=a[0]+f*(b[0]-a[0])
                    if 0<=x<=1:
                        run.entry_lane=lane;run.transit_state='in_lane'
                        break
        lane=run.entry_lane
        if lane is None or run.transit_state!='in_lane' or not previous:
            return
        a,b=self.local(lane,previous),self.local(lane,point)
        if gap>1500 or np.linalg.norm(b-a)>.4:
            run.transit_state='uncertain'  # disappearance / identity jump is never a finish or an exit
            return
        # A 2% lane-width tolerance suppresses boundary measurement jitter. Retain
        # the last inside anchor so a subsequent real departure still crosses it.
        if -.02<=b[0]<=1.02 and not 0<=b[0]<=1 and 0<b[1]<1:
            if not run.transit_anchor:
                run.transit_anchor=list(previous); run.transit_anchor_ms=previous_ms
            return
        if run.transit_anchor:
            previous=run.transit_anchor
            previous_ms=run.transit_anchor_ms
            gap=timestamp-previous_ms
            a=self.local(lane,previous)
            run.transit_anchor=[]
            if gap>1500:
                run.transit_state='uncertain'
                return
        # Solve the first intersection of the observed segment with the lane rectangle.
        crossings=[]
        for axis,edge,kind in [(0,0,'course_out'),(0,1,'course_out'),(1,0,'upstream'),(1,1,'finish')]:
            d=b[axis]-a[axis]
            outward=(edge==0 and d<0) or (edge==1 and d>0)
            if abs(d)<1e-8 or not outward:
                continue
            f=(edge-a[axis])/d
            other=a[1-axis]+f*(b[1-axis]-a[1-axis])
            if 0<=f<=1 and 0<=other<=1:
                crossings.append((float(f),kind))
        if crossings:
            f,kind=min(crossings)
            crossing=a+f*(b-a)
            where=cv2.perspectiveTransform(np.asarray([[crossing]],np.float32),np.linalg.inv(self.transforms[lane]))[0,0]
            segment=np.asarray(point)-previous
            image_fraction=float(np.clip(np.dot(where-previous,segment)/max(1e-12,np.dot(segment,segment)),0,1))
            when=previous_ms+image_fraction*gap
            if kind=='finish':
                run.completed_lane=lane; run.transit_state='completed'
            elif kind=='course_out':
                run.transit_state='course_out'
                self.event(run,'course_out',where,when,lane)
            else:
                run.transit_state='uncertain'
        # Conservative posture anomaly candidate. Not a confirmed stumble/cause diagnosis.
        if self.c.analysis_config.get('detection_backend')=='motion':
            return  # a foreground blob is not a person's posture bounding box
        x1,y1,x2,y2=track.bbox
        width,height=self.c.source.get('width',1),self.c.source.get('height',1)
        if self.c.rotation in (90,270):
            width,height=height,width
        ratio=(x2-x1)*width/max(1e-6,(y2-y1)*height)
        if run.transit_state=='in_lane' and ratio>=1.2 and run.upright_samples>=3:
            if run.balance_since is None:
                run.balance_since=timestamp
            elif timestamp-run.balance_since>=500:
                self.event(run,'balance_candidate',point,timestamp,lane)
        else:
            run.balance_since=None
        if ratio<.8:
            run.upright_samples+=1
