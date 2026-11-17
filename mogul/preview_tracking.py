"""Short-lived optical-flow overlays; these estimates never feed counting evidence."""
from collections import deque
from copy import deepcopy
import cv2
import numpy as np


class PreviewTracking:
    def __init__(self,max_age_ms=1200,region=None):
        self.history=deque(maxlen=90)
        self.tracks=[]
        self.detected_ms=None
        self.max_age_ms=max_age_ms
        self.region=region
        self.mapping=(0.,0.,1.,1.)

    def inside(self,tracks):
        if self.region is None:
            return tracks
        return [t for t in tracks if self.region.contains([(t['bbox'][0]+t['bbox'][2])/2,t['bbox'][3]])]

    def frame(self,image,timestamp,index):
        gray=self.gray(image)
        if self.history and timestamp<=self.history[-1][0]:
            self.history.clear(); self.tracks=[]
        if self.history:
            self.tracks=self.inside(self.advance(self.inside(self.tracks),self.history[-1][2],gray))
        self.history.append((timestamp,index,gray))
        while self.history and timestamp-self.history[0][0]>1500:
            self.history.popleft()
        if self.detected_ms is None or timestamp-self.detected_ms>self.max_age_ms:
            self.tracks=[]
        return deepcopy(self.tracks)

    def gray(self,image):
        if self.region is not None:
            height,width=image.shape[:2]
            image,(x,y)=self.region.crop(image)
            self.mapping=(x/width,y/height,image.shape[1]/width,image.shape[0]/height)
        h,w=image.shape[:2]
        scale=min(1,1280/w)
        return cv2.cvtColor(cv2.resize(image,(round(w*scale),round(h*scale))),cv2.COLOR_BGR2GRAY)

    def correct(self,tracks,timestamp,index,image=None):
        frames=list(self.history)
        start=next((i for i,f in enumerate(frames) if f[1]==index and abs(f[0]-timestamp)<1),None)
        if start is None and image is not None and frames and timestamp<=frames[-1][0]:
            frames=[(timestamp,index,self.gray(image))]+[f for f in frames if f[0]>timestamp]
            start=0
        if start is None or frames[-1][0]-timestamp>self.max_age_ms:
            return deepcopy(self.tracks)
        corrected=deepcopy(self.inside(tracks))
        for previous,current in zip(frames[start:],frames[start+1:]):
            corrected=self.inside(self.advance(corrected,previous[2],current[2]))
        self.tracks=corrected; self.detected_ms=timestamp
        return deepcopy(self.tracks)

    def advance(self,tracks,previous,current):
        if previous.shape!=current.shape:
            return []
        h,w=current.shape
        ox,oy,sx,sy=self.mapping
        result=[]
        for track in tracks:
            x1,y1,x2,y2=(np.asarray(track['bbox'])-[ox,oy,ox,oy])/[sx,sy,sx,sy]
            mask=np.zeros_like(previous)
            left,top,right,bottom=np.clip([x1*w,y1*h,x2*w,y2*h],[0,0,0,0],[w,h,w,h]).astype(int)
            mask[top:bottom,left:right]=255
            points=cv2.goodFeaturesToTrack(previous,maxCorners=24,qualityLevel=.02,minDistance=2,mask=mask)
            if points is None or len(points)<3:
                continue
            moved,status,_=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(15,15),maxLevel=2)
            if moved is None:
                continue
            back,back_status,_=cv2.calcOpticalFlowPyrLK(current,previous,moved,None,winSize=(15,15),maxLevel=2)
            if back is None:
                continue
            good=(status.ravel()>0)&(back_status.ravel()>0)&(np.linalg.norm(back-points,axis=2).ravel()<1)
            if good.sum()<3:
                continue
            delta=np.median((moved-points)[good].reshape(-1,2),axis=0)/[w,h]*[sx,sy]
            if not np.isfinite(delta).all() or np.linalg.norm(delta)>.1:
                continue
            bbox=(np.asarray(track['bbox'])+np.tile(delta,2)).tolist()
            if bbox[2]<=0 or bbox[0]>=1 or bbox[3]<=0 or bbox[1]>=1:
                continue
            updated=deepcopy(track)
            updated.update(bbox=bbox,point=[(bbox[0]+bbox[2])/2,bbox[3]],interpolated=True)
            result.append(updated)
        return result
