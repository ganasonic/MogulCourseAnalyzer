"""Predictive two-stage IoU tracking. Inspired by ByteTrack, independently implemented.

Uses global assignment, velocity prediction and high/low confidence association.
It is not the official ByteTrack implementation or appearance-based re-identification.
"""
from dataclasses import dataclass,field
import numpy as np


def iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]))
    h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    intersection=w*h
    return intersection/max(1e-12,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def assignment(cost):
    """Rectangular minimum-cost assignment, O(n^3), without SciPy dependency."""
    matrix=np.asarray(cost,float)
    if not matrix.size:
        return []
    transpose=matrix.shape[0]>matrix.shape[1]
    if transpose:
        matrix=matrix.T
    n,m=matrix.shape
    u=np.zeros(n+1); v=np.zeros(m+1)
    p=np.zeros(m+1,dtype=int); way=np.zeros(m+1,dtype=int)
    for row in range(1,n+1):
        p[0]=row
        column=0
        minima=np.full(m+1,np.inf)
        used=np.zeros(m+1,bool)
        while True:
            used[column]=True
            current=p[column]
            delta=np.inf; next_column=0
            for candidate in range(1,m+1):
                if used[candidate]:
                    continue
                value=matrix[current-1,candidate-1]-u[current]-v[candidate]
                if value<minima[candidate]:
                    minima[candidate]=value; way[candidate]=column
                if minima[candidate]<delta:
                    delta=minima[candidate]; next_column=candidate
            for candidate in range(m+1):
                if used[candidate]:
                    u[p[candidate]]+=delta; v[candidate]-=delta
                else:
                    minima[candidate]-=delta
            column=next_column
            if p[column]==0:
                break
        while column:
            previous=way[column]
            p[column]=p[previous]
            column=previous
    pairs=[(p[column]-1,column-1) for column in range(1,m+1) if p[column]]
    return [(j,i) for i,j in pairs] if transpose else pairs


@dataclass
class Track:
    id: int
    bbox: list
    confidence: float
    timestamp_ms: float
    hits: int=1
    velocity: list=field(default_factory=lambda:[0.,0.])

    @property
    def foot(self):
        return [(self.bbox[0]+self.bbox[2])/2,self.bbox[3]]

    def predicted(self,timestamp):
        delta=np.asarray(self.velocity)*max(0,(timestamp-self.timestamp_ms)/1000)
        return (np.asarray(self.bbox)+[delta[0],delta[1],delta[0],delta[1]]).tolist()


class AthleteTracker:
    def __init__(self,high=.35,low=.1,max_lost_ms=1500,min_hits=3):
        self.high,self.low=high,low
        self.max_lost_ms,self.min_hits=max_lost_ms,min_hits
        self.tracks=[]
        self.next_id=1

    def _match(self,tracks,detections,timestamp):
        if not tracks or not detections:
            return []
        costs=np.full((len(tracks),len(detections)),1e6)
        for i,track in enumerate(tracks):
            predicted=track.predicted(timestamp)
            point=np.array([(predicted[0]+predicted[2])/2,predicted[3]])
            diagonal=np.linalg.norm(np.array(predicted[2:])-predicted[:2])
            for j,detection in enumerate(detections):
                distance=np.linalg.norm(np.array(detection.foot)-point)
                intersection=iou(predicted,detection.bbox)
                gate=max(.025,min(.15,diagonal*1.2))
                if intersection>=.1 or distance<gate:
                    costs[i,j]=.7*(1-intersection)+.3*min(1,distance/gate)
        return [(i,j) for i,j in assignment(costs) if costs[i,j]<1]

    def update(self,detections,timestamp,can_start=None):
        self.tracks=[t for t in self.tracks if timestamp-t.timestamp_ms<=self.max_lost_ms]
        high=[d for d in detections if d.confidence>=self.high]
        low=[d for d in detections if self.low<=d.confidence<self.high]
        high_matches=self._match(self.tracks,high,timestamp)
        remaining=[t for i,t in enumerate(self.tracks) if i not in {a for a,_ in high_matches}]
        low_matches=self._match(remaining,low,timestamp)
        matched=[(self.tracks[i],high[j]) for i,j in high_matches]+[(remaining[i],low[j]) for i,j in low_matches]
        visible=[]
        for track,detection in matched:
            dt=max(.01,(timestamp-track.timestamp_ms)/1000)
            observed=(np.asarray(detection.foot)-track.foot)/dt
            track.velocity=(.5*np.asarray(track.velocity)+.5*observed).tolist()
            track.bbox=list(detection.bbox)
            track.timestamp_ms=timestamp
            track.confidence=detection.confidence
            track.hits+=1
            if track.hits>=self.min_hits:
                visible.append(track)
        for j,detection in enumerate(high):
            if j in {b for _,b in high_matches}:
                continue
            if can_start is not None and not can_start(detection.foot):
                continue
            track=Track(self.next_id,list(detection.bbox),detection.confidence,timestamp)
            self.next_id+=1
            self.tracks.append(track)
            if self.min_hits<=1:
                visible.append(track)
        return visible
