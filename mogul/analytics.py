"""Run identity, lane affiliation, passage counting and audited result aggregation."""
from dataclasses import dataclass,field,asdict
from uuid import uuid4
import cv2
import numpy as np
from .geometry import contains,intersects
from .landmarks import course_position


@dataclass
class Run:
    id: str
    first_ms: float
    last_ms: float
    first_depth: float
    max_depth: float
    track_ids: list=field(default_factory=list)
    first_lane: str | None=None
    first_lane_depth: float | None=None
    lane: str | None=None
    start_lane: str | None=None
    lane_ranges: dict=field(default_factory=dict)
    passed_lanes: list=field(default_factory=list)
    changes: list=field(default_factory=list)
    bumps: list=field(default_factory=list)
    qualified: bool=False
    samples: int=0
    point: list=field(default_factory=list)
    velocity: list=field(default_factory=lambda:[0.,0.])
    confidence: float=0
    relinked: bool=False
    review: str='unreviewed'
    merged_into: str | None=None
    pending_lane: str | None=None
    pending_since: float=0
    pending_hits: int=0


class AnalyticsEngine:
    def __init__(self,calibration,relink_ms=3000):
        self.c=calibration
        self.runs={}
        self.track_runs={}
        self.relink_ms=relink_ms

    def lane_for(self,point,current=None):
        # Keep the current lane near a shared edge; suppress single-frame jitter.
        if current:
            index=next((i for i,l in enumerate(self.c.lanes) if l.id==current),None)
            if index is not None and cv2.pointPolygonTest(np.array(self.c.lane_polygon(index),np.float32),tuple(map(float,point)),True)>=-.003:
                return current
        return next((lane.id for i,lane in enumerate(self.c.lanes) if contains(self.c.lane_polygon(i),point)),None)

    def _run(self,track,timestamp,visible_ids,claimed):
        if track.id in self.track_runs:
            return self.runs[self.track_runs[track.id]]
        matches=[]
        for run in self.runs.values():
            gap=timestamp-run.last_ms
            if run.id in claimed or any(i in visible_ids for i in run.track_ids) or not 0<gap<=self.relink_ms:
                continue
            predicted=np.asarray(run.point)+np.asarray(run.velocity)*gap/1000
            distance=np.linalg.norm(np.asarray(track.foot)-predicted)
            if distance<.06:
                matches.append((distance,run))
        matches.sort(key=lambda entry:entry[0])
        if matches and (len(matches)==1 or matches[1][0]-matches[0][0]>.025):
            run=matches[0][1]
            run.relinked=True
        else:
            depth=course_position(self.c,track.foot)[1]
            run=Run(str(uuid4()),timestamp,timestamp,depth,depth)
            self.runs[run.id]=run
        self.track_runs[track.id]=run.id
        run.track_ids.append(track.id)
        return run

    def update(self,tracks,timestamp,frame_index):
        records=[]
        visible_ids={t.id for t in tracks}
        claimed=set()
        for track in tracks:
            point=track.foot
            if not contains(self.c.middle,point):
                continue
            run=self._run(track,timestamp,visible_ids,claimed)
            claimed.add(run.id)
            previous=run.point[:]
            previous_ms=run.last_ms
            depth=course_position(self.c,point)[1]
            lane=self.lane_for(point,run.lane)
            if run.first_lane is None and lane is not None:
                run.first_lane=lane
                run.first_lane_depth=depth
            if run.lane is None:
                run.lane=lane
            elif lane is not None and lane!=run.lane:
                if run.pending_lane!=lane:
                    run.pending_lane=lane; run.pending_since=timestamp; run.pending_hits=1
                else:
                    run.pending_hits+=1
                if run.pending_hits>=3 and timestamp-run.pending_since>=300:
                    run.changes.append({'from':run.lane,'to':lane,'timestamp_ms':timestamp})
                    run.lane=lane; run.pending_lane=None; run.pending_hits=0
            else:
                run.pending_lane=None; run.pending_hits=0
            run.samples+=1
            run.max_depth=max(run.max_depth,depth)
            if lane is not None and lane==run.lane:
                bounds=run.lane_ranges.setdefault(run.lane,[depth,depth])
                bounds[1]=max(bounds[1],depth)  # progress from entry, never count uphill span
            run.qualified=run.qualified or (run.samples>=3 and run.max_depth-run.first_depth>=.08
                                            and any(b[1]-b[0]>=.06 for b in run.lane_ranges.values()))
            if run.qualified:
                if run.first_lane_depth is not None and run.first_lane_depth<=.25:
                    run.start_lane=run.first_lane
                run.passed_lanes=[lane for lane,bounds in run.lane_ranges.items() if bounds[1]-bounds[0]>=.06]
            if previous and lane==run.lane and lane is not None and 0<timestamp-previous_ms<=1500 and np.linalg.norm(np.array(point)-previous)<.25 and depth>course_position(self.c,previous)[1]+1e-5:
                visited={b['bump_id'] for b in run.bumps}
                for bump in self.c.bumps:
                    if bump.id in visited or bump.lane_id!=run.lane:
                        continue
                    polygon=bump.polygon
                    if contains(polygon,point) or contains(polygon,previous) or any(intersects(previous,point,polygon[i],polygon[(i+1)%4]) for i in range(4)):
                        segment=np.asarray(point)-previous
                        fraction=float(np.clip(np.dot(np.array(bump.center)-previous,segment)/max(1e-12,np.dot(segment,segment)),0,1))
                        run.bumps.append({'bump_id':bump.id,'label':bump.label,'timestamp_ms':previous_ms+fraction*(timestamp-previous_ms),'confidence':track.confidence})
            run.last_ms=timestamp
            run.point=list(point)
            run.velocity=list(track.velocity)
            run.confidence=track.confidence
            records.append({'track_id':track.id,'run_id':run.id,'timestamp_ms':timestamp,'frame_index':frame_index,
                            'bbox':track.bbox[:],'point':list(point),'lane_id':run.lane if lane==run.lane else None,'confidence':track.confidence})
        return records

    def snapshot(self):
        return [asdict(run) for run in self.runs.values()]


def summarize(runs,lanes):
    """Fold manual aliases, so a lane change or merged track cannot duplicate a run."""
    by_id={run['id']:run for run in runs}
    groups={}
    for run in runs:
        current=run
        visited=set()
        while current.get('merged_into'):
            if current['id'] in visited or current['merged_into'] not in by_id:
                raise ValueError('滑走IDの統合関係が不正です')
            visited.add(current['id'])
            current=by_id[current['merged_into']]
        groups.setdefault(current['id'],[]).append(run)
    result={'unique_runs':0,'course_changes':0,'starts':{l.id:0 for l in lanes},'passes':{l.id:0 for l in lanes},'groups':{'blue':0,'red':0,'neutral':0},'unreviewed':0}
    lane_groups={lane.id:lane.group for lane in lanes}
    for root,parts in groups.items():
        master=by_id[root]
        if master.get('review')=='excluded':
            continue
        active=[p for p in parts if p.get('review')!='excluded']
        if not any(p.get('qualified') for p in active):
            continue
        result['unique_runs']+=1
        if any(p.get('review')!='confirmed' for p in active):
            result['unreviewed']+=1
        active.sort(key=lambda r:r['first_ms'])
        start=next((p.get('start_lane') for p in active if p.get('start_lane')),None)
        if start in result['starts']:
            result['starts'][start]+=1
        passed={lane for p in active for lane in p.get('passed_lanes',[])}
        for lane in passed:
            if lane in result['passes']:
                result['passes'][lane]+=1
        for group in {lane_groups[l] for l in passed if l in lane_groups}:
            result['groups'][group]+=1
        override=master.get('change_count_override')
        result['course_changes']+=override if override is not None else len({(ch['from'],ch['to'],ch['timestamp_ms']) for p in active for ch in p.get('changes',[])})
    return result
