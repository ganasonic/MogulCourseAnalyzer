"""Off-GUI inference and sequential file analysis / bounded live frame queue."""
from copy import deepcopy
import queue
import time
import threading
from PySide6.QtCore import QThread,Signal
from .analysis_repository import AnalysisRepository
from .athlete_detector import AthleteDetector
from .athlete_tracker import AthleteTracker
from .analytics import AnalyticsEngine,summarize
from .landmarks import course_position
from .orientation import oriented_frame
from .video import FileVideoSource
from .athlete_tracker import iou
from .course_region import CourseRegion
from .motion_detector import MotionDetector


class AnalysisWorker(QThread):
    packet=Signal(object)
    completed=Signal(object)
    phase=Signal(str)

    def __init__(self,calibration,database,session_id,config,request_id,parent=None,detector_factory=AthleteDetector,source_factory=FileVideoSource):
        super().__init__(parent)
        self.calibration=deepcopy(calibration)
        self.database,self.session_id=database,session_id
        self.config=deepcopy(config)
        self.request_id=request_id
        self.detector_factory=detector_factory
        self.source_factory=source_factory
        self.inputs=queue.Queue(maxsize=1)
        self.dropped=0
        self.input_ended=threading.Event()
        self.queued=calibration.source.get('kind')=='uvc' or config.get('processing_mode')=='realtime'

    def end_input(self):
        self.input_ended.set()

    def submit(self,frame):
        if self.isInterruptionRequested() or self.input_ended.is_set():
            return
        if frame.timestamp_ms+.5<self.calibration.start_ms:
            return
        if frame.timestamp_ms<=self.config.get('live_start_after_ms',-1):
            return
        if self.inputs.full():
            try:
                self.inputs.get_nowait()
                self.dropped+=1
            except queue.Empty:
                pass
        self.inputs.put_nowait(frame)

    def run(self):
        source=None
        identifier=None
        status='stopped'
        error=None
        processed=0
        live=False
        last_timestamp=None
        start=time.perf_counter()
        repository=AnalysisRepository(self.database)
        try:
            motion=self.config.get('detection_backend')=='motion'
            self.calibration.analysis_config.update(self.config)
            self.phase.emit('コース内の移動対象を検出する準備中…' if motion else '人物モデルを検証・読み込み中…')
            detector=(MotionDetector(self.config.get('motion_min_area',3),self.config.get('motion_delta',12)) if motion
                      else self.detector_factory(self.config['model_path'],threshold=.1))
            # Preserve entrance evidence from the first detection. Counting still
            # requires >=3 observations in AnalyticsEngine; early boxes are tentative.
            tracker=AthleteTracker(high=self.config.get('confidence',.35),min_hits=1)
            engine=AnalyticsEngine(self.calibration)
            region=CourseRegion(self.calibration)
            if motion:
                tracker.project=region.project
            course_position(self.calibration,self.calibration.middle[0])
            identifier=repository.begin(self.calibration,self.session_id,self.config,detector.sha256)
            live=self.calibration.source.get('kind')=='uvc'
            if not self.queued:
                source=self.source_factory(self.calibration.source['path'])
                source.seek(self.calibration.start_ms)
            self.phase.emit('移動対象の検出・追跡を実行中' if motion else '人物検出・追跡を実行中（集計は未確認の自動推定を含みます）')
            while not self.isInterruptionRequested():
                if self.queued:
                    try:
                        frame=self.inputs.get(timeout=.2)
                    except queue.Empty:
                        if self.input_ended.is_set():
                            status='completed'
                            break
                        continue
                else:
                    frame=source.read()
                    if frame is None:
                        status='completed'
                        break
                    if frame.timestamp_ms+.5<self.calibration.start_ms:
                        continue
                if last_timestamp is not None and frame.timestamp_ms<=last_timestamp:
                    if self.queued:
                        self.dropped+=1
                        continue
                    raise ValueError('動画時刻が単調増加しません。デコーダーの時刻情報を確認してください')
                if live and frame.timestamp_ms<=self.config.get('live_start_after_ms',-1):
                    continue
                image=oriented_frame(frame.image,self.calibration.rotation,self.calibration.mirrored)
                tick=time.perf_counter()
                if hasattr(detector,'detect_course'):
                    detections=detector.detect_course(image,region,tiled=self.config.get('tiles',True))
                else:
                    detections=detector.detect(image,self.calibration.middle,tiled=self.config.get('tiles',True))
                detections=[d for d in detections if region.can_observe(d.foot)]
                visible=tracker.update(detections,frame.timestamp_ms,can_start=region.can_enter if motion else region.can_start)
                records=engine.update(visible,frame.timestamp_ms,frame.index)
                overlays=[r for r in records if region.contains(r['point']) and (not motion or not r['tentative'])]
                # Once an already tracked skier has crossed out, retain the event
                # evidence, then stop spending inference/tracker work on that ID.
                retired={t.id for t in visible if t.id in engine.track_runs and not region.contains(t.foot) and
                    engine.runs[engine.track_runs[t.id]].transit_state in ('completed','course_out','uncertain')}
                tracker.tracks=[t for t in tracker.tracks if t.id not in retired]
                for detection in detections:
                    if motion:
                        continue  # raw mid-lane foreground specks are not athlete tracks
                    if not region.contains(detection.foot):
                        continue
                    if detection.confidence<self.config.get('confidence',.35):
                        continue
                    if any(iou(detection.bbox,r['bbox'])>.3 for r in records):
                        continue
                    overlays.append({'bbox':detection.bbox,'confidence':detection.confidence,'track_id':'候補',
                                     'run_id':'未確定','lane_id':None,'point':detection.foot,'tentative':True,
                                     'detection_kind':'motion' if motion else 'person'})
                processed+=1
                last_timestamp=frame.timestamp_ms
                runs=engine.snapshot()
                repository.write_frame(identifier,records,runs,processed,self.dropped,last_timestamp)
                frame.source_id=self.request_id
                self.packet.emit({'analysis_id':identifier,'frame':frame,'tracks':records,'overlays':overlays,
                    'detection_count':sum(region.contains(d.foot) and d.confidence>=self.config.get('confidence',.35) for d in detections),
                    'detector_status':getattr(detector,'status','人物モデル'),'runs':runs,
                    'summary':summarize(runs,self.calibration.lanes),'processed':processed,'dropped':self.dropped,
                    'inference_ms':(time.perf_counter()-tick)*1000,'elapsed_seconds':time.perf_counter()-start})
        except Exception as exc:
            status='failed'; error=str(exc)
        finally:
            if source is not None:
                source.close()
            if identifier:
                try:
                    if self.queued:
                        self.dropped+=self.inputs.qsize()
                    repository.finish(identifier,status,error,{'frames_processed':processed,'dropped_frames':self.dropped,'last_timestamp_ms':last_timestamp})
                except Exception as exc:
                    status='failed'; error=f'{error or ""} 保存エラー: {exc}'
        self.completed.emit({'id':identifier,'status':status,'error':error,'processed':processed,'dropped':self.dropped})
