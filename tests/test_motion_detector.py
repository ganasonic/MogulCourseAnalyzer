import cv2
import numpy as np
from mogul.models import Calibration
from mogul.course_region import CourseRegion
from mogul.motion_detector import MotionDetector
from mogul.athlete_tracker import AthleteTracker
from mogul.analytics import AnalyticsEngine,summarize
from mogul.app import MainWindow
from mogul.video import VideoFrame
from mogul.analysis_worker import AnalysisWorker
from mogul.analysis_repository import AnalysisRepository


def course():
    c=Calibration(middle=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],middle_confirmed=True)
    c.configure_lanes(2)
    for lane in c.lanes:lane.confirmed=True
    c.analysis_config={'detection_backend':'motion'}
    return c


def moving_frames():
    for index in range(58):
        image=np.full((100,160,3),180,np.uint8)
        if 10<=index<57:
            bottom=4+(index-10)*2
            cv2.rectangle(image,(37,bottom-7),(43,bottom),(20,20,20),-1)
        yield image


def test_nonhuman_rectangle_enters_and_completes_without_model():
    c=course();region=CourseRegion(c);detector=MotionDetector()
    tracker=AthleteTracker(min_hits=1);engine=AnalyticsEngine(c)
    observed=[]
    for index,image in enumerate(moving_frames()):
        objects=detector.detect_course(image,region)
        tracks=tracker.update(objects,index*40,can_start=region.can_start)
        engine.update(tracks,index*40,index);observed.extend(objects)
    assert observed
    assert summarize(engine.snapshot(),c.lanes)['completions']=={'A':1,'B':0}
    assert not any(r['events'] for r in engine.snapshot())


def test_static_snow_and_uniform_light_change_do_not_become_skiers():
    c=course();region=CourseRegion(c);detector=MotionDetector()
    image=np.full((100,160,3),150,np.uint8)
    cv2.rectangle(image,(37,30),(43,45),(20,20,20),-1)
    assert detector.detect_course(image,region)==[]
    for amount in [0,0,10,20,30,20,0]:
        adjusted=np.clip(image.astype(int)+amount,0,255).astype(np.uint8)
        assert detector.detect_course(adjusted,region)==[]


def test_camera_translation_on_static_texture_is_compensated():
    c=course();region=CourseRegion(c);detector=MotionDetector()
    image=np.random.default_rng(11).integers(30,220,(200,320,3),dtype=np.uint8)
    assert detector.detect_course(image,region)==[]
    for dx,dy in [(1,0),(2,1),(1,1),(0,0)]:
        moved=cv2.warpAffine(image,np.float32([[1,0,dx],[0,1,dy]]),(320,200),borderMode=cv2.BORDER_REFLECT)
        assert detector.detect_course(moved,region)==[]


def test_motion_worker_saves_completed_run_without_model(qtbot,tmp_path):
    images=list(moving_frames())
    class Source:
        def __init__(self,*args):self.index=0
        def seek(self,*args):pass
        def read(self):
            if self.index>=len(images):return None
            i=self.index;self.index+=1
            return VideoFrame(images[i],i*40,i)
        def close(self):pass
    c=course();c.source={'kind':'file','path':'synthetic rectangle'}
    database=tmp_path/'motion.sqlite'
    config={'detection_backend':'motion','processing_mode':'all_frames'}
    job=AnalysisWorker(c,database,None,config,1,source_factory=Source)
    result=[];job.completed.connect(result.append);job.start()
    qtbot.waitUntil(lambda:bool(result),timeout=5000);assert job.wait(1000)
    assert result[0]['status']=='completed'
    saved=AnalysisRepository(database).load_analysis(result[0]['id'])
    assert summarize(saved['runs'],c.lanes)['completions']['A']==1
    assert saved['config']['detection_backend']=='motion'


def test_gui_default_motion_needs_no_onnx_file(qtbot,tmp_path):
    w=MainWindow(tmp_path/'gui.sqlite',startup=False)
    qtbot.addWidget(w,before_close_func=lambda _:setattr(w.workflow,'dirty',False))
    w.workflow.calibration=course();w.source_ok=True
    w.analysis_panel.model.setText(str(tmp_path/'missing.onnx'))
    w.refresh()
    assert w.analysis_panel.config()['detection_backend']=='motion'
    assert w.analyze_button.isEnabled() and not w.analysis_panel.requires_model
    w.analysis_panel.detection_backend.setCurrentIndex(1);w.refresh()
    assert not w.analyze_button.isEnabled()
    w.workflow.dirty=False;w.worker.requestInterruption();assert w.worker.wait(3000);w.close()


def test_motion_acquisition_is_limited_to_entrance_band():
    region=CourseRegion(course())
    assert region.can_enter([.25,.08]) and region.can_enter([.25,.12])
    assert not region.can_enter([.25,.5]) and not region.can_enter([.95,.12])


def test_motion_entrance_zone_handles_first_blob_just_inside_top():
    from mogul.athlete_tracker import Track
    c=course();engine=AnalyticsEngine(c)
    for i,y in enumerate([.12,.14,.18,.25,.35,.45,.55,.65,.75,.85,.92]):
        engine.update([Track(1,[.23,y-.04,.27,y],1.,i*100,hits=i+1)],i*100,i)
    assert summarize(engine.snapshot(),c.lanes)['completions']['A']==1


def test_projected_motion_tracker_does_not_jump_to_neighboring_lane():
    from mogul.athlete_detector import Detection
    c=Calibration(middle=[[.47,.1],[.53,.1],[.58,.9],[.42,.9]],middle_confirmed=True)
    c.configure_lanes(4);region=CourseRegion(c)
    tracker=AthleteTracker(min_hits=1,project=region.project)
    first=tracker.update([Detection([.4765,.1,.4785,.105],1.)],0)
    second=tracker.update([Detection([.4915,.1,.4935,.105],1.)],40)
    assert first[0].id!=second[0].id


def test_gui_motion_video_counts_rectangle_without_model(qtbot,tmp_path):
    path=tmp_path/'moving-rectangle.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25,(160,100))
    assert writer.isOpened()
    for image in moving_frames():writer.write(image)
    writer.release()
    w=MainWindow(tmp_path/'motion-gui.sqlite',startup=False)
    qtbot.addWidget(w,before_close_func=lambda _:setattr(w.workflow,'dirty',False))
    w.workflow.calibration=course();w.show();w.open_source({'kind':'file','path':str(path)})
    qtbot.waitUntil(lambda:w.source_ok and w.current_frame is not None,timeout=5000)
    w.analysis_panel.model.setText(str(tmp_path/'missing.onnx'))
    w.analysis_panel.processing_mode.setCurrentIndex(1)
    w.start_analysis()
    qtbot.waitUntil(lambda:not w.analysis_active,timeout=10000)
    assert w.analysis_result['status']=='completed'
    assert w.lane_total_labels['A'].text().endswith('1')
    assert w.lane_total_labels['B'].text().endswith('0')
    assert not w.analysis_panel.requires_model
    w.workflow.dirty=False;w.worker.requestInterruption();assert w.worker.wait(3000);w.close()
