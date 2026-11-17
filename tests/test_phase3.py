"""Phase 3: synthetic detections test logic; real-model tests are separately marked."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sqlite3
import json
import time
import cv2
import numpy as np
import pytest
from PySide6.QtCore import QTimer,Qt
from mogul.analytics import AnalyticsEngine,summarize
from mogul.athlete_detector import AthleteDetector,Detection,DEFAULT_MODEL
from mogul.athlete_tracker import AthleteTracker,Track,assignment
from mogul.analysis_repository import AnalysisRepository
from mogul.analysis_worker import AnalysisWorker
from mogul.app import MainWindow
from mogul.landmarks import ControlGateEditor,BumpRegistry
from mogul.models import Calibration
from mogul.video import VideoFrame


@pytest.fixture
def course():
    c=Calibration(middle=[[.05,.05],[.95,.05],[.95,.95],[.05,.95]],middle_confirmed=True)
    c.configure_lanes(2)
    for i,j,p in [(0,0,[.2,.05]),(0,3,[.2,.95]),(1,1,[.8,.05]),(1,2,[.8,.95])]:
        c.move_lane_corner(i,j,p)
    for lane in c.lanes:
        lane.confirmed=True
    gates=ControlGateEditor(c)
    gate=gates.add('LEFT',[.12,.3])
    gates.update(gate.id,'LEFT',3)
    gates.confirm()
    registry=BumpRegistry(c)
    for lane,x in [('A',.35),('B',.65)]:
        bump=registry.add(lane,[[x-.03,.38],[x+.03,.38],[x+.03,.42],[x-.03,.42]],3)
        registry.update(bump.id,lane,3,1,'UNKNOWN')
    registry.confirm()
    return c


def detection(x,y,confidence=.9):
    return Detection([x-.02,y-.06,x+.02,y],confidence)


def track(identifier,x,y,t,velocity=(0,.1)):
    return Track(identifier,detection(x,y).bbox,.9,t,hits=3,velocity=list(velocity))


@pytest.mark.parametrize('matrix,expected',[
    ([[1,2],[1.1,9]],{(0,1),(1,0)}),
    ([[9,1,9],[1,9,9]],{(0,1),(1,0)}),
    ([[9,1],[1,9],[8,8]],{(0,1),(1,0)}),
    (np.zeros((0,2)),set())])
def test_global_assignment(matrix,expected):
    assert set(assignment(matrix))==expected


def test_tracker_low_confidence_recovery_and_timeout():
    tracker=AthleteTracker(min_hits=2)
    assert not tracker.update([detection(.35,.2,.2)],0)  # low confidence cannot create ID
    assert not tracker.update([detection(.35,.2)],100)
    visible=tracker.update([detection(.35,.21,.2)],200)
    assert len(visible)==1 and visible[0].id==1
    assert not tracker.update([],300)
    assert tracker.update([detection(.35,.23)],500)[0].id==1
    assert not tracker.update([detection(.35,.7)],3000)
    assert tracker.update([detection(.35,.71)],3100)[0].id==2


def test_tracker_two_crossing_trajectories():
    tracker=AthleteTracker(min_hits=1)
    first=tracker.update([detection(.3,.4),detection(.7,.4)],0)
    left,right=first[0].id,first[1].id
    for step,(a,b) in enumerate([(.35,.65),(.4,.6),(.46,.54),(.54,.46),(.6,.4)],1):
        tracker.update([detection(a,.4),detection(b,.4)],step*200)
    by_id={t.id:t for t in tracker.tracks}
    assert by_id[left].foot[0]>.5 and by_id[right].foot[0]<.5


def test_count_downhill_only_and_unique_bump_passages(course):
    engine=AnalyticsEngine(course)
    for i,y in enumerate([.15,.2,.25,.3,.35,.45,.5]):
        engine.update([track(1,.35,y,i*200)],i*200,i)
    summary=summarize(engine.snapshot(),course.lanes)
    assert summary['unique_runs']==1 and summary['starts']['A']==1 and summary['passes']['A']==1
    run=next(iter(engine.runs.values()))
    assert len(run.bumps)==1 and run.bumps[0]['bump_id']==course.bumps[0].id
    for i in range(8,12):
        engine.update([track(1,.35,.4,i*200)],i*200,i)
    assert len(run.bumps)==1


@pytest.mark.parametrize('x,ys',[(.35,[.4]*8),(.12,[.15,.2,.25,.3,.35,.4,.45,.5]),(.35,[.8,.75,.7,.65,.6,.55,.5,.45])])
def test_stationary_margin_and_uphill_not_counted(course,x,ys):
    engine=AnalyticsEngine(course)
    for i,y in enumerate(ys):
        engine.update([track(1,x,y,i*200)],i*200,i)
    assert summarize(engine.snapshot(),course.lanes)['unique_runs']==0


def test_partial_entry_and_lane_change_no_double_count(course):
    engine=AnalyticsEngine(course)
    points=[(.35,.2),(.35,.25),(.35,.3),(.35,.35),(.51,.38),(.49,.4),
            (.65,.45),(.65,.5),(.65,.55),(.65,.6),(.65,.7)]
    for i,(x,y) in enumerate(points):
        engine.update([track(1,x,y,i*200)],i*200,i)
    s=summarize(engine.snapshot(),course.lanes)
    assert s['unique_runs']==1 and s['course_changes']==1
    assert s['starts']=={'A':1,'B':0} and s['passes']=={'A':1,'B':1}
    assert s['groups']['blue']==1 and s['groups']['red']==1
    partial=AnalyticsEngine(course)
    for i,y in enumerate([.5,.55,.6,.65]):
        partial.update([track(1,.35,y,i*200)],i*200,i)
    s=summarize(partial.snapshot(),course.lanes)
    assert s['unique_runs']==1 and s['starts']['A']==0 and s['passes']['A']==1


def test_run_relink_and_ambiguity(course):
    engine=AnalyticsEngine(course)
    for i,y in enumerate([.15,.2,.25,.3]):
        engine.update([track(1,.35,y,i*200)],i*200,i)
    identifier=next(iter(engine.runs))
    engine.update([track(2,.35,.48,2400)],2400,24)
    assert len(engine.runs)==1 and engine.runs[identifier].track_ids==[1,2]
    assert engine.runs[identifier].relinked
    assert summarize(engine.snapshot(),course.lanes)['unique_runs']==1
    ambiguous=AnalyticsEngine(course)
    ambiguous.update([track(1,.34,.3,0,velocity=(0,0)),track(2,.36,.3,0,velocity=(0,0))],0,0)
    ambiguous.update([track(3,.35,.3,2000,velocity=(0,0))],2000,20)
    assert len(ambiguous.runs)==3  # do not guess which person after an ambiguous gap


def test_uphill_visit_to_second_lane_is_not_a_pass(course):
    engine=AnalyticsEngine(course)
    points=[(.35,.15),(.35,.2),(.35,.25),(.35,.3),(.35,.4),(.35,.45),
            (.65,.6),(.65,.58),(.65,.56),(.65,.53),(.65,.5)]
    for i,(x,y) in enumerate(points):
        engine.update([track(1,x,y,i*200)],i*200,i)
    summary=summarize(engine.snapshot(),course.lanes)
    assert summary['unique_runs']==1 and summary['passes']=={'A':1,'B':0}


def test_yolox_decode_person_nms_and_shape():
    detector=AthleteDetector.__new__(AthleteDetector)
    detector.anchors=np.zeros((8400,2)); detector.strides=np.full((8400,1),8)
    detector.threshold=.1; detector.nms=.45
    output=np.zeros((8400,85),np.float32)
    output[:2,:4]=[2,2,np.log(2),np.log(4)]
    output[0,4:6]=[.9,.8]; output[1,4:6]=[.8,.8]
    output[2,:4]=[10,10,1,1]; output[2,4]=.9; output[2,5]=.2; output[2,7]=.9
    result=detector.decode(output,1,(0,0),(640,640))
    assert len(result)==1 and result[0].confidence==pytest.approx(.72)
    assert 0<=min(result[0].bbox)<=max(result[0].bbox)<=1
    with pytest.raises(ValueError):
        detector.decode(np.zeros((3,85)),1,(0,0),(640,640))


class FixtureDetector:
    """Synthetic detection fixture, never used by the production UI."""
    sha256='synthetic-test-only'
    def __init__(self,*args,**kwargs):
        self.calls=0
    def detect(self,image,middle,tiled=False):
        self.calls+=1
        return [detection(.35,.15+self.calls*.025)]


class FixtureSource:
    instances=[]
    def __init__(self,path):
        self.index=0; self.closed=False
        self.instances.append(self)
    def seek(self,timestamp):
        self.index=int(np.ceil(timestamp/100))
    def read(self):
        if self.index>=12:
            return None
        frame=VideoFrame(np.zeros((100,160,3),np.uint8),self.index*100,self.index)
        self.index+=1
        return frame
    def close(self):
        self.closed=True


def test_file_worker_sequential_start_persistence(qtbot,tmp_path,course):
    course.source={'kind':'file','path':'fixture.mp4'}
    course.start_ms=300
    path=tmp_path/'analysis.sqlite'
    job=AnalysisWorker(course,path,None,{'model_path':'fixture'},42,detector_factory=FixtureDetector,source_factory=FixtureSource)
    packets=[]; results=[]
    job.packet.connect(packets.append); job.completed.connect(results.append)
    job.start()
    qtbot.waitUntil(lambda:bool(results),timeout=5000)
    assert job.wait(1000)
    assert results[0]['status']=='completed' and results[0]['processed']==9
    assert [p['frame'].index for p in packets]==list(range(3,12))
    assert all(p['frame'].source_id==42 for p in packets)
    assert FixtureSource.instances[-1].closed
    data=AnalysisRepository(path).load_analysis(results[0]['id'])
    assert data['frames_processed']==9 and data['status']=='completed'
    assert data['model_sha256']=='synthetic-test-only'
    assert summarize(data['runs'],course.lanes)['unique_runs']==1


def test_live_queue_drops_are_explicit_and_stoppable(qtbot,tmp_path,course):
    course.source={'kind':'uvc','device':'fixture'}
    job=AnalysisWorker(course,tmp_path/'live.sqlite',None,{'model_path':'fixture','live_start_after_ms':500},2,detector_factory=FixtureDetector)
    packets=[]; results=[]
    job.packet.connect(packets.append); job.completed.connect(results.append)
    for i in range(20):
        job.submit(VideoFrame(np.zeros((100,160,3),np.uint8),i*100,i,2))
    assert job.dropped==13  # timestamps <= the start cutoff are not analysis inputs
    job.start()
    qtbot.waitUntil(lambda:bool(packets),timeout=5000)
    job.requestInterruption()
    assert job.wait(3000)
    qtbot.waitUntil(lambda:bool(results),timeout=1000)
    assert packets[0]['frame'].timestamp_ms>500
    assert results[0]['status']=='stopped' and results[0]['dropped']==13
    saved=AnalysisRepository(tmp_path/'live.sqlite').load_analysis(results[0]['id'])
    assert saved['dropped_frames']==13


def test_missing_or_corrupt_model_is_explicit(qtbot,tmp_path,course):
    course.source={'kind':'file','path':'unused.mp4'}
    with pytest.raises(ValueError,match='未導入'):
        AthleteDetector(tmp_path/'missing.onnx')
    bad=tmp_path/'bad.onnx'; bad.write_bytes(b'not an ONNX file')
    with pytest.raises(ValueError,match='SHA256'):
        AthleteDetector(bad)
    job=AnalysisWorker(course,tmp_path/'missing.sqlite',None,{'model_path':str(tmp_path/'none.onnx')},1)
    results=[]; job.completed.connect(results.append); job.start()
    qtbot.waitUntil(lambda:bool(results),timeout=3000)
    assert job.wait(1000)
    assert results[0]['status']=='failed' and results[0]['id'] is None


def make_analysis(tmp_path,course):
    repo=AnalysisRepository(tmp_path/'reviews.sqlite')
    identifier=repo.begin(course,None,{},'fixture')
    engine=AnalyticsEngine(course)
    records=[]
    for i,y in enumerate([.15,.2,.25,.3,.35]):
        records=engine.update([track(1,.35,y,i*200),track(2,.65,y,i*200)],i*200,i)
        repo.write_frame(identifier,records,engine.snapshot(),i+1,0,i*200)
    repo.finish(identifier,'completed')
    return repo,identifier,engine.snapshot()


def test_manual_reviews_merges_counts_and_immutable_tracks(tmp_path,course):
    repo,identifier,runs=make_analysis(tmp_path,course)
    assert summarize(runs,course.lanes)['unique_runs']==2
    a,b=runs
    with pytest.raises(ValueError):
        repo.review(identifier,a['id'],{'qualified':False},'')
    repo.review(identifier,b['id'],{'merged_into':a['id'],'review':'confirmed'},'追跡IDが分断されたため同一滑走へ統合')
    data=repo.load_analysis(identifier)
    assert summarize(data['runs'],course.lanes)['unique_runs']==1
    assert len({r['run_id'] for r in repo.tracks_at(identifier,4)})==1
    with sqlite3.connect(repo.path) as db:
        raw=[json.loads(row[0]) for row in db.execute('SELECT data_json FROM tracks WHERE frame_index=4')]
        assert len({r['run_id'] for r in raw})==2
    with pytest.raises(ValueError):
        repo.review(identifier,a['id'],{'merged_into':b['id']},'循環を拒否')
    repo.review(identifier,a['id'],{'start_lane':'B','passed_lanes':['B'],'review':'confirmed','change_count_override':2},'レーンを手動訂正')
    data=repo.load_analysis(identifier)
    assert summarize(data['runs'],course.lanes)['course_changes']==2
    repo.review(identifier,a['id'],{'review':'excluded'},'観客なので除外')
    assert summarize(repo.load_analysis(identifier)['runs'],course.lanes)['unique_runs']==0
    manual=repo.add_manual(identifier,'A',1200,'見逃した選手を追加')
    assert manual and summarize(repo.load_analysis(identifier)['runs'],course.lanes)['unique_runs']==1
    with sqlite3.connect(repo.path) as db:
        assert db.execute('SELECT COUNT(*) FROM run_reviews').fetchone()[0]==4


def test_schema2_backup_upgrade(tmp_path,course):
    repo=AnalysisRepository(tmp_path/'schema2.sqlite')
    repo.save(course)
    with sqlite3.connect(repo.path) as db:
        for table in ('run_reviews','tracks','runs','analysis_sessions'):
            db.execute(f'DROP TABLE {table}')
        db.execute('PRAGMA user_version=2')
    repo.begin(course,None,{},'fixture')
    backup=Path(str(repo.path)+'.v2.bak')
    assert backup.is_file()
    with sqlite3.connect(backup) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0]==2
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='analysis_sessions'").fetchall()


def test_manual_merge_can_be_reversed(tmp_path,course):
    repo,identifier,runs=make_analysis(tmp_path,course)
    a,b=runs
    repo.review(identifier,b['id'],{'merged_into':a['id']},'同一滑走として統合')
    assert summarize(repo.load_analysis(identifier)['runs'],course.lanes)['unique_runs']==1
    repo.review(identifier,b['id'],{'merged_into':None,'review':'confirmed'},'別選手だったため統合解除')
    assert summarize(repo.load_analysis(identifier)['runs'],course.lanes)['unique_runs']==2


def test_live_never_analyzes_pre_start_frames(qtbot,tmp_path,course):
    course.source={'kind':'uvc','device':'fixture'}
    job=AnalysisWorker(course,tmp_path/'cutoff.sqlite',None,{'model_path':'fixture','live_start_after_ms':500},3,detector_factory=FixtureDetector)
    phases=[]; packets=[]
    job.phase.connect(phases.append); job.packet.connect(packets.append)
    job.start()
    qtbot.waitUntil(lambda:any('実行中' in text for text in phases),timeout=3000)
    job.submit(VideoFrame(np.zeros((100,160,3),np.uint8),400,4,3))
    qtbot.wait(50)
    assert not packets and job.dropped==0
    job.submit(VideoFrame(np.zeros((100,160,3),np.uint8),600,6,3))
    qtbot.waitUntil(lambda:bool(packets),timeout=3000)
    assert packets[0]['frame'].timestamp_ms==600
    job.requestInterruption(); assert job.wait(3000)


@pytest.fixture
def window(qtbot,tmp_path,course):
    path=tmp_path/'video.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,100))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(np.zeros((100,160,3),np.uint8))
    writer.release()
    w=MainWindow(tmp_path/'gui.sqlite',startup=False)
    qtbot.addWidget(w,before_close_func=lambda w:setattr(w.workflow,'dirty',False))
    w.workflow.calibration=course
    w.show()
    w.open_source({'kind':'file','path':str(path)})
    qtbot.waitUntil(lambda:w.source_ok and w.current_frame is not None,timeout=8000)
    yield w
    if w.analysis_active:
        w.stop_analysis(); w.analysis_job.wait(5000); qtbot.wait(30)
    w.workflow.dirty=False
    w.worker.requestInterruption()
    assert w.worker.wait(5000)
    w.close()


def test_gui_analysis_results_manual_review_and_replay(qtbot,window,tmp_path,monkeypatch):
    import mogul.app as app
    original=AnalysisWorker
    monkeypatch.setattr(app,'AnalysisWorker',lambda *args,**kwargs:original(*args,**kwargs,detector_factory=FixtureDetector))
    model=tmp_path/'fixture.onnx'; model.write_bytes(b'synthetic test only')
    window.analysis_panel.model.setText(str(model))
    ticks=[]
    timer=QTimer(window); timer.timeout.connect(lambda:ticks.append(1)); timer.start(10)
    window.start_analysis()
    assert window.analysis_active and not window.right_tabs.isTabEnabled(0)
    window.course_map.selected.emit('bump',window.workflow.calibration.bumps[0].id)
    assert window.view.mode=='analysis'  # map selection must not unlock coordinate editing
    qtbot.waitUntil(lambda:not window.analysis_active,timeout=8000)
    qtbot.waitUntil(lambda:window.analysis_result is not None,timeout=1000)
    timer.stop()
    panel=window.analysis_panel
    assert ticks and window.analysis_result['status']=='completed'
    assert 'ユニーク滑走 1' in panel.summary.text()
    assert window.right_tabs.isTabEnabled(0) and panel.runs.count()==1
    panel.runs.setCurrentRow(0)
    panel.reason.setText('検証用の滑走確認')
    panel.review_state.setCurrentIndex(panel.review_state.findData('confirmed'))
    panel.apply_review()
    assert panel.data['runs'][0]['review']=='confirmed'
    panel.reason.setText('検証用の見逃し追加')
    panel.add_run()
    assert len(panel.data['runs'])==2 and 'ユニーク滑走 2' in panel.summary.text()
    panel.runs.setCurrentRow(0)
    panel.open_video()
    qtbot.waitUntil(lambda:window.source_ok,timeout=8000)
    assert panel.review_analysis_id and window.metadata['kind']=='file'
    window.worker.command('play',False)


def test_gui_corrupt_model_does_not_show_previous_counts(qtbot,window,tmp_path):
    bad=tmp_path/'bad.onnx'; bad.write_bytes(b'corrupt')
    panel=window.analysis_panel
    panel.model.setText(str(bad))
    panel.summary.setText('old results')
    window.start_analysis()
    qtbot.waitUntil(lambda:not window.analysis_active,timeout=5000)
    assert window.analysis_result['status']=='failed'
    assert panel.data is None and panel.runs.count()==0
    assert panel.summary.text()=='集計なし' and 'SHA256' in panel.progress.text()


def test_gui_counts_without_any_gate_or_bump(qtbot,window,tmp_path,monkeypatch):
    import mogul.app as app
    monkeypatch.setattr(app,'AnalysisWorker',lambda *args,**kwargs:AnalysisWorker(*args,**kwargs,detector_factory=FixtureDetector))
    c=window.workflow.calibration
    c.control_gates=[]; c.bumps=[]
    c.gates_confirmed=c.bumps_confirmed=False
    model=tmp_path/'fixture.onnx'; model.write_bytes(b'test fixture')
    window.analysis_panel.model.setText(str(model))
    window.refresh()
    assert window.analyze_button.isEnabled()
    window.start_analysis()
    qtbot.waitUntil(lambda:not window.analysis_active,timeout=8000)
    assert window.analysis_result['status']=='completed'
    assert 'ユニーク滑走 1' in window.analysis_panel.summary.text()
    assert not window.workflow.calibration.bumps and not window.workflow.calibration.control_gates


@pytest.mark.skipif(not DEFAULT_MODEL.is_file(),reason='optional real model: run tools.install_person_model')
def test_real_yolox_blank_cpu_inference():
    detector=AthleteDetector()
    result=detector.detect(np.zeros((144,256,3),np.uint8))
    assert result==[] and len(detector.sha256)==64
