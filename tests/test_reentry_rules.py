from pathlib import Path
from copy import deepcopy
import numpy as np
import cv2
import pytest
from mogul.models import Calibration
from mogul.analytics import AnalyticsEngine,summarize
from mogul.athlete_tracker import Track
from mogul.course_region import CourseRegion,visible_track
from mogul.analysis_repository import AnalysisRepository
from mogul.app import MainWindow


def course():
    c=Calibration(middle=[[.05,.05],[.95,.05],[.95,.95],[.05,.95]],middle_confirmed=True)
    c.configure_lanes(2)
    for i,j,p in [(0,0,[.2,.1]),(0,3,[.2,.9]),(0,1,[.45,.1]),(0,2,[.45,.9]),
                  (1,0,[.55,.1]),(1,3,[.55,.9]),(1,1,[.8,.1]),(1,2,[.8,.9])]:
        assert c.move_lane_corner(i,j,p)
    for lane in c.lanes:lane.confirmed=True
    c.analysis_config={'detection_backend':'motion'}
    return c


def feed(engine,points,start=0,step=200,identity=1):
    for i,(x,y) in enumerate(points):
        t=start+i*step
        engine.update([Track(identity,[x-.02,y-.06,x+.02,y],1.,t,hits=5)],t,int(t/step))


def test_original_lane_return_resumes_and_counts_once():
    c=course();e=AnalyticsEngine(c)
    feed(e,[(.35,.08),(.35,.12),(.35,.2),(.4,.3),(.44,.35),(.48,.4),(.5,.43),(.48,.45),(.44,.48),(.4,.55),(.35,.65),(.35,.78),(.35,.88),(.35,.92)])
    run=e.snapshot()[0]
    assert run['entry_lane']=='A' and run['completed_lane']=='A'
    assert run['entry_kind']=='top' and len(run['resumes'])==1
    assert any(v['kind']=='course_out' for v in run['events'])
    assert summarize(e.snapshot(),c.lanes)['completions']=={'A':1,'B':0}
    feed(e,[(.35,.93)],start=3000)
    assert summarize(e.snapshot(),c.lanes)['completions']['A']==1


def test_other_lane_is_never_counted_as_original_return():
    c=course();e=AnalyticsEngine(c)
    feed(e,[(.35,.08),(.35,.12),(.35,.2),(.4,.3),(.44,.35),(.48,.4),(.52,.45),(.56,.5),(.6,.6),(.65,.7),(.65,.85),(.65,.92)])
    assert summarize(e.snapshot(),c.lanes)['completions']=={'A':0,'B':0}
    assert not e.snapshot()[0]['resumes']


def test_side_arrival_has_events_but_never_completion():
    c=course();e=AnalyticsEngine(c)
    feed(e,[(.58,.3),(.6,.35),(.65,.4),(.7,.45),(.75,.5),(.79,.55),(.83,.6)])
    run=e.snapshot()[0]
    assert run['entry_kind']=='side' and run['completed_lane'] is None
    assert any(v['kind']=='course_out' for v in run['events'])
    assert summarize(e.snapshot(),c.lanes)['completions']['B']==0


def test_bottom_jump_stays_uncounted_after_turning_downhill():
    c=course();e=AnalyticsEngine(c)
    feed(e,[(.35,.93),(.35,.88),(.35,.8),(.35,.7),(.35,.6),(.35,.7),(.35,.8),(.35,.92)])
    run=e.snapshot()[0]
    assert run['entry_kind']=='bottom' and run['entry_lane'] is None
    assert summarize(e.snapshot(),c.lanes)['completions']['A']==0


def test_existing_skier_in_gap_is_visible_but_gap_does_not_acquire_staff():
    c=course();region=CourseRegion(c)
    point=[.5,.5]
    assert region.tracking_contains(point) and region.can_observe(point)
    assert not region.contains(point) and not region.can_start(point)
    assert visible_track(c,{'point':point,'course_tracking':True})
    assert not visible_track(c,{'point':point})


def test_side_arrival_abrupt_stop_and_spread_get_independent_candidates():
    c=course();e=AnalyticsEngine(c)
    feed(e,[(.65,.3),(.65,.34),(.65,.38),(.65,.42),(.65,.46),(.65,.5)])
    for i in range(8):
        t=1200+i*200
        e.update([Track(1,[.58,.46,.72,.5],1.,t,hits=20)],t,6+i)
    run=e.snapshot()[0];kinds={v['kind'] for v in run['events']}
    assert {'stumble_candidate','fall_candidate'}<=kinds
    assert run['completed_lane'] is None and all(v['review']=='unreviewed' for v in run['events'])


def test_reentry_events_and_origin_restore(tmp_path):
    c=course();e=AnalyticsEngine(c)
    feed(e,[(.35,.08),(.35,.12),(.35,.2),(.4,.3),(.44,.35),(.48,.4),(.44,.48),(.35,.6),(.35,.75),(.35,.88),(.35,.92)])
    repo=AnalysisRepository(tmp_path/'resume.sqlite');aid=repo.begin(c,None,{},'motion')
    repo.write_frame(aid,[],e.snapshot(),11,0,2000);repo.finish(aid,'completed')
    saved=repo.load_analysis(aid)['runs'][0]
    assert saved['entry_kind']=='top' and saved['resumes'] and saved['completed_lane']=='A'


def test_save_as_uses_selected_names_and_keeps_original(qtbot,tmp_path,monkeypatch):
    import mogul.app as app
    original=tmp_path/'original.sqlite';w=MainWindow(original,startup=False)
    qtbot.addWidget(w,before_close_func=lambda _:setattr(w.workflow,'dirty',False))
    w.workflow.calibration=course();assert w.save();before=original.read_bytes()
    paths=iter([tmp_path/'朝のコース.sqlite',tmp_path/'午後のコース.sqlite'])
    suggestions=[]
    def choose(*args):
        suggestions.append(args[2]);return str(next(paths)),''
    monkeypatch.setattr(app.QFileDialog,'getSaveFileName',choose)
    assert w.save_as() and w.database.name=='朝のコース.sqlite'
    assert w.save_as() and w.database.name=='午後のコース.sqlite'
    assert suggestions[0]!=suggestions[1] and original.read_bytes()==before
    assert (tmp_path/'朝のコース.sqlite').exists() and (tmp_path/'午後のコース.sqlite').exists()
    w.workflow.dirty=False;w.worker.requestInterruption();assert w.worker.wait(3000);w.close()


def test_overall_course_exit_is_marked_separately():
    c=course();e=AnalyticsEngine(c)
    points=[(.35,.08),(.35,.12),(.35,.2),(.4,.3),(.44,.35),(.48,.4)]
    points += [(x,.45) for x in [.52,.56,.60,.64,.68,.72,.76,.80,.84,.88,.92,.94,.97]]
    feed(e,points)
    run=e.snapshot()[0]
    assert run['transit_state']=='out_of_course'
    exit_event=next(v for v in run['events'] if v['kind']=='course_exit')
    assert abs(exit_event['point'][0]-.95)<1e-5
    assert run['completed_lane'] is None


def test_uncounted_side_run_supports_manual_fall_annotation(tmp_path):
    c=course();e=AnalyticsEngine(c);repo=AnalysisRepository(tmp_path/'side-events.sqlite')
    aid=repo.begin(c,None,{},'motion')
    for i,y in enumerate([.3,.35,.4,.45,.5]):
        records=e.update([Track(1,[.63,y-.06,.67,y],1.,i*200,hits=5)],i*200,i)
        repo.write_frame(aid,records,e.snapshot(),i+1,0,i*200)
    repo.finish(aid,'completed');run=e.snapshot()[0]
    with pytest.raises(ValueError):repo.add_event(aid,run['id'],'fall_candidate',400,'')
    repo.add_event(aid,run['id'],'fall_candidate',400,'映像で転倒を確認')
    data=repo.load_analysis(aid)
    event=data['runs'][0]['events'][-1]
    assert event['review']=='confirmed' and event['kind']=='fall_candidate'
    assert event['point']==[.65,.4]
    assert summarize(data['runs'],c.lanes)['completions']['B']==0


def test_cancel_save_as_leaves_database_and_dirty_state(qtbot,tmp_path,monkeypatch):
    import mogul.app as app
    original=tmp_path/'working.sqlite';w=MainWindow(original,startup=False)
    qtbot.addWidget(w,before_close_func=lambda _:setattr(w.workflow,'dirty',False))
    w.workflow.calibration=course();w.workflow.dirty=True
    monkeypatch.setattr(app.QFileDialog,'getSaveFileName',lambda *args:('',''))
    assert not w.save_as() and w.database==original and w.workflow.dirty
    assert not original.exists()
    w.workflow.dirty=False;w.worker.requestInterruption();assert w.worker.wait(3000);w.close()


def test_trapezoid_course_exit_marker_lies_on_actual_boundary():
    c=Calibration(middle=[[.3,.1],[.7,.1],[.9,.9],[.1,.9]],middle_confirmed=True)
    c.configure_lanes(2);e=AnalyticsEngine(c)
    matrix=cv2.getPerspectiveTransform(np.asarray(c.middle,np.float32),np.array([[0,0],[1,0],[1,1],[0,1]],np.float32))
    points=cv2.perspectiveTransform(np.array([[[.25,-.02],[.25,.03],[.25,.2],[.25,.4],[.1,.5],[-.05,.6]]],np.float32),np.linalg.inv(matrix))[0]
    feed(e,points)
    event=next(v for r in e.snapshot() for v in r['events'] if v['kind']=='course_exit')
    projected=cv2.perspectiveTransform(np.asarray([[event['point']]],np.float32),matrix)[0,0]
    assert abs(projected[0])<1e-5


def test_live_packet_does_not_reset_counter_or_event_list(qtbot,tmp_path):
    from types import SimpleNamespace
    c=course();e=AnalyticsEngine(c)
    feed(e,[(.35,.08),(.35,.12),(.35,.2),(.4,.3),(.44,.35),(.48,.4),(.44,.48),(.35,.6),(.35,.75),(.35,.88),(.35,.92)])
    w=MainWindow(tmp_path/'live-ui.sqlite',startup=False)
    qtbot.addWidget(w,before_close_func=lambda _:setattr(w.workflow,'dirty',False));w.workflow.calibration=c
    packet={'analysis_id':'fixture','runs':e.snapshot(),'summary':summarize(e.snapshot(),c.lanes),
            'tracks':[],'processed':11,'dropped':0,'inference_ms':1,'frame':SimpleNamespace(timestamp_ms=2200)}
    w.update_analysis_events(e.snapshot(),2200)
    for _ in range(2):
        w.analysis_panel.show_packet(packet)
        assert w.lane_total_labels['A'].text().endswith('1')
        assert w.analysis_panel.event_list.count()==1 and len(w.view.analysis_events)==1
    w.analysis_panel.reset_result()
    assert w.lane_total_labels['A'].text().endswith('0') and w.analysis_panel.event_list.count()==0
    w.workflow.dirty=False;w.worker.requestInterruption();assert w.worker.wait(3000);w.close()


def test_side_entry_departing_from_exact_lane_edge_gets_marker():
    c=course();e=AnalyticsEngine(c)
    feed(e,[(.65,.3),(.65,.35),(.7,.4),(.75,.45),(.8,.5),(.83,.55)])
    assert any(v['kind']=='course_out' for v in e.snapshot()[0]['events'])
    assert summarize(e.snapshot(),c.lanes)['completions']['B']==0
