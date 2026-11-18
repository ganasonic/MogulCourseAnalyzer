from types import SimpleNamespace
import numpy as np
from mogul.app import MainWindow
from mogul.analytics import AnalyticsEngine
from mogul.models import Calibration
from mogul.athlete_tracker import Track
from mogul.analysis_repository import AnalysisRepository
from mogul.video import VideoFrame


def make_window(qtbot,tmp_path):
    c=Calibration(middle=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],middle_confirmed=True)
    c.configure_lanes(2)
    for lane in c.lanes:lane.confirmed=True
    path=tmp_path/'saved.sqlite';repo=AnalysisRepository(path);aid=repo.begin(c,None,{},'fixture')
    engine=AnalyticsEngine(c)
    records=engine.update([Track(1,[.2,.2,.3,.3],.9,0,hits=5)],0,0)
    run=engine.snapshot()[0];run.update(qualified=True,completed_lane='A')
    run['events']=[{'id':'event','kind':'course_out','point':[.5,.5],'timestamp_ms':0,
                    'review':'unreviewed','lane_id':'A','bump_id':None,'bump_label':None}]
    repo.write_frame(aid,records,[run],1,0,0);repo.finish(aid,'completed')
    w=MainWindow(path,startup=False)
    qtbot.addWidget(w,before_close_func=lambda _:setattr(w.workflow,'dirty',False))
    w.workflow.calibration=c;w.analysis_panel.load(aid);w.analysis_panel.review_analysis_id=aid
    w.analysis_result={'id':aid,'status':'completed'}
    w.view.analysis_tracks=records;w.update_analysis_events([run],100)
    w.preview_tracking.tracks=records.copy()
    w.analysis_panel.refresh_history();w.view.redraw()
    return w,aid


def close(w):
    w.workflow.dirty=False;w.worker.requestInterruption();assert w.worker.wait(3000);w.close()


def test_reset_clears_all_visible_results_and_preserves_settings_and_saved_data(qtbot,tmp_path):
    w,aid=make_window(qtbot,tmp_path)
    settings=w.workflow.calibration.to_dict();before=w.database.read_bytes()
    assert w.lane_total_labels['A'].text().endswith('1') and w.view.analysis_events
    assert w.reset_analysis_results()
    assert w.analysis_result is None and w.analysis_panel.data is None
    assert not w.view.analysis_tracks and not w.view.analysis_events and not w.course_map.analysis_events
    assert not w.preview_tracking.tracks and not w.preview_tracking.history
    assert w.analysis_panel.runs.count()==0 and w.analysis_panel.event_list.count()==0
    assert w.lane_total_labels['A'].text().endswith('0') and w.analysis_panel.summary.text()=='集計なし'
    assert w.analysis_panel.review_analysis_id is None and w.analysis_panel.history.currentIndex()==-1
    assert w.workflow.calibration.to_dict()==settings and w.database.read_bytes()==before
    assert AnalysisRepository(w.database).load_analysis(aid)['runs'][0]['completed_lane']=='A'
    # Late preview/seek frames cannot automatically restore the cleared review overlay.
    w.on_frame(VideoFrame(np.zeros((100,160,3),np.uint8),0,0,w.request_id))
    assert not w.view.analysis_tracks and w.analysis_panel.data is None
    close(w)


def test_reset_is_disabled_and_refused_until_analysis_stops(qtbot,tmp_path):
    w,_=make_window(qtbot,tmp_path);w.analysis_job=SimpleNamespace()
    try:
        w.refresh_state()
        assert not w.reset_analysis_button.isEnabled()
        assert not w.reset_analysis_results() and w.analysis_panel.data is not None
    finally:
        w.analysis_job=None
    w.refresh_state();assert w.reset_analysis_button.isEnabled()
    close(w)


def test_live_reset_keeps_capture_and_recording_running(qtbot,tmp_path,monkeypatch):
    w,_=make_window(qtbot,tmp_path);w.metadata={'live':True};w.recording_active=True
    calls=[];monkeypatch.setattr(w.worker,'command',lambda *args:calls.append(args))
    assert w.reset_analysis_results()
    assert ('play',False) not in calls and w.recording_active
    assert w.analysis_panel.data is None and not w.view.analysis_events
    close(w)
