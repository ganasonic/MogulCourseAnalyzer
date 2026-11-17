"""Counting must never require users to annotate hundreds of bumps."""
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from mogul.app import MainWindow
from mogul.analytics import AnalyticsEngine,summarize
from mogul.athlete_tracker import Track
from mogul.landmarks import BumpRegistry,bump_errors
from mogul.models import Calibration
from mogul.repository import EventRepository
from mogul.workflow import AnalysisController,CalibrationWorkflow


@pytest.fixture
def course():
    c=Calibration(middle=[[.05,.05],[.95,.05],[.95,.95],[.05,.95]],middle_confirmed=True)
    c.configure_lanes(2)
    for lane in c.lanes:
        lane.confirmed=True
    return c


def test_counting_needs_no_gate_or_bump(course):
    controller=AnalysisController(CalibrationWorkflow(course))
    assert not course.bumps and not course.control_gates
    assert controller.can_start(True)
    assert not controller.can_start(False)
    assert not controller.can_start(True,require_landmarks=True)
    course.lanes[0].confirmed=False
    assert not controller.can_start(True)


def test_center_region_batch_and_single_undo(tmp_path,course):
    workflow=CalibrationWorkflow(course)
    registry=BumpRegistry(course)
    bump=registry.add_center('A',[.3,.3],3)
    assert bump.center==[.3,.3] and bump.origin=='manual-center'
    assert not bump.confirmed and bump.side=='UNKNOWN'
    before=deepcopy(course.to_dict())
    workflow.checkpoint()
    batch=registry.add_line('A',[.3,.4],[.3,.85],30,3)
    assert len(batch)==30 and all(b.origin=='manual-batch' and not b.confirmed for b in batch)
    assert [b.sequence for b in batch]==list(range(2,32))
    assert not any('Bレーン' in error for error in bump_errors(course))
    registry.confirm_all()
    assert all(b.confirmed for b in course.bumps) and course.bumps_confirmed
    path=tmp_path/'optional.sqlite'
    repo=EventRepository(path); repo.save(course)
    _,restored=repo.load()
    assert len(restored.bumps)==31 and restored.bumps_confirmed
    assert workflow.undo() and workflow.calibration.to_dict()==before


def test_invalid_batch_is_atomic_and_drafts_not_passages(course):
    registry=BumpRegistry(course)
    registry.add_center('A',[.3,.4],3)
    before=deepcopy(course.to_dict())
    with pytest.raises(ValueError):
        registry.add_line('A',[.3,.4],[.8,.85],20,3)
    assert course.to_dict()==before
    engine=AnalyticsEngine(course)
    for i,y in enumerate([.2,.25,.3,.35,.4,.45,.5]):
        engine.update([Track(1,[.28,y-.06,.32,y],.9,i*200,hits=3)],i*200,i)
    assert summarize(engine.snapshot(),course.lanes)['unique_runs']==1
    assert not next(iter(engine.runs.values())).bumps


def test_gui_center_and_two_click_bulk(qtbot,tmp_path,course):
    w=MainWindow(tmp_path/'gui.sqlite',startup=False)
    qtbot.addWidget(w,before_close_func=lambda widget:setattr(widget.workflow,'dirty',False))
    w.workflow.calibration=course
    w.source_ok=True
    w.view.set_frame(np.zeros((400,640,3),np.uint8))
    w.show(); qtbot.waitExposed(w)
    w.refresh(); w.right_tabs.setCurrentIndex(1)
    panel=w.landmarks_panel; view=w.view
    def click(point):
        qtbot.mouseClick(view.viewport(),Qt.LeftButton,pos=view.mapFromScene(point[0]*640,point[1]*400))
    try:
        assert w.analyze_button.isEnabled()==Path(panel.window.analysis_panel.model.text()).is_file()
        panel.tabs.setCurrentIndex(1)
        panel.tool.setCurrentIndex(panel.tool.findData('add_bump_center'))
        click([.3,.3])
        assert len(course.bumps)==1
        np.testing.assert_allclose(course.bumps[0].center,[.3,.3],atol=.005)
        panel.batch_count.setValue(25)
        panel.tool.setCurrentIndex(panel.tool.findData('add_bump_line'))
        click([.3,.4]); click([.3,.85])
        assert len(course.bumps)==26
        assert not course.bumps_confirmed
        panel.confirm_bumps.click()
        assert course.bumps_confirmed
        assert w.save()
    finally:
        w.workflow.dirty=False
        w.worker.requestInterruption(); assert w.worker.wait(5000)
        w.close()
