from copy import deepcopy
import json
import sqlite3
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from mogul.models import Calibration
from mogul.repository import EventRepository
from mogul.workflow import CalibrationWorkflow
from mogul.editor import CalibrationView
from mogul.course_region import CourseRegion
from mogul.analytics import AnalyticsEngine
from mogul.athlete_tracker import Track
from mogul.orientation import rotate_calibration


def course(count=4):
    c=Calibration(middle=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],middle_confirmed=True)
    c.configure_lanes(count)
    for lane in c.lanes: lane.confirmed=True
    return c


def gap_course():
    c=course()
    for index,corner,target in [(1,1,[.47,.1]),(1,2,[.47,.9]),(2,0,[.53,.1]),(2,3,[.53,.9])]:
        assert c.move_lane_corner(index,corner,target)
    return c


@pytest.mark.parametrize('count',[2,3,4])
def test_all_four_corners_are_independent_in_every_lane(count):
    c=course(count)
    for index in range(count):
        for corner in range(4):
            for lane in c.lanes: lane.confirmed=True
            before=[c.lane_polygon(i) for i in range(count)]
            center=np.mean(before[index],axis=0)
            target=(np.array(before[index][corner])*.9+center*.1).tolist()
            assert c.move_lane_corner(index,corner,target)
            assert c.lane_polygon(index)[corner]==target and c.lanes_valid()
            for other in range(count):
                if other!=index:
                    assert c.lane_polygon(other)==before[other] and c.lanes[other].confirmed
            assert not c.lanes[index].confirmed


def test_b_c_gap_is_not_course_or_lane_and_overlap_is_rejected():
    c=gap_course();region=CourseRegion(c);engine=AnalyticsEngine(c)
    for x in [.48,.50,.52]:
        assert not region.contains([x,.5]) and not region.can_start([x,.5])
        assert engine.lane_for([x,.5],'B') is None
    engine.update([Track(1,[.49,.4,.51,.5],.9,0,hits=3)],0,0)
    assert not engine.snapshot()
    before=c.to_dict()
    assert not c.move_lane_corner(1,1,[.6,.1])
    assert c.to_dict()==before


def test_gap_save_restore_rotate_and_undo(tmp_path):
    c=gap_course();workflow=CalibrationWorkflow(c)
    original=c.to_dict();workflow.checkpoint()
    assert c.move_lane_corner(1,1,[.45,.12])
    assert workflow.undo() and workflow.calibration.to_dict()==original
    repo=EventRepository(tmp_path/'independent.sqlite');repo.save(c)
    _,restored=repo.load()
    assert restored.to_dict()==c.to_dict()
    before=restored.to_dict()
    for angle in [90,180,270,0]:rotate_calibration(restored,angle)
    np.testing.assert_allclose(restored.boundaries,before['boundaries'],atol=1e-12)


def test_shared_v2_snapshot_migrates_without_rewriting_original(tmp_path):
    c=course();legacy=c.to_dict();polygons=[c.lane_polygon(i) for i in range(4)]
    legacy['boundary_format']=2
    legacy['boundaries']=[[polygons[0][0],polygons[0][3]]]+[[p[1],p[2]] for p in polygons]
    before=deepcopy(legacy)
    restored=Calibration.from_dict(legacy)
    assert legacy==before and restored.boundary_format==3
    assert [restored.lane_polygon(i) for i in range(4)]==polygons
    neighbor=restored.lane_polygon(2)
    assert restored.move_lane_corner(1,1,[.47,.1]) and restored.lane_polygon(2)==neighbor
    repo=EventRepository(tmp_path/'legacy.sqlite');session=repo.save(c)
    with sqlite3.connect(repo.path) as db:
        db.execute('update calibrations set data_json=?',(json.dumps(legacy),))
    _,converted=repo.load();repo.save(converted,session)
    with sqlite3.connect(repo.path) as db:
        assert json.loads(db.execute('select data_json from calibrations order by id limit 1').fetchone()[0])==before


def test_gui_b_right_corner_does_not_drag_c_left_corner(qtbot):
    c=course();workflow=CalibrationWorkflow(c);view=CalibrationView(workflow)
    qtbot.addWidget(view);view.resize(850,530);view.show();qtbot.waitExposed(view)
    view.set_frame(np.zeros((360,640,3),np.uint8));c.lanes[1].confirmed=False;view.set_mode('lane',1)
    assert view.overlays[0].brush().color().alpha()==0  # middle fill must not color the gap as a lane
    neighbor=c.lane_polygon(2)
    start=view.mapFromScene(.5*640,.1*360);end=view.mapFromScene(.47*640,.12*360)
    qtbot.mousePress(view.viewport(),Qt.LeftButton,pos=start)
    QTest.mouseMove(view.viewport(),end,delay=30)
    qtbot.mouseRelease(view.viewport(),Qt.LeftButton,pos=end)
    np.testing.assert_allclose(c.lane_polygon(1)[1],[.47,.12],atol=.005)
    assert c.lane_polygon(2)==neighbor and c.lanes[2].confirmed
    view.confirm();assert c.lanes[1].confirmed
    assert workflow.undo() and workflow.calibration.lane_polygon(2)==neighbor
