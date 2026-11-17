"""Phase 2 registration, real pixel candidate extraction, GUI and SQLite migration."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import sqlite3
import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest

from mogul.app import MainWindow
from mogul.geometry import interpolate
from mogul.landmarks import (BumpRegistry, ControlGateEditor, ControlGateDetector,
    ControlGate, BumpDetector, course_position, gate_errors, bump_errors)
from mogul.models import Calibration
from mogul.repository import EventRepository
from mogul.workflow import CalibrationWorkflow, AnalysisController


@pytest.fixture
def course():
    c = Calibration(middle=[[.05,.05],[.95,.05],[.95,.95],[.05,.95]],middle_confirmed=True)
    c.configure_lanes(2)
    for lane,corner,point in [(0,0,[.2,.05]),(0,3,[.2,.95]),(1,1,[.8,.05]),(1,2,[.8,.95])]:
        assert c.move_lane_corner(lane,corner,point)
    for lane in c.lanes:
        lane.confirmed = True
    return c


def rectangle(x,y,size=.025):
    return [[x-size,y-size],[x+size,y-size],[x+size,y+size],[x-size,y+size]]


def populated(c):
    editor = ControlGateEditor(c)
    for side,point,number in [('LEFT',[.12,.2],3),('LEFT',[.12,.6],5),('RIGHT',[.88,.35],3)]:
        gate = editor.add(side,point)
        editor.update(gate.id,side,number)
    editor.confirm()
    registry = BumpRegistry(c)
    for lane,x in [('A',.35),('B',.65)]:
        bump = registry.add(lane,rectangle(x,.4),3)
        registry.update(bump.id,lane,3,1,'UNKNOWN')
    registry.confirm()
    return c


def test_gate_numbering_sides_sparse_and_outside(course):
    editor = ControlGateEditor(course)
    original = deepcopy(course.boundaries)
    left = editor.add('LEFT',[.12,.2],origin='HSV',candidate_point=[.11,.19])
    right = editor.add('RIGHT',[.88,.35])
    with pytest.raises(ValueError):
        editor.confirm()
    editor.update(left.id,'LEFT',3)
    editor.update(right.id,'RIGHT',3)
    extra = editor.add('LEFT',[.12,.7])
    with pytest.raises(ValueError):
        editor.update(extra.id,'LEFT',3)
    editor.update(extra.id,'LEFT',7)
    editor.confirm()
    assert course.gates_confirmed
    assert [g.number for g in course.control_gates] == [3,3,7]
    assert left.candidate_point == [.11,.19]
    assert course.boundaries == original
    with pytest.raises(ValueError):
        editor.add('LEFT',[.5,.5])  # internal separator is not an outside control gate
    with pytest.raises(ValueError):
        editor.add('RIGHT',[.01,.5])
    old = left.point[:]
    assert not editor.move(left.id,[.5,.5]) and left.point == old
    assert editor.move(left.id,[.13,.25])
    assert not left.confirmed and not course.gates_confirmed
    assert left.candidate_point == [.11,.19]


def test_gate_order_uses_perspective_depth(course):
    editor = ControlGateEditor(course)
    a = editor.add('LEFT',[.12,.7])
    b = editor.add('LEFT',[.12,.2])
    editor.update(a.id,'LEFT',3)
    editor.update(b.id,'LEFT',4)
    with pytest.raises(ValueError,match='上流'):
        editor.confirm()
    c = Calibration(middle=[[.1,.1],[.8,.25],[.95,.9],[.05,.8]],middle_confirmed=True)
    midpoint = interpolate(c.middle[0],c.middle[3],.5)
    assert course_position(c,midpoint)[0] == pytest.approx(0,abs=1e-5)


def test_bump_registry_labels_ids_sides_and_regions(course):
    registry = BumpRegistry(course)
    first = registry.add('B',rectangle(.65,.3),3)
    second = registry.add('B',rectangle(.65,.45),3)
    third = registry.add('B',rectangle(.65,.6),5)
    assert [b.sequence for b in course.bumps] == [1,2,1]
    assert first.label == 'B-3-1' and third.label == 'B-5-1'
    identifier = first.id
    registry.update(first.id,'B',3,1,'L')
    assert first.label == 'B-3-1-L' and first.id == identifier and first.side_confidence == 1
    registry.update(first.id,'B',3,1,'UNKNOWN')
    assert first.side_confidence is None and first.label == 'B-3-1'
    with pytest.raises(ValueError):
        registry.update(second.id,'B',3,1,'R')
    with pytest.raises(ValueError):
        registry.update(first.id,'A',3,1,'L')
    previous = deepcopy(first.polygon)
    assert not registry.move(first.id,[.4,.3]) and first.polygon == previous
    assert registry.move(first.id,[.68,.32])
    assert first.id == identifier and not first.confirmed
    assert registry.resize(first.id,0,[.64,.28])
    assert not registry.resize(first.id,0,[.1,.1])
    assert 'Aレーン' in ' '.join(bump_errors(course))
    registry.remove(second.id)
    assert all(b.id != second.id for b in course.bumps)


def test_real_registration_is_required_and_rechecks_layout(course):
    controller = AnalysisController(CalibrationWorkflow(course))
    course.gates_confirmed = course.bumps_confirmed = True
    assert not controller.can_start(True)
    populated(course)
    assert controller.can_start(True) and not controller.can_start(False)
    data = course.to_dict()
    restored = Calibration.from_dict(data)
    assert restored.gates_confirmed and restored.bumps_confirmed
    original_ids = [b.id for b in course.bumps]
    assert course.move_lane_corner(0,0,[.21,.06])
    assert not controller.can_start(True)
    assert [b.id for b in course.bumps] == original_ids
    assert not any(b.confirmed for b in course.bumps)
    assert not any(g.confirmed for g in course.control_gates)
    assert Calibration.from_dict(course.to_dict()).bumps


def test_color_candidates_are_real_pixels_not_ai(course):
    frame = np.full((400,400,3),240,np.uint8)
    cv2.rectangle(frame,(43,93),(53,107),(255,0,0),-1)
    cv2.rectangle(frame,(347,273),(357,287),(0,0,255),-1)
    cv2.rectangle(frame,(195,195),(205,205),(255,0,0),-1)  # inside lanes, ignored
    detector = ControlGateDetector()
    result = detector.detect([frame,frame],course)
    assert len(result.candidates) == 2
    assert {g['side'] for g in result.candidates} == {'LEFT','RIGHT'}
    assert all(g['confidence'] is None and g['origin'] == 'HSV' for g in result.candidates)
    assert 'AI未導入' in result.message
    assert not detector.detect([np.zeros_like(frame)],course).candidates
    assert not detector.detect([frame],course,cancelled=lambda:True).candidates
    assert not BumpDetector().detect([],course)[0]


def test_phase2_snapshot_tables_and_v1_backup(tmp_path,course):
    populated(course)
    path = tmp_path/'phase2.sqlite'
    repo = EventRepository(path)
    session = repo.save(course)
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA user_version=1')  # simulate previous on-disk schema marker
    editor = ControlGateEditor(course)
    editor.move(course.control_gates[0].id,[.13,.21])
    repo.save(course,session)
    assert Path(str(path)+'.v1.bak').is_file()
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == EventRepository.SCHEMA_VERSION
        assert db.execute('SELECT COUNT(*) FROM control_gates').fetchone()[0] == 6
        assert db.execute('SELECT COUNT(*) FROM bumps').fetchone()[0] == 4
    _,restored = repo.load()
    assert restored.to_dict() == course.to_dict()
    _,backup = EventRepository(str(path)+'.v1.bak').load()
    assert backup.gates_confirmed and backup.control_gates[0].point == [.12,.2]


def test_actual_legacy_schema_upgrades_without_rewriting_history(tmp_path,course):
    path = tmp_path/'legacy.sqlite'
    legacy = course.to_dict()
    for key in ('control_gates','bumps'):
        legacy.pop(key)
    with sqlite3.connect(path) as db:
        db.executescript('''PRAGMA user_version=1;
            CREATE TABLE sessions(id TEXT PRIMARY KEY,created_at TEXT NOT NULL,source_json TEXT NOT NULL);
            CREATE TABLE calibrations(id INTEGER PRIMARY KEY,session_id TEXT NOT NULL,created_at TEXT NOT NULL,data_json TEXT NOT NULL);
            CREATE TABLE calibration_edits(id INTEGER PRIMARY KEY,session_id TEXT NOT NULL,created_at TEXT NOT NULL,data_json TEXT NOT NULL);''')
        db.execute('INSERT INTO sessions VALUES(?,?,?)',('old','2026-11-16','{}'))
        db.execute('INSERT INTO calibrations VALUES(?,?,?,?)',(1,'old','2026-11-16',json.dumps(legacy)))
    repo = EventRepository(path)
    session,restored = repo.load()
    assert session == 'old' and not restored.control_gates
    populated(restored)
    repo.save(restored,session)
    with sqlite3.connect(path) as db:
        assert json.loads(db.execute('SELECT data_json FROM calibrations WHERE id=1').fetchone()[0]) == legacy
        assert db.execute('SELECT COUNT(*) FROM control_gates').fetchone()[0] == 3
    with sqlite3.connect(str(path)+'.v1.bak') as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 1
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='control_gates'").fetchall()


@pytest.fixture
def window(qtbot,tmp_path,course):
    window = MainWindow(tmp_path/'gui.sqlite',startup=False)
    qtbot.addWidget(window,before_close_func=lambda w:setattr(w.workflow,'dirty',False))
    window.workflow.calibration = course
    window.source_ok = True
    window.view.set_frame(np.zeros((400,640,3),np.uint8))
    window.refresh()
    window.show()
    qtbot.waitExposed(window)
    window.right_tabs.setCurrentIndex(1)
    yield window
    if window.landmarks_panel.busy:
        window.landmarks_panel.task.requestInterruption()
        window.landmarks_panel.task.wait(5000)
        qtbot.wait(20)
    window.workflow.dirty = False
    window.worker.requestInterruption()
    assert window.worker.wait(5000)
    window.close()


def position(view,point):
    return view.mapFromScene(point[0]*view.width_px,point[1]*view.height_px)


def test_gui_add_drag_confirm_map_and_save(qtbot,window):
    panel,view = window.landmarks_panel,window.view
    panel.tool.setCurrentIndex(panel.tool.findData('add_left'))
    qtbot.mouseClick(view.viewport(),Qt.LeftButton,pos=position(view,[.12,.2]))
    assert len(panel.c.control_gates) == 1
    gate = panel.c.control_gates[0]
    panel.gate_number.setValue(3)
    panel.confirm_gate()
    assert gate.confirmed
    panel.confirm_gates.click()
    assert panel.c.gates_confirmed
    # Markers move without accidentally moving a lane or auto-confirming.
    panel.tool.setCurrentIndex(0)
    qtbot.mousePress(view.viewport(),Qt.LeftButton,pos=position(view,gate.point))
    QTest.mouseMove(view.viewport(),position(view,[.13,.25]),delay=30)
    qtbot.mouseRelease(view.viewport(),Qt.LeftButton,pos=position(view,[.13,.25]))
    np.testing.assert_allclose(gate.point,[.13,.25],atol=.005)
    assert not gate.confirmed
    panel.confirm_gate()
    panel.confirm_gates.click()
    for lane,x in [('A',.35),('B',.65)]:
        panel.tool.setCurrentIndex(panel.tool.findData('add_bump'))
        panel.bump_lane.setCurrentIndex(panel.bump_lane.findData(lane))
        panel.bump_gate.setValue(3)
        rect = rectangle(x,.4,.035)
        qtbot.mousePress(view.viewport(),Qt.LeftButton,pos=position(view,rect[0]))
        QTest.mouseMove(view.viewport(),position(view,rect[2]),delay=30)
        qtbot.mouseRelease(view.viewport(),Qt.LeftButton,pos=position(view,rect[2]))
        assert len(panel.c.bumps) == (1 if lane == 'A' else 2)
        panel.bump_side.setCurrentIndex(panel.bump_side.findData('R' if lane == 'B' else 'UNKNOWN'))
        panel.confirm_bump()
    panel.confirm_bumps.click()
    assert panel.c.bumps_confirmed and window.controller.can_start(window.source_ok)
    assert window.analyze_button.isEnabled() == Path(window.analysis_panel.model.text()).is_file()
    assert panel.c.bumps[1].label == 'B-3-1-R'
    window.course_map.selected.emit('bump',panel.c.bumps[1].id)
    assert view.selected_landmark == ('bump',panel.c.bumps[1].id)
    assert not view.handles
    panel.tool.setCurrentIndex(panel.tool.findData('resize_bump'))
    assert len(view.handles) == 4
    assert window.save()
    _,restored = EventRepository(window.database).load()
    assert restored.gates_confirmed and restored.bumps_confirmed
    assert restored.bumps[1].label == 'B-3-1-R'
    assert not window.course_map.grab().isNull()


def test_landmark_mirror_and_undo(window):
    populated(window.workflow.calibration)
    original = deepcopy(window.workflow.calibration)
    window.mirror.setChecked(True)
    c = window.workflow.calibration
    assert c.control_gates[0].point == pytest.approx([.88,.2])
    assert c.control_gates[0].side == 'RIGHT'
    assert c.bumps[0].side == original.bumps[0].side
    assert not c.gates_confirmed and not c.bumps_confirmed
    window.undo()
    assert window.workflow.calibration.to_dict() == original.to_dict()


def test_gui_restore_phase2_with_actual_video(qtbot,window,tmp_path):
    path = tmp_path/'restored.mp4'
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,100))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((100,160,3),np.uint8))
    writer.release()
    c = populated(window.workflow.calibration)
    c.source = {'kind':'file','path':str(path)}
    assert window.save()
    window.workflow.calibration = Calibration()
    assert window.restore(window.database)
    qtbot.waitUntil(lambda:window.source_ok and window.current_frame is not None,timeout=8000)
    restored = window.workflow.calibration
    assert restored.gates_confirmed and restored.bumps_confirmed
    assert window.landmarks_panel.gates.count() == 3
    assert window.landmarks_panel.bumps.count() == 2
    assert window.view.mode == 'landmarks' and window.right_tabs.currentIndex() == 1
    assert window.analyze_button.isEnabled() == Path(window.analysis_panel.model.text()).is_file()


def test_bump_gui_move_resize_delete_undo_and_map_click(qtbot,window):
    c = populated(window.workflow.calibration)
    window.refresh()
    panel,view = window.landmarks_panel,window.view
    bump = c.bumps[0]
    identifier = bump.id
    panel.select_record('bump',identifier)
    qtbot.mousePress(view.viewport(),Qt.LeftButton,pos=position(view,bump.center))
    QTest.mouseMove(view.viewport(),position(view,[.37,.42]),delay=30)
    qtbot.mouseRelease(view.viewport(),Qt.LeftButton,pos=position(view,[.37,.42]))
    np.testing.assert_allclose(bump.center,[.37,.42],atol=.005)
    assert not bump.confirmed and not c.bumps_confirmed and c.gates_confirmed
    old_corner = bump.polygon[0][:]
    panel.tool.setCurrentIndex(panel.tool.findData('resize_bump'))
    qtbot.mousePress(view.viewport(),Qt.LeftButton,pos=position(view,old_corner))
    QTest.mouseMove(view.viewport(),position(view,[.33,.38]),delay=30)
    qtbot.mouseRelease(view.viewport(),Qt.LeftButton,pos=position(view,[.33,.38]))
    np.testing.assert_allclose(bump.polygon[0],[.33,.38],atol=.005)
    panel.remove('bump')
    assert all(b.id != identifier for b in panel.c.bumps)
    window.undo()
    assert any(b.id == identifier for b in panel.c.bumps)
    panel.select_record('gate',panel.c.control_gates[0].id)
    window.right_tabs.setCurrentIndex(2)
    qtbot.wait(30)
    graphics = next(item for item in window.course_map.scene().items() if item.data(0) == 'bump' and item.data(1) == identifier and hasattr(item,'rect'))
    qtbot.mouseClick(window.course_map.viewport(),Qt.LeftButton,pos=window.course_map.mapFromScene(graphics.scenePos()))
    assert panel.selected == ('bump',identifier)


def test_tiny_bump_center_can_move_without_resize_collision(qtbot,window):
    c = window.workflow.calibration
    bump = BumpRegistry(c).add('A',rectangle(.35,.4,.008),3)
    panel,view = window.landmarks_panel,window.view
    panel.select_record('bump',bump.id)
    panel.tool.setCurrentIndex(panel.tool.findData('select'))
    assert not view.handles
    qtbot.mousePress(view.viewport(),Qt.LeftButton,pos=position(view,bump.center))
    QTest.mouseMove(view.viewport(),position(view,[.38,.43]),delay=30)
    qtbot.mouseRelease(view.viewport(),Qt.LeftButton,pos=position(view,[.38,.43]))
    np.testing.assert_allclose(bump.center,[.38,.43],atol=.005)


def test_detection_worker_does_not_overwrite_changed_settings(qtbot,window,monkeypatch):
    from mogul.video import VideoFrame
    original = ControlGateDetector.detect
    def slow(self,frames,calibration,cancelled=lambda:False):
        time.sleep(.15)
        return original(self,frames,calibration,cancelled)
    monkeypatch.setattr(ControlGateDetector,'detect',slow)
    frame = np.zeros((400,640,3),np.uint8)
    window.current_frame = VideoFrame(frame,0,0,window.request_id)
    panel = window.landmarks_panel
    ticks = []
    timer = QTimer(window)
    timer.timeout.connect(lambda:ticks.append(1))
    timer.start(10)
    panel.detect()
    window.workflow.checkpoint()
    window.workflow.calibration.opacity = .5
    qtbot.waitUntil(lambda:not panel.busy,timeout=5000)
    timer.stop()
    assert ticks and '適用しません' in panel.status.text()
    assert window.workflow.calibration.opacity == .5
