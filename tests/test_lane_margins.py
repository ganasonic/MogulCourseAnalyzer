"""Lanes describe bump corridors, independently of the outer control gates."""
from copy import deepcopy
import json
import sqlite3

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from mogul.models import Calibration
from mogul.geometry import area, overlap
from mogul.repository import EventRepository
from mogul.editor import CalibrationView
from mogul.workflow import CalibrationWorkflow


MIDDLE = [[.2,.12], [.8,.12], [.95,.9], [.05,.9]]
OUTER_MOVES = [(0, 0, [.25,.18]), (0, 3, [.15,.82]),
               (-1, 1, [.75,.18]), (-1, 2, [.85,.82])]


@pytest.mark.parametrize('count', [2, 3, 4])
def test_outer_corners_move_in_both_axes_and_keep_shared_edges(count):
    c = Calibration(middle=deepcopy(MIDDLE), middle_confirmed=True)
    c.configure_lanes(count)
    for lane, corner, target in OUTER_MOVES:
        for item in c.lanes:
            item.confirmed = True
        index = lane if lane >= 0 else count-1
        assert c.move_lane_corner(index, corner, target)
        assert c.lane_polygon(index)[corner] == target
        assert c.middle == MIDDLE and c.middle_confirmed
        assert not c.lanes[index].confirmed
        assert all(item.confirmed for i, item in enumerate(c.lanes) if i != index)
    assert sum(area(c.lane_polygon(i)) for i in range(count)) < area(c.middle)
    assert c.lanes_valid()
    for i in range(count-1):
        a, b = c.lane_polygon(i), c.lane_polygon(i+1)
        assert a[1] == b[0] and a[2] == b[3]
        assert overlap(a,b) < 2e-6
    before = c.to_dict()
    for point in [[-.1,.2], [.01,.5], [.95,.95], [float('nan'),.3]]:
        assert not c.move_lane_corner(0, 0, point)
        assert c.to_dict() == before


@pytest.mark.parametrize('count', [2, 3, 4])
def test_gui_outer_four_corners_drag_undo_and_confirm(qtbot, count):
    c = Calibration(middle=deepcopy(MIDDLE), middle_confirmed=True)
    c.configure_lanes(count)
    workflow = CalibrationWorkflow(c)
    view = CalibrationView(workflow)
    qtbot.addWidget(view)
    view.resize(850,530)
    view.show()
    qtbot.waitExposed(view)
    view.set_frame(np.zeros((360,640,3), np.uint8))
    for lane, corner, target in OUTER_MOVES:
        index = lane if lane >= 0 else count-1
        view.set_mode('lane',index)
        if c.lanes[index].confirmed:
            center = np.mean(c.lane_polygon(index), axis=0)*[640,360]
            position = view.mapFromScene(*center)
            qtbot.mouseDClick(view.viewport(), Qt.LeftButton, pos=position)
            qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=position)
        assert len(view.handles) == 4
        before = deepcopy(c.boundaries)
        start = view.mapFromScene(*(np.array(c.lane_polygon(index)[corner])*[640,360]))
        end = view.mapFromScene(*(np.array(target)*[640,360]))
        qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end, delay=30)
        qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)
        np.testing.assert_allclose(c.lane_polygon(index)[corner], target, atol=.005)
        assert workflow.undo()
        c = workflow.calibration
        assert c.boundaries == before
        # Apply again after Undo, then exercise confirmation and marker hiding.
        assert c.move_lane_corner(index, corner, target)
        view.redraw()
        view.confirm()
        assert c.lanes[index].confirmed and not view.handles
    assert c.middle == MIDDLE
    assert c.lanes_valid()


def test_margin_save_restore_and_legacy_migration(tmp_path):
    c = Calibration(middle=deepcopy(MIDDLE), middle_confirmed=True)
    c.configure_lanes(3)
    for lane, corner, target in OUTER_MOVES:
        assert c.move_lane_corner(lane if lane >= 0 else 2, corner, target)
    repo = EventRepository(tmp_path/'margins.sqlite')
    session = repo.save(c)
    _, restored = repo.load()
    assert restored.to_dict() == c.to_dict()

    legacy = c.to_dict()
    legacy.pop('boundary_format')
    legacy['boundaries'] = [[i/3,i/3] for i in range(4)]
    with sqlite3.connect(repo.path) as db:
        db.execute('UPDATE calibrations SET data_json=?', (json.dumps(legacy),))
    _, migrated = repo.load()
    assert migrated.boundary_format == 2 and migrated.lanes_valid()
    assert migrated.lane_polygon(0)[0] == MIDDLE[0]
    assert migrated.move_lane_corner(0,0,[.25,.18])
    repo.save(migrated, session)
    _, saved_again = repo.load()
    assert saved_again.to_dict() == migrated.to_dict()
    with sqlite3.connect(repo.path) as db:
        original = json.loads(db.execute('SELECT data_json FROM calibrations ORDER BY id LIMIT 1').fetchone()[0])
        assert original == legacy  # previous snapshot was not rewritten
