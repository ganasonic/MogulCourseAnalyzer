"""Executable environment checks, not application acceptance tests."""
import json
import subprocess

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from tools.v18_probe import Probe, normalized, pixels, valid_quad, overlap


def test_geometry():
    a = [[0, 0], [100, 0], [80, 100], [20, 100]]
    assert valid_quad(a)
    assert not valid_quad([a[0], a[2], a[1], a[3]])
    np.testing.assert_allclose(pixels(normalized(a, (100, 100)), (200, 300)), np.array(a)*[2, 3])
    assert cv2.pointPolygonTest(np.array(a, np.float32), (50, 50), False) == 1
    assert cv2.pointPolygonTest(np.array(a, np.float32), (0, 0), False) == 0
    assert overlap(a, [[40, 20], [60, 20], [60, 40], [40, 40]]) == pytest.approx(400)
    left = [[0, 0], [50, 0], [50, 100], [0, 100]]
    right = [[50, 0], [100, 0], [100, 100], [50, 100]]
    assert overlap(left, right) == 0  # shared boundary is not area overlap
    transform = cv2.getPerspectiveTransform(np.array(a, np.float32), np.array([[0,0],[100,0],[100,100],[0,100]], np.float32))
    mapped = cv2.perspectiveTransform(np.array([a], np.float32), transform)[0]
    np.testing.assert_allclose(mapped, [[0,0],[100,0],[100,100],[0,100]], atol=1e-4)


def test_gui(qtbot, tmp_path):
    view = Probe()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    assert view.poly.brush().color().alpha() == 90
    assert len(view.handles) == 4
    start = view.mapFromScene(view.handles[0].pos())
    end = start + QPoint(25, 20)
    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    QTest.mouseMove(view.viewport(), end, delay=30)
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)
    qtbot.wait(50)
    assert view.points()[0] == pytest.approx([205, 90], abs=1)
    assert view.poly.polygon()[0] == view.handles[0].pos()
    with qtbot.waitSignal(view.confirmed):
        qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=view.mapFromScene(320, 180))
    assert view.locked
    with qtbot.waitSignal(view.reedited):
        qtbot.mouseDClick(view.viewport(), Qt.LeftButton, pos=view.mapFromScene(320, 180))
    assert not view.locked
    for count in (2, 3, 4):
        view.set_lanes(count)
        assert len(view.lanes) == count
    expected = view.points()
    path = tmp_path / 'calibration.json'
    view.save(path)
    view.handles[0].setPos(1, 1)
    view.set_lanes(2)
    view.restore(path)
    np.testing.assert_allclose(view.points(), expected)
    assert view.lane_count == 4
    assert not view.grab().isNull()


@pytest.mark.parametrize('extension', ['mp4', 'mov'])
def test_video(extension, tmp_path):
    path = tmp_path / ('sample.' + extension)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc=size=160x120:rate=10', '-t', '2', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', str(path)], check=True, capture_output=True, timeout=30)
    metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)]))
    assert float(metadata['format']['duration']) == pytest.approx(2, abs=.1)
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(10)
        assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == 160
        assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == 120
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 20
        assert capture.set(cv2.CAP_PROP_POS_MSEC, 1000)
        ok, frame = capture.read()
        assert ok and frame.shape == (120, 160, 3)
        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
        assert timestamp == pytest.approx(1000, abs=100)
        # CFR marker lookup follows decoded media timestamp, not wall clock.
        markers = {i: i / 10 for i in range(20)}
        assert markers[round(timestamp / 100)] == pytest.approx(1, abs=.1)
    finally:
        capture.release()
