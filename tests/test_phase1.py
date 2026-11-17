"""Phase 1 model, GUI, persistence and threaded video regression checks."""
from copy import deepcopy
from pathlib import Path
import sqlite3
import subprocess
import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest

from mogul.app import MainWindow
from mogul.detectors import LaneDetector, MiddlePartDetector
from mogul.editor import CalibrationView
from mogul.geometry import area, inside, overlap, valid_quad
from mogul.models import Calibration
from mogul.repository import EventRepository
from mogul.video import FileVideoSource, VideoFrame, VideoRecorder, VideoWorker
from mogul.workflow import AnalysisController, CalibrationWorkflow


MIDDLE = [[.2, .12], [.8, .12], [.95, .9], [.05, .9]]


@pytest.fixture
def calibration():
    return Calibration(middle=deepcopy(MIDDLE), middle_confirmed=True)


@pytest.fixture
def view(qtbot):
    widget = CalibrationView(CalibrationWorkflow(Calibration()))
    qtbot.addWidget(widget)
    widget.resize(850, 530)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_frame(np.zeros((360, 640, 3), np.uint8))
    yield widget
    widget.click_timer.stop()


def at(view, x, y):
    return view.mapFromScene(x*view.width_px, y*view.height_px)


def drag(qtbot, view, start, end):
    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=at(view, *start))
    QTest.mouseMove(view.viewport(), at(view, *end), delay=30)
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=at(view, *end))


def click_confirm(qtbot, view, point=(.5, .5)):
    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=at(view, *point))
    qtbot.wait(view.click_timer.interval()+40)


def test_middle_rectangle_drag_confirm_reedit(qtbot, view):
    drag(qtbot, view, (.2, .15), (.85, .88))
    assert valid_quad(view.calibration.middle)
    assert len(view.handles) == 4
    before = deepcopy(view.calibration.middle)
    drag(qtbot, view, before[0], (.3, .12))
    np.testing.assert_allclose(view.calibration.middle[0], [.3, .12], atol=.005)
    # Clicking a handle does not confirm.
    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=at(view, *view.calibration.middle[0]))
    qtbot.wait(450)
    assert not view.calibration.middle_confirmed
    click_confirm(qtbot, view)
    assert view.calibration.middle_confirmed
    assert not view.handles
    expected = deepcopy(view.calibration.middle)
    # Outside clicks and double-clicks preserve the polygon and its confirmation.
    qtbot.mouseDClick(view.viewport(), Qt.LeftButton, pos=at(view, .02, .02))
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=at(view, .02, .02))
    assert view.calibration.middle == expected and view.calibration.middle_confirmed
    # Simulate a real double click sequence, including the first single click/release.
    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=at(view, .5, .5))
    qtbot.mouseDClick(view.viewport(), Qt.LeftButton, pos=at(view, .5, .5))
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=at(view, .5, .5))
    qtbot.wait(450)
    assert not view.calibration.middle_confirmed
    assert len(view.handles) == 4


def test_invalid_middle_and_resize(qtbot, view):
    view.workflow.middle_changed(deepcopy(MIDDLE))
    view.redraw()
    before = deepcopy(view.calibration.middle)
    # First corner across the opposite edge must be rejected.
    drag(qtbot, view, before[0], (.65, .99))
    assert view.calibration.middle == before
    drag(qtbot, view, before[0], (-.1, -.1))
    assert np.min(view.calibration.middle) >= 0
    saved = deepcopy(view.calibration.middle)
    view.set_frame(np.zeros((720, 1280, 3), np.uint8))
    assert view.calibration.middle == saved
    assert view.width_px == 1280
    assert view.workflow.undo()
    assert view.calibration.middle == before


@pytest.mark.parametrize('count', [2, 3, 4])
def test_independent_lanes_start_with_equal_split(calibration, count):
    calibration.configure_lanes(count)
    assert calibration.lanes_valid()
    polygons = [calibration.lane_polygon(i) for i in range(count)]
    assert sum(area(p) for p in polygons) == pytest.approx(area(MIDDLE), abs=1e-6)
    for i in range(count-1):
        assert polygons[i][1] == polygons[i+1][0]
        assert polygons[i][2] == polygons[i+1][3]
        assert overlap(polygons[i], polygons[i+1]) < 2e-6
    for lane in calibration.lanes:
        lane.confirmed = True
    neighbor=calibration.lane_polygon(1)
    assert calibration.move_lane_corner(0, 1, [.45 if count == 2 else .34, .12])
    assert calibration.lane_polygon(1)==neighbor
    assert calibration.lane_polygon(0)[1] != calibration.lane_polygon(1)[0]
    assert not calibration.lanes[0].confirmed and calibration.lanes[1].confirmed
    assert not calibration.move_lane_corner(0, 0, [.1, .1])
    old = deepcopy(calibration.boundaries)
    assert not calibration.move_lane_corner(0, 1, [.99, .12])
    assert calibration.boundaries == old
    assert calibration.lanes_valid()


def test_lane_gui_drag_and_confirmation(qtbot, view):
    c = view.calibration
    c.middle = deepcopy(MIDDLE)
    c.middle_confirmed = True
    c.configure_lanes(3)
    view.set_mode('lane', 0)
    before = c.lane_polygon(0)[1]
    neighbor=c.lane_polygon(1)
    drag(qtbot, view, before, (.36, .12))
    assert c.lane_polygon(1)==neighbor
    assert c.lane_polygon(0)[1] != c.lane_polygon(1)[0]
    np.testing.assert_allclose(c.lane_polygon(0)[1], [.36, .12], atol=.005)
    for i in range(3):
        view.set_mode('lane', i)
        click_confirm(qtbot, view)
        assert c.lanes[i].confirmed and not view.handles
    polygon = c.lane_polygon(1)
    center = np.mean(polygon, axis=0)
    qtbot.mouseDClick(view.viewport(), Qt.LeftButton, pos=at(view, *center))
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=at(view, *center))
    qtbot.wait(450)
    assert view.lane_index == 1 and not c.lanes[1].confirmed
    assert len(view.handles) == 4


def test_general_quad_geometry():
    concave = [[.1,.1], [.9,.1], [.4,.4], [.1,.9]]
    assert valid_quad(concave)
    assert inside(concave, concave)
    assert not valid_quad([[.1,.1],[.9,.9],[.9,.1],[.1,.9]])
    assert not valid_quad([[0,0],[1,0],[1,0],[0,1]])
    assert not valid_quad([[0,0],[1,0],[float('nan'),1],[0,1]])
    assert not valid_quad([[0,0],[1,0],[2,1],[0,1]])


def test_repository_roundtrip_history_and_reject(tmp_path, calibration):
    calibration.configure_lanes(4)
    calibration.source = {'kind':'file', 'path': '日本語.mp4'}
    calibration.lanes[0].name = 'ブルー外側'
    calibration.candidate = {'middle': {'success': False, 'message': 'モデル未導入'}}
    path = tmp_path/'calibration.sqlite'
    repo = EventRepository(path)
    session = repo.save(calibration, history=[deepcopy(calibration)])
    calibration.lanes[0].confirmed = True
    repo.save(calibration, session)
    restored_id, restored = repo.load()
    assert restored_id == session
    assert restored.to_dict() == calibration.to_dict()
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM calibrations').fetchone()[0] == 2
        assert db.execute('SELECT COUNT(*) FROM calibration_edits').fetchone()[0] == 1
    bad = calibration.to_dict()
    bad['boundaries'][1][0] = 1
    with pytest.raises(ValueError):
        Calibration.from_dict(bad)


def test_analysis_prerequisites(calibration):
    workflow = CalibrationWorkflow(calibration)
    controller = AnalysisController(workflow)
    assert not controller.can_start(False)
    calibration.configure_lanes(2)
    for lane in calibration.lanes:
        lane.confirmed = True
    assert controller.can_start(True)  # people counting does not require landmark registration
    assert len(workflow.reasons(True,require_landmarks=True)) == 2
    calibration.gates_confirmed = calibration.bumps_confirmed = True
    assert not controller.can_start(True,require_landmarks=True)  # feature-specific strict checks
    assert not controller.can_start(False)


def test_ai_failure_is_explicit():
    middle = MiddlePartDetector().detect([])
    lanes = LaneDetector().detect([], MIDDLE, 4)
    assert not middle.success and not middle.polygons and middle.model is None
    assert not lanes.success and not lanes.polygons
    assert 'モデル未導入' in middle.message and '手動' in lanes.message


@pytest.fixture(scope='module')
def video_path(tmp_path_factory):
    path = tmp_path_factory.mktemp('video')/'日本語 clip.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=size=320x180:rate=10',
                    '-t', '2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True, timeout=30)
    return path


def test_file_source(video_path):
    source = FileVideoSource(video_path)
    try:
        assert source.metadata['duration_ms'] == pytest.approx(2000)
        assert source.read().timestamp_ms == pytest.approx(0)
        source.seek(1000)
        frame = source.read()
        assert frame.timestamp_ms == pytest.approx(1000, abs=100)
        assert frame.image.shape == (180,320,3)
        source.seek(99999)
        assert source.read() is not None
    finally:
        source.close()


def test_video_recorder(tmp_path):
    path = tmp_path/'record.mp4'
    recorder = VideoRecorder(path, 10, (64, 48))
    for index in range(10):
        recorder.write(VideoFrame(np.full((48,64,3), index*20, np.uint8), index*100, index))
    recorder.close()
    source = FileVideoSource(path)
    try:
        assert source.count == 10 and source.read() is not None
    finally:
        source.close()
    assert Path(str(path)+'.timestamps.json').is_file()


def test_threaded_play_seek_ui_responsive(qtbot, video_path, tmp_path):
    window = MainWindow(tmp_path/'session.sqlite', startup=False)
    qtbot.addWidget(window)
    window.show()
    try:
        window.open_source({'kind':'file','path':str(video_path)})
        qtbot.waitUntil(lambda: window.source_ok and window.current_frame is not None, timeout=10000)
        assert window.play_button.isEnabled()
        qtbot.waitUntil(lambda: not window.auto_detect_pending)
        assert not window.analyze_button.isEnabled()
        window.worker.command('seek', 1000)
        qtbot.waitUntil(lambda: abs(window.current_frame.timestamp_ms-1000) <= 100, timeout=5000)
        window.step(1)
        qtbot.waitUntil(lambda: window.current_frame.timestamp_ms >= 1100, timeout=5000)
        window.worker.command('seek', 0)
        qtbot.waitUntil(lambda: window.current_frame.timestamp_ms == 0, timeout=5000)
        ticks = []
        timer = QTimer(window)
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start(20)
        window.toggle_play()
        drag(qtbot, window.view, (.2,.15), (.85,.88))
        qtbot.waitUntil(lambda: window.current_frame.index >= 4, timeout=5000)
        assert len(ticks) >= 3 and valid_quad(window.workflow.calibration.middle)
        assert window.save()
        assert window.database.is_file()
        expected = deepcopy(window.workflow.calibration.middle)
        assert window.restore(window.database)
        assert window.workflow.calibration.middle == expected
        timer.stop()
    finally:
        window.workflow.dirty = False
        window.worker.requestInterruption()
        assert window.worker.wait(5000)
        window.close()


def test_disconnect_reconnect_simulated(qtbot):
    calls = []

    class FakeUVC:
        live = True
        fps = 30
        metadata = {'kind':'uvc','width':64,'height':48,'fps':30,'duration_ms':0,'live':True}

        def read(self):
            time.sleep(.02)
            if len(calls) == 1:
                raise OSError('simulated unplug')
            return VideoFrame(np.zeros((48,64,3), np.uint8), 0, 0)

        def close(self):
            pass

    def factory(spec):
        calls.append(spec)
        return FakeUVC()

    worker = VideoWorker(factory=factory)
    statuses, frames = [], []
    worker.status.connect(lambda text, healthy, request_id: statuses.append((text, healthy)))
    worker.frame_ready.connect(frames.append)
    worker.start()
    try:
        worker.command('open', {'kind':'uvc','device':'simulated'})
        qtbot.waitUntil(lambda: any(not healthy for _, healthy in statuses), timeout=2000)
        qtbot.waitUntil(lambda: bool(frames) and any(healthy for _, healthy in statuses), timeout=7000)
        assert len(calls) >= 2 and any(healthy for _, healthy in statuses)
    finally:
        worker.requestInterruption()
        assert worker.wait(3000)


@pytest.fixture
def window(qtbot, tmp_path):
    window = MainWindow(tmp_path/'session.sqlite', startup=False)
    qtbot.addWidget(window, before_close_func=lambda widget: setattr(widget.workflow, 'dirty', False))
    window.show()
    yield window
    window.workflow.dirty = False
    window.worker.requestInterruption()
    assert window.worker.wait(5000)
    window.close()


def test_source_switch_stale_frames_and_failure(qtbot, window, video_path, tmp_path):
    window.open_source({'kind':'file','path':str(video_path)})
    qtbot.waitUntil(lambda: window.source_ok, timeout=8000)
    qtbot.wait(30)  # allow deferred no-model notification
    old = window.current_frame
    old_id = window.request_id
    c = window.workflow.calibration
    c.middle = deepcopy(MIDDLE)
    c.middle_confirmed = True
    c.configure_lanes(2)
    window.open_source({'kind':'file','path':str(tmp_path/'missing.mp4')})
    window.on_frame(old)
    window.on_status('old success', True, old_id)
    assert window.current_frame is None and not window.source_ok
    qtbot.waitUntil(lambda: '入力異常' in window.input_label.text(), timeout=8000)
    assert window.workflow.calibration.middle == MIDDLE
    assert len(window.workflow.calibration.lanes) == 2
    assert not window.analyze_button.isEnabled()
    assert window.save()


def test_restart_restore_and_controls(qtbot, window, calibration, video_path):
    calibration.configure_lanes(4)
    calibration.opacity = .45
    calibration.source = {'kind':'file', 'path':str(video_path)}
    calibration.start_ms = 1000
    calibration.lanes[2].name = 'レッド内側（変更済み）'
    for lane in calibration.lanes:
        lane.confirmed = True
    window.workflow.calibration = calibration
    window.sync_settings()
    assert window.save()
    window.workflow.calibration = Calibration()
    assert window.restore(window.database)
    qtbot.waitUntil(lambda: window.source_ok and window.current_frame.timestamp_ms >= 900, timeout=8000)
    assert window.opacity.value() == 45
    assert window.start_time.value() == 1
    assert window.lane_count.value() == 4
    assert window.workflow.calibration.lanes[2].name.endswith('（変更済み）')
    assert all(x.confirmed for x in window.workflow.calibration.lanes)
    assert not window.view.handles
    assert window.analyze_button.isEnabled() == Path(window.analysis_panel.model.text()).is_file()


def test_mirror_undo_and_no_silent_reset(window, calibration):
    calibration.middle[0][0] = .1
    original = deepcopy(calibration.middle)
    calibration.configure_lanes(3)
    window.workflow.calibration = calibration
    window.refresh()
    window.mirror.setChecked(True)
    assert calibration.mirrored
    assert not calibration.middle_confirmed and not calibration.lanes
    np.testing.assert_allclose(calibration.middle[1], [.9, original[0][1]])
    window.undo()
    restored = window.workflow.calibration
    assert not restored.mirrored and restored.middle_confirmed
    assert len(restored.lanes) == 3 and restored.middle == original


def test_enumeration_parses_only_video(monkeypatch):
    from mogul import video
    class Result:
        stderr = b'''[dshow] DirectShow video devices\n[dshow] "Integrated Camera"\n[dshow] Alternative name "unique-video-id"\n[dshow] DirectShow audio devices\n[dshow] "Microphone"\n[dshow] Alternative name "audio-id"\n'''
    monkeypatch.setattr(video.subprocess, 'run', lambda *a, **k: Result())
    assert video.enumerate_uvc_devices() == [{'name':'Integrated Camera', 'device':'unique-video-id'}]


def test_uvc_worker_recording_simulated(qtbot, tmp_path):
    class FakeUVC:
        live = True
        fps = 30
        metadata = {'kind':'uvc','width':64,'height':48,'fps':30,'duration_ms':0,'live':True}

        def read(self):
            time.sleep(.025)
            return VideoFrame(np.zeros((48,64,3), np.uint8), 0, 0)

        def close(self):
            pass

    worker = VideoWorker(factory=lambda spec: FakeUVC())
    recordings, frames = [], []
    worker.recording.connect(lambda text, request_id: recordings.append(text))
    worker.frame_ready.connect(frames.append)
    worker.start()
    path = tmp_path/'live.mp4'
    try:
        worker.command('open', {'kind':'uvc','_request_id':7})
        qtbot.waitUntil(lambda: bool(frames), timeout=3000)
        worker.command('record', str(path))
        qtbot.waitUntil(lambda: any(x.startswith('録画中:') for x in recordings), timeout=3000)
        qtbot.wait(200)
        worker.command('stop_record')
        qtbot.waitUntil(lambda: any(x.startswith('保存済み:') for x in recordings), timeout=3000)
        assert all(x.source_id == 7 for x in frames)
        assert all(a.timestamp_ms <= b.timestamp_ms for a,b in zip(frames,frames[1:]))
    finally:
        worker.requestInterruption()
        assert worker.wait(3000)
    source = FileVideoSource(path)
    try:
        assert source.count >= 3 and source.read() is not None
    finally:
        source.close()
