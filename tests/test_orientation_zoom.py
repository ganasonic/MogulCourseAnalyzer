"""Source orientation, metadata, viewport navigation and normalized editing."""
from copy import deepcopy
import subprocess
import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from mogul.app import MainWindow
from mogul.editor import CalibrationView
from mogul.landmarks import ControlGateEditor,BumpRegistry
from mogul.models import Calibration
from mogul.orientation import oriented_frame,rotate_calibration,rotated_point
from mogul.repository import EventRepository
from mogul.source_selector import VideoSourceSelector
from mogul.video import FileVideoSource,VideoFrame
from mogul.workflow import CalibrationWorkflow


@pytest.mark.parametrize('angle',[0,90,180,270])
@pytest.mark.parametrize('mirror',[False,True])
def test_orientation_pixels(angle,mirror):
    image = np.arange(6*10*3,dtype=np.uint8).reshape(6,10,3)
    expected = image.copy() if angle == 0 else cv2.rotate(image,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
    if mirror:
        expected = cv2.flip(expected,1)
    np.testing.assert_array_equal(oriented_frame(image,angle,mirror),expected)
    assert oriented_frame(image,angle,mirror).flags.c_contiguous


def populated():
    c = Calibration(middle=[[.05,.05],[.95,.05],[.95,.95],[.05,.95]],middle_confirmed=True)
    c.configure_lanes(2)
    for i,j,p in [(0,0,[.2,.05]),(0,3,[.2,.95]),(1,1,[.8,.05]),(1,2,[.8,.95])]:
        assert c.move_lane_corner(i,j,p)
    for lane in c.lanes:
        lane.confirmed = True
    service=ControlGateEditor(c)
    gate=service.add('LEFT',[.12,.3],candidate_point=[.11,.29])
    service.update(gate.id,'LEFT',3)
    service.confirm()
    registry=BumpRegistry(c)
    for lane,x in [('A',.35),('B',.65)]:
        b=registry.add(lane,[[x-.02,.4],[x+.02,.4],[x+.02,.45],[x-.02,.45]],3)
        registry.update(b.id,lane,3,1,'L')
    registry.confirm()
    return c


@pytest.mark.parametrize('mirror',[False,True])
def test_rotation_tracks_all_coordinates_and_roundtrips(tmp_path,mirror):
    c=populated()
    c.mirrored=mirror
    original=deepcopy(c)
    for rotation in [90,180,270,0]:
        rotate_calibration(c,rotation)
        assert c.lanes_valid() and c.gates_confirmed and c.bumps_confirmed
        np.testing.assert_allclose(c.control_gates[0].point,rotated_point(original.control_gates[0].point,rotation,mirror))
        np.testing.assert_allclose(c.control_gates[0].candidate_point,rotated_point(original.control_gates[0].candidate_point,rotation,mirror))
        assert c.control_gates[0].side == 'LEFT' and c.bumps[0].side == 'L'
        assert c.bumps[0].id == original.bumps[0].id
        path=tmp_path/f'{rotation}.sqlite'
        repo=EventRepository(path)
        repo.save(c)
        _,restored=repo.load()
        assert restored.rotation==rotation and restored.gates_confirmed and restored.bumps_confirmed
    np.testing.assert_allclose(c.middle,original.middle,atol=1e-12)
    np.testing.assert_allclose(c.boundaries,original.boundaries,atol=1e-12)
    np.testing.assert_allclose(c.bumps[0].polygon,original.bumps[0].polygon,atol=1e-12)
    data=c.to_dict()
    data.pop('rotation')
    assert Calibration.from_dict(data).rotation==0
    data['rotation']=45
    with pytest.raises(ValueError):
        Calibration.from_dict(data)


@pytest.fixture
def view(qtbot):
    v=CalibrationView(CalibrationWorkflow(Calibration()))
    qtbot.addWidget(v)
    v.resize(800,600)
    v.show()
    qtbot.waitExposed(v)
    v.set_frame(np.zeros((360,640,3),np.uint8))
    yield v
    v.click_timer.stop()


def test_zoom_wheel_anchor_pan_and_precise_edit(qtbot,view):
    view.calibration.middle=[[.4,.3],[.6,.3],[.6,.6],[.4,.6]]
    view.redraw()
    data=deepcopy(view.calibration.to_dict())
    view.focus_middle()
    assert view.zoom_factor>2
    assert view.calibration.to_dict()==data
    anchor=QPoint(view.viewport().width()//2+30,view.viewport().height()//2+10)
    before=view.mapToScene(anchor)
    event=QWheelEvent(QPointF(anchor),QPointF(view.viewport().mapToGlobal(anchor)),QPoint(),QPoint(0,120),Qt.NoButton,Qt.NoModifier,Qt.ScrollUpdate,False)
    QApplication.sendEvent(view.viewport(),event)
    after=view.mapToScene(anchor)
    assert (before-after).manhattanLength()<2
    center=view.viewport().rect().center()
    before=view.mapToScene(center)
    qtbot.mousePress(view.viewport(),Qt.MiddleButton,pos=center)
    QTest.mouseMove(view.viewport(),center+QPoint(40,30),delay=30)
    qtbot.mouseRelease(view.viewport(),Qt.MiddleButton,pos=center+QPoint(40,30))
    assert (view.mapToScene(center)-before).manhattanLength()>5
    assert view.calibration.to_dict()==data and not view.click_timer.isActive()
    view.setFocus()
    qtbot.keyPress(view,Qt.Key_Space)
    qtbot.mousePress(view.viewport(),Qt.LeftButton,pos=center)
    QTest.mouseMove(view.viewport(),center-QPoint(20,15),delay=30)
    qtbot.mouseRelease(view.viewport(),Qt.LeftButton,pos=center-QPoint(20,15))
    qtbot.keyRelease(view,Qt.Key_Space)
    assert view.calibration.to_dict()==data and not view.click_timer.isActive()
    # A marker remains editable in normalized source coordinates while zoomed.
    start=view.mapFromScene(.4*640,.3*360)
    end=view.mapFromScene(.42*640,.32*360)
    qtbot.mousePress(view.viewport(),Qt.LeftButton,pos=start)
    QTest.mouseMove(view.viewport(),end,delay=30)
    qtbot.mouseRelease(view.viewport(),Qt.LeftButton,pos=end)
    np.testing.assert_allclose(view.calibration.middle[0],[.42,.32],atol=.003)
    view.set_frame(np.zeros((360,640,3),np.uint8))
    assert view.zoom_factor>2
    view.resize(850,620)
    qtbot.wait(30)
    assert view.zoom_factor>2
    view.fit_all()
    assert view.zoom_factor==1


def test_create_initial_middle_while_zoomed(qtbot,view):
    view.set_zoom(3)
    center=view.viewport().rect().center()
    qtbot.keyPress(view,Qt.Key_Space)
    qtbot.mousePress(view.viewport(),Qt.LeftButton,pos=center)
    QTest.mouseMove(view.viewport(),center+QPoint(20,10),delay=30)
    qtbot.mouseRelease(view.viewport(),Qt.LeftButton,pos=center+QPoint(20,10))
    qtbot.keyRelease(view,Qt.Key_Space)
    assert not view.calibration.middle and not view.workflow.dirty
    start=view.mapFromScene(.45*640,.4*360)
    end=view.mapFromScene(.55*640,.6*360)
    qtbot.mousePress(view.viewport(),Qt.LeftButton,pos=start)
    QTest.mouseMove(view.viewport(),end,delay=30)
    qtbot.mouseRelease(view.viewport(),Qt.LeftButton,pos=end)
    np.testing.assert_allclose(view.calibration.middle,[[.45,.4],[.55,.4],[.55,.6],[.45,.6]],atol=.003)


@pytest.fixture
def window(qtbot,tmp_path):
    w=MainWindow(tmp_path/'orientation.sqlite',startup=False)
    qtbot.addWidget(w,before_close_func=lambda w:setattr(w.workflow,'dirty',False))
    w.show()
    yield w
    if w.landmarks_panel.busy:
        w.landmarks_panel.task.requestInterruption()
        w.landmarks_panel.task.wait(5000)
        qtbot.wait(30)
    w.workflow.dirty=False
    w.worker.requestInterruption()
    assert w.worker.wait(5000)
    w.close()


@pytest.mark.parametrize('size,rotation,expected',[((160,90),0,(160,90)),((160,90),90,(90,160)),((90,160),0,(90,160))])
def test_three_video_types_replay_seek_rotation_restore(qtbot,window,tmp_path,size,rotation,expected):
    path=tmp_path/'input.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,size)
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.full((size[1],size[0],3),50,np.uint8))
    writer.release()
    window.open_source({'kind':'file','path':str(path),'rotation':rotation},reset=True)
    qtbot.waitUntil(lambda:window.source_ok and window.current_frame is not None,timeout=8000)
    assert (window.view.width_px,window.view.height_px)==expected
    assert window.rotation.currentData()==rotation
    window.view.set_zoom(3)
    window.worker.command('play',True)
    qtbot.waitUntil(lambda:window.current_frame.index>=3,timeout=5000)
    assert window.view.zoom_factor==3
    window.worker.command('play',False)
    window.worker.command('seek',1000)
    qtbot.waitUntil(lambda:abs(window.current_frame.timestamp_ms-1000)<100,timeout=5000)
    assert (window.view.width_px,window.view.height_px)==expected
    assert window.save()
    assert window.restore(window.database)
    qtbot.waitUntil(lambda:window.source_ok,timeout=8000)
    assert window.workflow.calibration.rotation==rotation
    assert (window.view.width_px,window.view.height_px)==expected


def test_rotation_undo_and_detector_source(qtbot,window,monkeypatch):
    from mogul.landmarks import ControlGateDetector,GateDetectionResult
    c=populated()
    window.workflow.calibration=c
    frame=np.zeros((360,640,3),np.uint8)
    window.on_frame(VideoFrame(frame,100,1,window.request_id))
    before=deepcopy(c)
    window.rotation.setCurrentIndex(window.rotation.findData(90))
    assert (window.view.width_px,window.view.height_px)==(360,640)
    np.testing.assert_allclose(c.bumps[0].center,rotated_point(before.bumps[0].center,90))
    shapes=[]
    def detect(self,frames,c,cancelled=lambda:False):
        shapes.extend(image.shape[:2] for image in frames)
        return GateDetectionResult([], '検証用候補なし')
    monkeypatch.setattr(ControlGateDetector,'detect',detect)
    window.landmarks_panel.detect()
    qtbot.waitUntil(lambda:not window.landmarks_panel.busy,timeout=5000)
    assert shapes==[(640,360)]
    window.undo()  # candidate status
    window.undo()  # rotation
    assert window.workflow.calibration.rotation==0 and window.rotation.currentData()==0
    assert window.frames and window.frames[-1].shape[:2]==(360,640)
    np.testing.assert_allclose(window.workflow.calibration.bumps[0].center,before.bumps[0].center)


def test_selector_rotation(qtbot):
    dialog=VideoSourceSelector()
    qtbot.addWidget(dialog)
    dialog.path.setText('sideways.mp4')
    dialog.rotation.setCurrentIndex(dialog.rotation.findData(270))
    dialog.accept_source()
    assert dialog.spec['rotation']==270


def test_smartphone_metadata_orientation(tmp_path):
    raw=tmp_path/'raw.mp4'
    rotated=tmp_path/'metadata.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,90))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((90,160,3),np.uint8))
    writer.release()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(raw),'-c','copy','-metadata:s:v:0','rotate=90',str(rotated)],check=True,timeout=30)
    source=FileVideoSource(rotated)
    try:
        frame=source.read()
        assert source.metadata['orientation_auto']
        assert abs(source.metadata['metadata_rotation']) in (90,270)
        assert frame.image.shape[:2]==(160,90)
        assert oriented_frame(frame.image,0).shape[:2]==(160,90)  # no double rotation
    finally:
        source.close()
