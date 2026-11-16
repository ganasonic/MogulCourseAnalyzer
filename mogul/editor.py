"""Interactive overlays: click confirmation is deferred to distinguish double clicks."""
import math
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsScene, QGraphicsView
from .geometry import contains, valid_quad
from .landmarks import ControlGateEditor, BumpRegistry, layout_ready


class MiddlePartEditor:
    def __init__(self, workflow):
        self.workflow = workflow

    def move(self, corner, point):
        points = [p[:] for p in self.workflow.calibration.middle]
        points[corner] = list(point)
        return self.workflow.middle_changed(points)


class LaneEditor:
    def __init__(self, workflow):
        self.workflow = workflow

    def move(self, lane, corner, point):
        return self.workflow.calibration.move_lane_corner(lane, corner, point)


class CalibrationView(QGraphicsView):
    changed = Signal()
    notice = Signal(str)
    landmark_selected = Signal(str, str)
    landmark_add_requested = Signal(str, object)

    def __init__(self, workflow, parent=None):
        super().__init__(parent)
        self.workflow = workflow
        self.setScene(QGraphicsScene(self))
        self.image_item = self.scene().addPixmap(QPixmap())
        self.image_item.setZValue(-10)
        self.overlays = []
        self.handles = []
        self.width_px, self.height_px = 640, 360
        self.scene().setSceneRect(0, 0, self.width_px, self.height_px)
        self.mode = 'middle'
        self.lane_index = 0
        self.has_frame = False
        self.landmark_tool = 'select'
        self.selected_landmark = None
        self.landmark_markers = []
        self.pending_bump_polygon = None
        self.drag = None
        self.press = None
        self.suppress_release = False
        self.click_timer = QTimer(self)
        self.click_timer.setSingleShot(True)
        self.click_timer.timeout.connect(self.confirm)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor('#17202b'))
        self.setMinimumSize(480, 270)
        self.setMouseTracking(True)

    @property
    def calibration(self):
        return self.workflow.calibration

    def set_frame(self, image):
        height, width = image.shape[:2]
        resized = (width, height) != (self.width_px, self.height_px)
        self.width_px, self.height_px = width, height
        self.scene().setSceneRect(0, 0, width, height)
        rgb = image[:, :, ::-1].copy()
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        self.image_item.setPixmap(QPixmap.fromImage(qimage))
        if not self.has_frame or resized:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            self.redraw()
        self.has_frame = True

    def clear_frame(self):
        self.click_timer.stop()
        self.has_frame = False
        self.image_item.setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def set_mode(self, mode, lane=0):
        self.click_timer.stop()
        self.drag = None
        self.pending_bump_polygon = None
        self.mode = mode
        self.lane_index = lane
        self.redraw()

    def polygon(self):
        c = self.calibration
        if self.mode == 'middle':
            return c.middle
        if self.mode == 'landmarks':
            return []
        if 0 <= self.lane_index < len(c.lanes):
            return c.lane_polygon(self.lane_index)
        return []

    def confirmed(self):
        c = self.calibration
        return c.middle_confirmed if self.mode == 'middle' else (
            0 <= self.lane_index < len(c.lanes) and c.lanes[self.lane_index].confirmed)

    def redraw(self):
        for item in self.overlays:
            self.scene().removeItem(item)
        self.overlays.clear()
        self.handles.clear()
        c = self.calibration
        if c.middle:
            self._polygon(c.middle, c.middle_color, c.opacity, c.middle_confirmed)
        for i, lane in enumerate(c.lanes):
            self._polygon(c.lane_polygon(i), lane.color, c.opacity, lane.confirmed)
            center = np.mean(c.lane_polygon(i), axis=0)
            text = self.scene().addText(f'{lane.id}: {lane.name} ' + ('✓' if lane.confirmed else '編集中'))
            text.setDefaultTextColor(QColor('white'))
            text.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            text.setPos(float(center[0]*self.width_px), float(center[1]*self.height_px))
            self.overlays.append(text)
        if self.polygon() and not self.confirmed():
            for i, point in enumerate(self.polygon()):
                pen = QPen(QColor('white'))
                pen.setCosmetic(True)
                item = self.scene().addEllipse(-7, -7, 14, 14, pen, QColor('#ffbe55'))
                item.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                item.setPos(point[0]*self.width_px, point[1]*self.height_px)
                item.setZValue(10)
                self.overlays.append(item)
                self.handles.append((i, item))
        self._draw_landmarks()

    def select_landmark(self, kind, identifier):
        self.selected_landmark = (kind,identifier)
        self.set_mode('landmarks')

    def _draw_landmarks(self):
        self.landmark_markers = []
        c = self.calibration
        for gate in c.control_gates:
            label = f'{"左" if gate.side == "LEFT" else "右"}G{gate.number if gate.number is not None else "?"}'
            self._landmark_marker('gate',gate.id,gate.point,label,'#64b5ff' if gate.side == 'LEFT' else '#ff8c96',gate.confirmed)
        for bump in c.bumps:
            color = next((lane.color for lane in c.lanes if lane.id == bump.lane_id),'#c4cbd0')
            self._polygon(bump.polygon,color,.18,bump.confirmed)
            self._landmark_marker('bump',bump.id,bump.center,bump.label,color,bump.confirmed)
            if self.mode == 'landmarks' and self.landmark_tool == 'resize_bump' and self.selected_landmark == ('bump',bump.id):
                for corner,point in enumerate(bump.polygon):
                    item = self.scene().addEllipse(-5,-5,10,10,QPen(QColor('white')),QColor('#ffbe55'))
                    item.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                    item.setPos(point[0]*self.width_px,point[1]*self.height_px)
                    item.setZValue(10)
                    self.overlays.append(item)
                    self.handles.append((corner,item))
        if self.pending_bump_polygon:
            self._polygon(self.pending_bump_polygon,'#ffbe55',.3,False)

    def _landmark_marker(self, kind, identifier, point, label, color, confirmed):
        pen = QPen(QColor(color),3 if self.selected_landmark == (kind,identifier) else 2,
                   Qt.SolidLine if confirmed else Qt.DashLine)
        pen.setCosmetic(True)
        item = self.scene().addEllipse(-7,-7,14,14,pen,QColor(color) if confirmed else QColor('#18212d'))
        item.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        item.setPos(point[0]*self.width_px,point[1]*self.height_px)
        item.setZValue(5)
        text = self.scene().addText(label + ('' if confirmed else ' ?'))
        text.setDefaultTextColor(QColor('white'))
        text.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        text.setPos(point[0]*self.width_px+10,point[1]*self.height_px-10)
        text.setZValue(5)
        self.overlays.extend([item,text])
        self.landmark_markers.append((kind,identifier,item))

    def _polygon(self, points, color, opacity, confirmed):
        color = QColor(color)
        if not color.isValid():
            color = QColor('#78d6eb')
        fill = QColor(color)
        fill.setAlphaF(opacity)
        pen = QPen(color, 2, Qt.SolidLine if confirmed else Qt.DashLine)
        pen.setCosmetic(True)
        item = self.scene().addPolygon(QPolygonF([QPointF(p[0]*self.width_px, p[1]*self.height_px) for p in points]), pen, fill)
        self.overlays.append(item)

    def normalized_position(self, event, clamp=True):
        p = self.mapToScene(event.position().toPoint())
        point = [p.x()/self.width_px, p.y()/self.height_px]
        return np.clip(point, 0, 1).tolist() if clamp else point

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or not self.has_frame:
            return super().mousePressEvent(event)
        self.click_timer.stop()
        self.press = event.position()
        self.suppress_release = False
        raw = self.normalized_position(event, False)
        if not all(0 <= v <= 1 for v in raw):
            self.press = None
            return
        if self.mode == 'landmarks' and not layout_ready(self.calibration):
            self.notice.emit('先にミドルパートと全レーンを確定してください。')
            self.press = None
            return
        for corner, item in self.handles:
            pos = self.mapFromScene(item.pos())
            if math.hypot(pos.x()-event.position().x(), pos.y()-event.position().y()) < 13:
                self.suppress_release = True
                self.workflow.checkpoint()
                self.drag = ('bump_corner' if self.mode == 'landmarks' else 'corner', corner)
                return
        if self.mode == 'landmarks':
            for kind,identifier,item in reversed(self.landmark_markers):
                pos = self.mapFromScene(item.pos())
                if math.hypot(pos.x()-event.position().x(),pos.y()-event.position().y()) < 13:
                    self.workflow.checkpoint()
                    self.selected_landmark = (kind,identifier)
                    self.drag = (kind,identifier)
                    self.suppress_release = True
                    self.landmark_selected.emit(kind,identifier)
                    self.redraw()
                    return
            if self.landmark_tool == 'add_bump':
                self.drag = ('bump_rectangle',raw)
                self.suppress_release = True
            return
        if self.mode == 'middle' and not self.calibration.middle:
            self.workflow.checkpoint()
            self.drag = ('rectangle', self.normalized_position(event))
            self.suppress_release = True

    def mouseMoveEvent(self, event):
        if self.drag:
            point = self.normalized_position(event)
            kind, value = self.drag
            if kind in ('rectangle','bump_rectangle'):
                x0, y0 = value
                x1, y1 = point
                points = [[min(x0,x1),min(y0,y1)], [max(x0,x1),min(y0,y1)],
                          [max(x0,x1),max(y0,y1)], [min(x0,x1),max(y0,y1)]]
                if kind == 'rectangle':
                    self.workflow.middle_changed(points)
                else:
                    self.pending_bump_polygon = points
            elif kind in ('gate','bump'):
                service = ControlGateEditor(self.calibration) if kind == 'gate' else BumpRegistry(self.calibration)
                if not service.move(value,point):
                    self.notice.emit('所属レーン・ゲート側・領域内の制約を満たす位置に移動してください。')
            elif kind == 'bump_corner':
                if not self.selected_landmark or not BumpRegistry(self.calibration).resize(self.selected_landmark[1],value,point):
                    self.notice.emit('コブ領域は所属レーン内の有効な四角形にしてください。')
            elif self.mode == 'middle':
                if not MiddlePartEditor(self.workflow).move(value, point):
                    self.notice.emit('自己交差・面積ゼロになる四角形には変更できません。')
            else:
                if not LaneEditor(self.workflow).move(self.lane_index, value, point):
                    self.notice.emit('レーンの重複・自己交差・ミドルパート範囲外になる変更はできません。ゲート側との余白は設定できます。')
                else:
                    self.workflow.dirty = True
            self.redraw()

    def mouseReleaseEvent(self, event):
        was_drag = self.drag is not None
        if self.drag and self.drag[0] == 'bump_rectangle' and self.pending_bump_polygon:
            polygon = self.pending_bump_polygon
            self.pending_bump_polygon = None
            self.landmark_add_requested.emit('add_bump',polygon)
            self.redraw()
        self.drag = None
        if was_drag:
            self.changed.emit()
        if self.suppress_release:
            self.suppress_release = False
            return
        if event.button() == Qt.LeftButton and self.press is not None and (event.position()-self.press).manhattanLength() < 4:
            raw = self.normalized_position(event, False)
            if self.mode == 'landmarks':
                if all(0 <= v <= 1 for v in raw) and self.landmark_tool in ('add_left','add_right'):
                    self.landmark_add_requested.emit(self.landmark_tool,raw)
                return
            if all(0 <= v <= 1 for v in raw) and not self.confirmed():
                self.click_timer.start(QApplication.doubleClickInterval()+30)

    def confirm(self):
        c = self.calibration
        if self.mode == 'middle' and valid_quad(c.middle):
            self.workflow.checkpoint()
            c.middle_confirmed = True
        elif self.mode == 'lane' and c.lanes_valid():
            self.workflow.checkpoint()
            c.lanes[self.lane_index].confirmed = True
        else:
            return
        self.redraw()
        self.changed.emit()

    def mouseDoubleClickEvent(self, event):
        self.click_timer.stop()
        self.suppress_release = True
        self.drag = None
        point = self.normalized_position(event, False)
        c = self.calibration
        if event.button() != Qt.LeftButton or not self.has_frame:
            return
        if self.mode == 'landmarks':
            return
        # Lanes take priority in lane mode; clicking an unselected lane selects it.
        if self.mode == 'lane':
            for i, lane in enumerate(c.lanes):
                if contains(c.lane_polygon(i), point):
                    self.workflow.checkpoint()
                    self.lane_index = i
                    lane.confirmed = False
                    c.invalidate_landmarks()
                    break
        elif c.middle and contains(c.middle, point):
            self.workflow.checkpoint()
            c.middle_confirmed = False
            c.clear_lanes()
            self.notice.emit('ミドルパートを再編集します。レーンは再設定が必要です（Undoで戻せます）。')
        self.redraw()
        self.changed.emit()
        event.accept()
