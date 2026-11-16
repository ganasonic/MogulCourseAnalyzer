"""Interactive overlays: click confirmation is deferred to distinguish double clicks."""
import math
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsScene, QGraphicsView
from .geometry import contains, valid_quad


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
        self.mode = mode
        self.lane_index = lane
        self.redraw()

    def polygon(self):
        c = self.calibration
        if self.mode == 'middle':
            return c.middle
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
                fixed = self.mode == 'lane' and ((self.lane_index == 0 and i in (0, 3)) or
                    (self.lane_index == len(c.lanes)-1 and i in (1, 2)))
                pen = QPen(QColor('white'))
                pen.setCosmetic(True)
                item = self.scene().addEllipse(-7, -7, 14, 14, pen, QColor('#78838f' if fixed else '#ffbe55'))
                item.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                item.setPos(point[0]*self.width_px, point[1]*self.height_px)
                item.setZValue(10)
                self.overlays.append(item)
                self.handles.append((i, item, fixed))

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
        for corner, item, fixed in self.handles:
            pos = self.mapFromScene(item.pos())
            if math.hypot(pos.x()-event.position().x(), pos.y()-event.position().y()) < 13:
                self.suppress_release = True
                if fixed:
                    self.notice.emit('外側境界はミドルパートと共有しています。変更はミドルパート再編集で行ってください。')
                else:
                    self.workflow.checkpoint()
                    self.drag = ('corner', corner)
                return
        if self.mode == 'middle' and not self.calibration.middle:
            self.workflow.checkpoint()
            self.drag = ('rectangle', self.normalized_position(event))
            self.suppress_release = True

    def mouseMoveEvent(self, event):
        if self.drag:
            point = self.normalized_position(event)
            kind, value = self.drag
            if kind == 'rectangle':
                x0, y0 = value
                x1, y1 = point
                points = [[min(x0,x1),min(y0,y1)], [max(x0,x1),min(y0,y1)],
                          [max(x0,x1),max(y0,y1)], [min(x0,x1),max(y0,y1)]]
                self.workflow.middle_changed(points)
            elif self.mode == 'middle':
                if not MiddlePartEditor(self.workflow).move(value, point):
                    self.notice.emit('自己交差・面積ゼロになる四角形には変更できません。')
            else:
                if not LaneEditor(self.workflow).move(self.lane_index, value, point):
                    self.notice.emit('重複・空白・範囲外になる変更はできません。マーカーは上下の外周上を移動します。')
                else:
                    self.workflow.dirty = True
            self.redraw()

    def mouseReleaseEvent(self, event):
        was_drag = self.drag is not None
        self.drag = None
        if was_drag:
            self.changed.emit()
        if self.suppress_release:
            self.suppress_release = False
            return
        if event.button() == Qt.LeftButton and self.press is not None and (event.position()-self.press).manhattanLength() < 4:
            raw = self.normalized_position(event, False)
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
        # Lanes take priority in lane mode; clicking an unselected lane selects it.
        if self.mode == 'lane':
            for i, lane in enumerate(c.lanes):
                if contains(c.lane_polygon(i), point):
                    self.workflow.checkpoint()
                    self.lane_index = i
                    lane.confirmed = False
                    c.gates_confirmed = c.bumps_confirmed = False
                    break
        elif c.middle and contains(c.middle, point):
            self.workflow.checkpoint()
            c.middle_confirmed = False
            c.clear_lanes()
            self.notice.emit('ミドルパートを再編集します。レーンは再設定が必要です（Undoで戻せます）。')
        self.redraw()
        self.changed.emit()
        event.accept()
