"""Interactive overlays: click confirmation is deferred to distinguish double clicks."""
import math
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QApplication, QGraphicsItem, QGraphicsScene, QGraphicsView
from .geometry import contains, valid_quad
from .landmarks import ControlGateEditor, BumpRegistry, layout_ready
from .course_region import in_course


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
    zoom_changed = Signal(float)

    def __init__(self, workflow, parent=None):
        super().__init__(parent)
        self.workflow = workflow
        self.setScene(QGraphicsScene(self))
        self.image_item = self.scene().addPixmap(QPixmap())
        self.image_item.setTransformationMode(Qt.SmoothTransformation)
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
        self.analysis_tracks = []
        self.analysis_events = []
        self.analysis_items = []
        self.show_analysis_geometry = False
        self.drag = None
        self.press = None
        self.zoom_factor = 1.0
        self.base_scale = 1.0
        self.panning = False
        self.space_pressed = False
        self.pan_position = None
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setToolTip('ホイール: 拡大・縮小 / 中ボタンまたはSpace+左ドラッグ: 表示位置を移動')
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
            self.fit_all()
            self.redraw()
        self.has_frame = True

    def clear_frame(self):
        self.click_timer.stop()
        self.has_frame = False
        self.image_item.setPixmap(QPixmap())
        self.drag = None
        self.panning = self.space_pressed = False
        self.fit_all()

    def resizeEvent(self, event):
        center = self.mapToScene(self.viewport().rect().center())
        super().resizeEvent(event)
        self._refit(center)

    def _refit(self, center=None):
        self.resetTransform()
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self.base_scale = self.transform().m11()
        self.scale(self.zoom_factor,self.zoom_factor)
        if center is not None and self.zoom_factor > 1:
            self.centerOn(center)

    def fit_all(self):
        self.zoom_factor = 1.0
        self._refit()
        self.zoom_changed.emit(self.zoom_factor)

    def set_zoom(self,factor,anchor=None):
        if self.drag is not None:
            return
        factor = max(1.0,min(32.0,float(factor)))
        if anchor is None:
            anchor = self.viewport().rect().center()
        before = self.mapToScene(anchor)
        self.scale(factor/self.zoom_factor,factor/self.zoom_factor)
        self.zoom_factor = factor
        after = self.mapToScene(anchor)
        center = self.mapToScene(self.viewport().rect().center())
        self.centerOn(center+before-after)
        self.zoom_changed.emit(self.zoom_factor)

    def focus_middle(self):
        points = self.calibration.middle
        if not points:
            self.notice.emit('ミドルパートを指定してから拡大できます。ホイールで先に拡大して設定することもできます。')
            return
        self.click_timer.stop()
        self.fit_all()
        xs = [p[0]*self.width_px for p in points]
        ys = [p[1]*self.height_px for p in points]
        rect = QRectF(min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys))
        rect = rect.adjusted(-rect.width()*.08,-rect.height()*.08,rect.width()*.08,rect.height()*.08)
        self.fitInView(rect,Qt.KeepAspectRatio)
        factor = self.transform().m11()/self.base_scale
        self.zoom_factor = max(1,min(32,factor))
        self._refit(rect.center())
        self.zoom_changed.emit(self.zoom_factor)

    def wheelEvent(self,event):
        self.click_timer.stop()
        if self.has_frame and event.angleDelta().y():
            self.set_zoom(self.zoom_factor*1.2**(event.angleDelta().y()/120),event.position().toPoint())
        event.accept()

    def keyPressEvent(self,event):
        if event.key() == Qt.Key_Space:
            self.space_pressed = True
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self,event):
        if event.key() == Qt.Key_Space:
            self.space_pressed = False
            if not self.panning:
                self.unsetCursor()
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self,event):
        self.space_pressed = self.panning = False
        self.unsetCursor()
        super().focusOutEvent(event)

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
        analyzing=self.mode=='analysis'
        labels=not analyzing or self.show_analysis_geometry
        opacity=min(c.opacity,.08) if analyzing else c.opacity
        if c.middle:
            self._polygon(c.middle, c.middle_color, 0 if c.lanes else opacity, c.middle_confirmed)
        for i, lane in enumerate(c.lanes):
            self._polygon(c.lane_polygon(i), lane.color, opacity, lane.confirmed)
            if not labels:
                continue
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
        if labels:
            self._draw_landmarks()
        self._draw_analysis()

    def redraw_analysis(self):
        old=set(self.analysis_items)
        for item in self.analysis_items:
            self.scene().removeItem(item)
        self.overlays=[item for item in self.overlays if item not in old]
        self._draw_analysis()

    def _draw_analysis(self):
        start=len(self.overlays)
        self._draw_tracks()
        for event in self.analysis_events:
            color='#ff5252' if event['kind']=='course_out' else '#ffa726'
            label='レーン逸脱' if event['kind']=='course_out' else '姿勢異常候補'
            pen=QPen(QColor(color),3,Qt.SolidLine if event.get('review')=='confirmed' else Qt.DashLine)
            pen.setCosmetic(True)
            point=event['point']
            item=self.scene().addEllipse(-9,-9,18,18,pen)
            item.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            item.setPos(point[0]*self.width_px,point[1]*self.height_px); item.setZValue(20)
            text=self.scene().addText(f"{label} {'✓' if event.get('review')=='confirmed' else '?'} {event['timestamp_ms']/1000:.1f}s"+(f" / 位置対応 {event['bump_label']}（原因未確認）" if event.get('bump_label') else ' / 原因コブ未特定'))
            text.setDefaultTextColor(QColor(color)); text.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            text.setPos(point[0]*self.width_px+12,point[1]*self.height_px-10); text.setZValue(20)
            self.overlays.extend([item,text])
        self.analysis_items=self.overlays[start:]

    def _draw_tracks(self):
        for track in self.analysis_tracks:
            x1,y1,x2,y2=track['bbox']
            if not in_course(self.calibration,[(x1+x2)/2,y2]):
                continue
            pen=QPen(QColor('#8bff72'),2,Qt.DashLine if track.get('interpolated') or track.get('tentative') else Qt.SolidLine)
            pen.setCosmetic(True)
            item=self.scene().addRect(x1*self.width_px,y1*self.height_px,(x2-x1)*self.width_px,(y2-y1)*self.height_px,pen)
            scale=max(.001,abs(self.transform().m11()))
            pad_x=max(0,(8/scale-(x2-x1)*self.width_px)/2)
            pad_y=max(0,(12/scale-(y2-y1)*self.height_px)/2)
            item.setRect(x1*self.width_px-pad_x,y1*self.height_px-pad_y,
                         (x2-x1)*self.width_px+2*pad_x,(y2-y1)*self.height_px+2*pad_y)
            item.setZValue(15)
            suffix=' / 追従補間' if track.get('interpolated') else ''
            score='移動対象' if track.get('detection_kind')=='motion' else f"{track['confidence']:.2f}"
            text=self.scene().addText(f"T{track['track_id']} / {track['run_id'][:6]} / {track.get('lane_id') or '?'} / {score}"+suffix)
            text.setDefaultTextColor(QColor('#8bff72'))
            text.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            text.setPos(x1*self.width_px,y1*self.height_px-22)
            text.setZValue(15)
            self.overlays.extend([item,text])

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
        if self.has_frame and (event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.space_pressed)):
            self.click_timer.stop()
            self.panning = True
            self.pan_position = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self.mode == 'analysis':
            event.accept()
            return
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
        if self.panning:
            delta = event.position()-self.pan_position
            self.pan_position = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-round(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-round(delta.y()))
            event.accept()
            return
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
        if self.panning:
            self.panning = False
            self.setCursor(Qt.OpenHandCursor if self.space_pressed else Qt.ArrowCursor)
            self.press = None
            event.accept()
            return
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
                if all(0 <= v <= 1 for v in raw) and self.landmark_tool in ('add_left','add_right','add_bump_center','add_bump_line'):
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
        if self.mode == 'analysis':
            event.accept()
            return
        if self.space_pressed or self.panning:
            event.accept()
            return
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
