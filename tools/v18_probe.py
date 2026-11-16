"""Standalone environment probe, deliberately separate from application code."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene,
    QGraphicsPolygonItem, QGraphicsEllipseItem, QGraphicsItem, QGraphicsSceneMouseEvent)


def normalized(points, size):
    return np.asarray(points, dtype=float) / np.asarray(size)


def pixels(points, size):
    return np.asarray(points, dtype=float) * np.asarray(size)


def valid_quad(points):
    # Phase 1 probe accepts convex, nondegenerate quadrilaterals only.
    p = np.asarray(points, np.float32)
    return p.shape == (4, 2) and bool(cv2.isContourConvex(p)) and abs(cv2.contourArea(p)) > 1e-6


def overlap(a, b):
    return cv2.intersectConvexConvex(np.asarray(a, np.float32), np.asarray(b, np.float32))[0]


class Handle(QGraphicsEllipseItem):
    def __init__(self, owner, point):
        super().__init__(-7, -7, 14, 14)
        self.owner = owner
        self.setBrush(QColor('orange'))
        self.setZValue(2)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setPos(point)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.owner.update_polygon()
        return super().itemChange(change, value)


class Probe(QGraphicsView):
    confirmed = Signal()
    reedited = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle('v1.8 environment probe - drag corners / click / double-click')
        self.setScene(QGraphicsScene(0, 0, 640, 360))
        image = QImage(640, 360, QImage.Format_RGB32)
        image.fill(QColor('#526b80'))
        painter = QPainter(image)
        painter.setPen(QColor('white'))
        painter.drawText(20, 30, 'Synthetic video frame / environment verification')
        painter.end()
        self.scene().addPixmap(QPixmap.fromImage(image))
        self.poly = QGraphicsPolygonItem()
        self.poly.setBrush(QColor(0, 255, 120, 90))
        self.scene().addItem(self.poly)
        self.handles = []
        self.lane_count = 2
        self.lanes = []
        for p in [(180, 70), (460, 70), (580, 300), (60, 300)]:
            h = Handle(self, QPointF(*p))
            self.handles.append(h)
            self.scene().addItem(h)
        self.locked = False
        self.lane_count = 2
        self.lanes = []
        self.update_polygon()
        self.resize(680, 410)

    def points(self):
        return [[h.pos().x(), h.pos().y()] for h in self.handles]

    def update_polygon(self):
        self.poly.setPolygon(QPolygonF([h.pos() for h in self.handles]))
        for item in self.lanes:
            self.scene().removeItem(item)
        self.lanes = []
        if len(self.handles) != 4:
            return
        a, b, c, d = np.asarray(self.points())
        for i in range(self.lane_count):
            u, v = i / self.lane_count, (i + 1) / self.lane_count
            vertices = [a+(b-a)*u, a+(b-a)*v, d+(c-d)*v, d+(c-d)*u]
            item = self.scene().addPolygon(QPolygonF([QPointF(*p) for p in vertices]))
            item.setBrush(QColor(40+i*45, 100, 255, 45))
            self.lanes.append(item)

    def set_lanes(self, count):
        if count not in (2, 3, 4):
            raise ValueError('lane count must be 2, 3 or 4')
        self.lane_count = count
        self.update_polygon()

    @Slot()
    def confirm(self):
        if valid_quad(self.points()):
            self.locked = True
            for h in self.handles:
                h.setFlag(QGraphicsItem.ItemIsMovable, False)
            self.confirmed.emit()

    def mousePressEvent(self, event):
        self.press_position = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and (event.position()-self.press_position).manhattanLength() < 3:
            self.confirm()

    def mouseDoubleClickEvent(self, event):
        self.locked = False
        for h in self.handles:
            h.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.reedited.emit()
        # Avoid treating the release of the second click as a new confirmation.
        self.press_position = event.position() + QPointF(100, 100)
        event.accept()

    def save(self, path):
        Path(path).write_text(json.dumps({'version': 1, 'points': normalized(self.points(), (640, 360)).tolist(), 'lanes': self.lane_count}), encoding='utf-8')

    def restore(self, path):
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        for h, p in zip(self.handles, pixels(data['points'], (640, 360))):
            h.setPos(QPointF(*p))
        self.set_lanes(data['lanes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=0)
    args = parser.parse_args()
    app = QApplication([])
    view = Probe()
    view.show()
    if args.seconds:
        QTimer.singleShot(int(args.seconds * 1000), app.quit)
    raise SystemExit(app.exec())
