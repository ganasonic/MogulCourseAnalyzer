"""Selectable schematic course map. It has no metric distance claim."""
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsItem
from .landmarks import course_position


class CourseMap(QGraphicsView):
    selected = Signal(str,str)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor('#18212d'))
        self.setRenderHint(QPainter.Antialiasing)
        self.setMinimumWidth(300)
        self.analysis_events=[]

    def update_course(self,c,selected=None,tracks=None):
        self.scene().clear()
        if not c.middle:
            self.scene().addText('ミドルパート設定後に表示します').setDefaultTextColor(QColor('white'))
            return
        try:
            course_position(c,c.middle[0])
            project = lambda p: course_position(c,p)
            title = '上流 ↑  /  俯瞰模式図（距離縮尺なし）'
        except ValueError:
            project = lambda p:p
            title = '画像座標の模式図（透視変換不可）'
        def position(p):
            x,y = project(p)
            return QPointF(20+x*320,45+y*540)
        text = self.scene().addText(title)
        text.setDefaultTextColor(QColor('white'))
        def polygon(points,color):
            fill = QColor(color)
            fill.setAlphaF(.25)
            self.scene().addPolygon(QPolygonF([position(p) for p in points]),QPen(QColor(color)),fill)
        polygon(c.middle,c.middle_color)
        for i,lane in enumerate(c.lanes):
            polygon(c.lane_polygon(i),lane.color)
        def marker(kind,item,point,label,color):
            p = position(point)
            pen = QPen(QColor('white' if selected == (kind,item.id) else color),3,
                       Qt.SolidLine if item.confirmed else Qt.DashLine)
            ellipse = self.scene().addEllipse(-5,-5,10,10,pen,QColor(color))
            ellipse.setPos(p)
            text = self.scene().addText(label+('' if item.confirmed else ' ?'))
            text.setDefaultTextColor(QColor('white'))
            text.setPos(p+QPointF(7,-12))
            for graphic in (ellipse,text):
                graphic.setData(0,kind)
                graphic.setData(1,item.id)
                graphic.setZValue(3)
        for gate in c.control_gates:
            marker('gate',gate,gate.point,f'{"左" if gate.side == "LEFT" else "右"}G{gate.number or "?"}',
                   '#64b5ff' if gate.side == 'LEFT' else '#ff8c96')
        for bump in c.bumps:
            polygon(bump.polygon,'#ffbe55')
            marker('bump',bump,bump.center,bump.label,'#ffbe55')
        for track in tracks or []:
            p=position(track['point'])
            item=self.scene().addEllipse(-5,-5,10,10,QPen(QColor('#8bff72')),QColor('#8bff72'))
            item.setPos(p); item.setZValue(10)
            text=self.scene().addText(f"T{track['track_id']} / {track.get('lane_id') or '?'}")
            text.setDefaultTextColor(QColor('#8bff72'))
            text.setPos(p+QPointF(8,-12)); text.setZValue(10)
        for event in self.analysis_events:
            p=position(event['point'])
            color=QColor('#ff5252' if event['kind']=='course_out' else '#ffa726')
            pen=QPen(color,3,Qt.SolidLine if event.get('review')=='confirmed' else Qt.DashLine)
            item=self.scene().addEllipse(-8,-8,16,16,pen); item.setPos(p); item.setZValue(12)
            text=self.scene().addText(('逸脱' if event['kind']=='course_out' else '姿勢候補')+f" {event['timestamp_ms']/1000:.1f}s")
            text.setDefaultTextColor(color); text.setPos(p+QPointF(10,-10)); text.setZValue(12)
        self.scene().setSceneRect(self.scene().itemsBoundingRect().adjusted(-15,-15,15,15))
        self.fitInView(self.sceneRect(),Qt.KeepAspectRatio)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(),Qt.KeepAspectRatio)

    def mousePressEvent(self,event):
        item = self.itemAt(event.position().toPoint())
        if item and item.data(0) in ('gate','bump'):
            self.selected.emit(item.data(0),item.data(1))
        super().mousePressEvent(event)
