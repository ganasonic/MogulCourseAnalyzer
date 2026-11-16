"""Phase 2 controls, separate from video playback and Phase 1 controls."""
from copy import deepcopy
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (QComboBox,QFormLayout,QGroupBox,QLabel,QListWidget,
    QPushButton,QSpinBox,QTabWidget,QVBoxLayout,QWidget,QScrollArea)
from .landmarks import ControlGateEditor, BumpRegistry, ControlGateDetector, layout_ready


class GateDetectionTask(QThread):
    result = Signal(object)

    def __init__(self,frames,calibration,parent=None):
        super().__init__(parent)
        self.frames,self.calibration = frames,calibration

    def run(self):
        try:
            result = ControlGateDetector().detect(self.frames,self.calibration,self.isInterruptionRequested)
            self.result.emit(result)
        except Exception as exc:
            self.result.emit(exc)


class LandmarkPanel(QScrollArea):
    def __init__(self,window):
        super().__init__(window)
        self.window = window
        self.selected = None
        self.task = None
        self.snapshot = None
        self.request_id = None
        self.refreshing = False
        self.setWidgetResizable(True)
        body = QWidget()
        layout = QVBoxLayout(body)
        self.explanation = QLabel('ミドルパートと全レーン確定後に設定できます。\nゲートはレーン外側の余白へ、コブはレーン内へ登録します。')
        self.explanation.setWordWrap(True)
        layout.addWidget(self.explanation)
        self.tool = QComboBox()
        for name,value in [('選択・ドラッグ移動','select'),('左ゲートをクリック追加','add_left'),
                           ('右ゲートをクリック追加','add_right'),('選択レーン内をドラッグしてコブ追加','add_bump'),
                           ('選択コブの四隅を補正','resize_bump')]:
            self.tool.addItem(name,value)
        self.tool.currentIndexChanged.connect(self.arm_tool)
        layout.addWidget(self.tool)
        self.detect_button = QPushButton('ゲートの色候補を抽出（AI未導入）')
        self.detect_button.clicked.connect(self.detect)
        layout.addWidget(self.detect_button)
        self.status = QLabel('候補は未確認です。実際の旗・コブを見て確認してください。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        tabs = QTabWidget()
        self.tabs = tabs
        layout.addWidget(tabs)
        gate_page = QWidget()
        gates_layout = QVBoxLayout(gate_page)
        self.gates = QListWidget()
        self.gates.setMinimumHeight(140)
        self.gates.currentItemChanged.connect(self.gate_selected)
        gates_layout.addWidget(self.gates)
        form = QFormLayout()
        self.gate_side = QComboBox()
        self.gate_side.addItem('左側（Aレーンの外側）','LEFT')
        self.gate_side.addItem('右側（最右レーンの外側）','RIGHT')
        form.addRow('ゲート側',self.gate_side)
        self.gate_number = QSpinBox()
        self.gate_number.setRange(0,9)
        self.gate_number.setSpecialValueText('未設定')
        form.addRow('コース全体の番号',self.gate_number)
        gates_layout.addLayout(form)
        self.apply_gate = QPushButton('選択ゲートの位置・番号を確認')
        self.apply_gate.clicked.connect(self.confirm_gate)
        gates_layout.addWidget(self.apply_gate)
        self.delete_gate = QPushButton('選択ゲートを削除')
        self.delete_gate.clicked.connect(lambda:self.remove('gate'))
        gates_layout.addWidget(self.delete_gate)
        self.confirm_gates = QPushButton('映っている全ゲートの設定を確定')
        self.confirm_gates.clicked.connect(lambda:self.perform(lambda:ControlGateEditor(self.c).confirm(),'ゲート設定を確定しました。'))
        gates_layout.addWidget(self.confirm_gates)
        gates_layout.addWidget(QLabel('全体の1〜9番を維持します。\n左右の旗は同じY座標で対応付けません。\n表示範囲内の登録数・左右の本数は可変です。'))
        tabs.addTab(gate_page,'ゲート')
        bump_page = QWidget()
        bumps_layout = QVBoxLayout(bump_page)
        self.bumps = QListWidget()
        self.bumps.setMinimumHeight(140)
        self.bumps.currentItemChanged.connect(self.bump_selected)
        bumps_layout.addWidget(self.bumps)
        form = QFormLayout()
        self.bump_lane = QComboBox()
        form.addRow('所属レーン',self.bump_lane)
        self.bump_gate = QSpinBox()
        self.bump_gate.setRange(1,9)
        form.addRow('基準ゲート（全体番号）',self.bump_gate)
        self.bump_sequence = QSpinBox()
        self.bump_sequence.setRange(1,9999)
        form.addRow('ゲートからのコブ順序',self.bump_sequence)
        self.bump_side = QComboBox()
        for label,value in [('不明（UNKNOWN）','UNKNOWN'),('スキーヤーの左（L）','L'),('スキーヤーの右（R）','R')]:
            self.bump_side.addItem(label,value)
        form.addRow('スキーヤーから見た左右',self.bump_side)
        bumps_layout.addLayout(form)
        self.apply_bump = QPushButton('選択コブのラベル・位置・左右を確認')
        self.apply_bump.clicked.connect(self.confirm_bump)
        bumps_layout.addWidget(self.apply_bump)
        self.delete_bump = QPushButton('選択コブを削除')
        self.delete_bump.clicked.connect(lambda:self.remove('bump'))
        bumps_layout.addWidget(self.delete_bump)
        self.confirm_bumps = QPushButton('全レーンのコブ設定を確定')
        self.confirm_bumps.clicked.connect(lambda:self.perform(lambda:BumpRegistry(self.c).confirm(),'コブ設定を確定しました。'))
        bumps_layout.addWidget(self.confirm_bumps)
        hint = QLabel('コブは手動登録です。「選択・ドラッグ移動」で中心を移動。\n「選択コブの四隅を補正」で領域を変更できます。\n新規登録の順序は同レーン・同ゲート内で次の番号になります。\nゲート番号が変わると1から開始します。\nL/Rは画面の左右や選手のターン方向とは別です。')
        hint.setWordWrap(True)
        bumps_layout.addWidget(hint)
        tabs.addTab(bump_page,'コブ')
        self.summary = QLabel('')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addStretch()
        self.setWidget(body)
        window.view.landmark_selected.connect(self.select_record)
        window.view.landmark_add_requested.connect(self.add_record)

    @property
    def c(self):
        return self.window.workflow.calibration

    @property
    def busy(self):
        return self.task is not None and self.task.isRunning()

    def arm_tool(self):
        if self.refreshing:
            return
        self.window.view.landmark_tool = self.tool.currentData()
        self.window.view.set_mode('landmarks')
        self.status.setText(self.tool.currentText()+'。映像上で操作してください。')

    def perform(self,action,message='変更しました。'):
        if not layout_ready(self.c):
            self.status.setText('ミドルパートと全レーンを先に確定してください。')
            return False
        before = deepcopy(self.c)
        previous_dirty = self.window.workflow.dirty
        self.window.workflow.checkpoint()
        try:
            action()
            self.status.setText(message)
            self.window.refresh()
            return True
        except (ValueError,StopIteration) as exc:
            self.window.workflow.calibration = before
            self.window.workflow.history.pop()
            self.window.workflow.journal.pop()
            self.window.workflow.dirty = previous_dirty
            self.status.setText(str(exc) or '対象がありません')
            self.window.refresh()
            return False

    def add_record(self,tool,geometry):
        created = []
        if tool == 'add_bump':
            action = lambda:created.append(BumpRegistry(self.c).add(self.bump_lane.currentData(),geometry,self.bump_gate.value()))
            kind = 'bump'
        else:
            action = lambda:created.append(ControlGateEditor(self.c).add('LEFT' if tool == 'add_left' else 'RIGHT',geometry))
            kind = 'gate'
        if self.perform(action,'追加しました。位置・番号を確認して確定してください。'):
            self.select_record(kind,created[0].id)

    def gate_selected(self,item,previous=None):
        if item and not self.refreshing:
            self.select_record('gate',item.data(Qt.UserRole))

    def bump_selected(self,item,previous=None):
        if item and not self.refreshing:
            self.select_record('bump',item.data(Qt.UserRole))

    def select_record(self,kind,identifier):
        self.selected = (kind,identifier)
        if self.window.view.selected_landmark != self.selected or self.window.view.mode != 'landmarks':
            self.window.view.select_landmark(kind,identifier)
        self.tabs.setCurrentIndex(0 if kind == 'gate' else 1)
        self.refresh()
        self.window.course_map.update_course(self.c,self.selected)

    def refresh(self):
        self.refreshing = True
        ready = layout_ready(self.c)
        self.tool.setEnabled(ready and self.window.source_ok)
        self.detect_button.setEnabled(ready and self.window.source_ok and not self.busy)
        selected_lane = self.bump_lane.currentData()
        self.bump_lane.clear()
        for lane in self.c.lanes:
            self.bump_lane.addItem(f'{lane.id}: {lane.name}',lane.id)
        self.bump_lane.setCurrentIndex(max(0,self.bump_lane.findData(selected_lane)))
        self.gates.clear()
        self.bumps.clear()
        current_gate = current_bump = None
        for gate in self.c.control_gates:
            score = '信頼度未校正' if gate.confidence is None else f'{gate.confidence:.2f}'
            self.gates.addItem(f'{"左" if gate.side == "LEFT" else "右"} G{gate.number or "?"} / {gate.origin} / {score} / {"確認済" if gate.confirmed else "未確認"}')
            item = self.gates.item(self.gates.count()-1)
            item.setData(Qt.UserRole,gate.id)
            if self.selected == ('gate',gate.id):
                self.gates.setCurrentItem(item)
                current_gate = gate
        for bump in self.c.bumps:
            self.bumps.addItem(bump.label + (' / 確認済' if bump.confirmed else ' / 未確認'))
            item = self.bumps.item(self.bumps.count()-1)
            item.setData(Qt.UserRole,bump.id)
            if self.selected == ('bump',bump.id):
                self.bumps.setCurrentItem(item)
                current_bump = bump
        if current_gate:
            self.gate_side.setCurrentIndex(self.gate_side.findData(current_gate.side))
            self.gate_number.setValue(current_gate.number or 0)
        if current_bump:
            self.bump_lane.setCurrentIndex(self.bump_lane.findData(current_bump.lane_id))
            self.bump_gate.setValue(current_bump.gate_number)
            self.bump_sequence.setValue(current_bump.sequence)
            self.bump_side.setCurrentIndex(self.bump_side.findData(current_bump.side))
        for button,condition in [(self.apply_gate,current_gate),(self.delete_gate,current_gate),
                                 (self.apply_bump,current_bump),(self.delete_bump,current_bump)]:
            button.setEnabled(ready and bool(condition))
        self.confirm_gates.setEnabled(ready and bool(self.c.control_gates))
        self.confirm_bumps.setEnabled(ready and bool(self.c.bumps))
        self.summary.setText(f'ゲート {len(self.c.control_gates)}本 / {"設定確定 ✓" if self.c.gates_confirmed else "設定未確定"}\nコブ {len(self.c.bumps)}個 / {"設定確定 ✓" if self.c.bumps_confirmed else "設定未確定"}')
        self.refreshing = False

    def confirm_gate(self):
        if self.selected and self.selected[0] == 'gate':
            self.perform(lambda:ControlGateEditor(self.c).update(self.selected[1],self.gate_side.currentData(),self.gate_number.value()),'ゲートを確認しました。全ゲート確認後に設定を確定してください。')

    def confirm_bump(self):
        if self.selected and self.selected[0] == 'bump':
            self.perform(lambda:BumpRegistry(self.c).update(self.selected[1],self.bump_lane.currentData(),self.bump_gate.value(),self.bump_sequence.value(),self.bump_side.currentData()),'コブを確認しました。全コブ確認後に設定を確定してください。')

    def remove(self,kind):
        if self.selected and self.selected[0] == kind:
            identifier = self.selected[1]
            if self.perform(lambda:(ControlGateEditor(self.c) if kind == 'gate' else BumpRegistry(self.c)).remove(identifier),'削除しました（Undo可）。'):
                self.selected = None
                self.window.view.selected_landmark = None
                self.window.refresh()

    def detect(self):
        if self.busy or not layout_ready(self.c) or self.window.current_frame is None:
            return
        if self.window.metadata.get('live'):
            frames = list(self.window.frames)
        else:
            frame = self.window.current_frame.image
            frames = [frame[:,::-1].copy() if self.c.mirrored else frame.copy()]
        self.snapshot = self.c.to_dict()
        self.request_id = self.window.request_id
        self.task = GateDetectionTask(frames,deepcopy(self.c),self)
        self.task.result.connect(self.detected)
        self.task.finished.connect(self.task_finished)
        self.task.start()
        self.status.setText('実画像から色候補を抽出中…（AI未導入）')
        self.refresh()

    def detected(self,result):
        if self.window.closing:
            return
        if isinstance(result,Exception):
            self.status.setText('候補抽出失敗: '+str(result))
            return
        if self.request_id != self.window.request_id or self.c.to_dict() != self.snapshot:
            self.status.setText('抽出中に入力または設定が変わったため候補を適用しません。再実行してください。')
            return
        def apply():
            for candidate in result.candidates:
                if any(g.side == candidate['side'] and np.linalg.norm(np.array(g.point)-candidate['point']) < .025 for g in self.c.control_gates):
                    continue
                attributes = {k:v for k,v in candidate.items() if k not in ('side','point')}
                ControlGateEditor(self.c).add(candidate['side'],candidate['point'],**attributes)
            self.c.candidate['gates'] = {'method':'HSV','message':result.message,'count':len(result.candidates)}
        self.perform(apply,result.message)

    def task_finished(self):
        task = self.task
        self.task = None
        task.deleteLater()
        self.refresh()
