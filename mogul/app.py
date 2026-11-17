"""Japanese calibration workflow. UI delegates capture, geometry and persistence."""
from collections import deque
from pathlib import Path
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget, QTabWidget)

from .detectors import LaneDetector, MiddlePartDetector
from .editor import CalibrationView
from .models import Calibration
from .repository import EventRepository
from .source_selector import VideoSourceSelector
from .video import VideoWorker
from .workflow import AnalysisController, CalibrationWorkflow
from .landmarks import layout_ready
from .landmark_panel import LandmarkPanel
from .course_map import CourseMap
from .orientation import oriented_frame, rotate_calibration
from .analysis_panel import AnalysisPanel
from .analysis_worker import AnalysisWorker
from .analysis_repository import AnalysisRepository
from .landmarks import course_position
from .preview_tracking import PreviewTracking


def timecode(ms):
    total = max(0, int(ms))
    return f'{total//3600000:02}:{total//60000%60:02}:{total//1000%60:02}.{total%1000:03}'


class MainWindow(QMainWindow):
    def __init__(self, database='data/calibration.sqlite', startup=True):
        super().__init__()
        self.setWindowTitle('Mogul Course Analyzer 1.8 — コース設定')
        self.resize(1380, 900)
        self.setMinimumSize(1000, 700)
        self.workflow = CalibrationWorkflow(Calibration())
        self.controller = AnalysisController(self.workflow)
        self.database = Path(database)
        self.session_id = None
        self.source_ok = False
        self.metadata = {}
        self.current_frame = None
        self.preview_tracking = PreviewTracking()
        self.lane_total_values = {}
        self.lane_total_signature = None
        self.frames = deque(maxlen=4)
        self.last_sample = 0
        self.playing = False
        self.recording_active = False
        self.pending_recording = False
        self.auto_detect_pending = False
        self.request_id = 0
        self.loading = False
        self.closing = False
        self.analysis_job = None
        self.analysis_result = None
        self.worker = VideoWorker(self)
        self._build()
        self.pending_preview=None
        self.preview_timer=QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.flush_preview)
        self.worker.frame_ready.connect(self.queue_preview_frame)
        self.worker.opened.connect(self.on_opened)
        self.worker.status.connect(self.on_status)
        self.worker.recording.connect(self.on_recording)
        self.worker.ended.connect(self.on_ended)
        self.worker.playback.connect(self.on_playback)
        self.worker.start()
        self.refresh()
        if startup:
            QTimer.singleShot(0, self.initial_source)

    def _button(self, text, slot, layout):
        button = QPushButton(text)
        button.clicked.connect(slot)
        layout.addWidget(button)
        return button

    def _build(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel('MOGUL  /  COURSE ANALYZER')
        title.setStyleSheet('font-size: 20px; font-weight: 700; color: #8ed7ed;')
        header.addWidget(title)
        header.addStretch()
        header.addWidget(QLabel('v1.8  •  Phase 3'))
        root.addLayout(header)
        actions = QHBoxLayout()
        self.source_button = self._button('映像ソースを変更…', self.select_source, actions)
        self._button('設定を保存  Ctrl+S', self.save, actions)
        self._button('設定を復元…', self.restore_dialog, actions)
        self._button('元に戻す  Ctrl+Z', self.undo, actions)
        actions.addStretch()
        self.dirty_label = QLabel('')
        actions.addWidget(self.dirty_label)
        root.addLayout(actions)
        self.input_label = QLabel('映像未選択 — 録画済み動画またはUVCカメラを選択してください')
        root.addWidget(self.input_label)
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        video_layout = QVBoxLayout(left)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.view = CalibrationView(self.workflow)
        self.view.changed.connect(self.calibration_changed)
        self.view.notice.connect(lambda text: self.notice.setText(text))
        display_controls = QHBoxLayout()
        display_controls.addWidget(QLabel('表示回転'))
        self.rotation = QComboBox()
        for label,value in [('なし',0),('右90°',90),('180°',180),('左90°',270)]:
            self.rotation.addItem(label,value)
        self.rotation.currentIndexChanged.connect(self.change_rotation)
        display_controls.addWidget(self.rotation)
        self._button('－',lambda:self.view.set_zoom(self.view.zoom_factor/1.25),display_controls)
        self._button('＋',lambda:self.view.set_zoom(self.view.zoom_factor*1.25),display_controls)
        self._button('全体表示',self.view.fit_all,display_controls)
        self._button('ミドルを拡大',self.view.focus_middle,display_controls)
        self.zoom_label = QLabel('全体比 100%')
        self.view.zoom_changed.connect(lambda factor:self.zoom_label.setText(f'全体比 {factor*100:.0f}%'))
        display_controls.addWidget(self.zoom_label)
        self.geometry_labels=QCheckBox('分析中も設定マーカーを表示')
        self.geometry_labels.toggled.connect(self.change_analysis_geometry)
        display_controls.addWidget(self.geometry_labels)
        display_controls.addStretch()
        video_layout.addLayout(display_controls)
        video_layout.addWidget(self.view, 1)
        playback = QHBoxLayout()
        self.play_button = self._button('▶ 再生', self.toggle_play, playback)
        self.back_button = self._button('◀ 1フレーム', lambda: self.step(-1), playback)
        self.step_button = self._button('1フレーム ▶', lambda: self.step(1), playback)
        self.speed = QComboBox()
        for value in (.25, .5, 1, 1.5, 2):
            self.speed.addItem(f'{value:g} ×', value)
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(lambda: self.worker.command('speed', self.speed.currentData()))
        playback.addWidget(self.speed)
        self.clock = QLabel('00:00:00.000 / 00:00:00.000')
        playback.addWidget(self.clock)
        video_layout.addLayout(playback)
        self.seek = QSlider(Qt.Horizontal)
        self.seek.setRange(0, 0)
        self.seek.sliderReleased.connect(lambda: self.worker.command('seek', self.seek.value()))
        video_layout.addWidget(self.seek)
        video_layout.addWidget(QLabel('上端から入り、同じレーンの下端へ抜けた人数（自動推定・要確認）'))
        self.lane_totals=QHBoxLayout()
        video_layout.addLayout(self.lane_totals)
        recording = QHBoxLayout()
        self.record_button = self._button('● MP4録画', self.toggle_record, recording)
        self.record_label = QLabel('UVC入力で録画できます')
        recording.addWidget(self.record_label, 1)
        video_layout.addLayout(recording)
        self.notice = QLabel('人数分析はミドルパートと全レーンの確定だけで開始できます。ゲート・コブ登録は不要です。')
        self.notice.setWordWrap(True)
        self.notice.setMinimumHeight(46)
        video_layout.addWidget(self.notice)
        splitter.addWidget(left)
        panel = QWidget()
        side = QVBoxLayout(panel)
        middle_box = QGroupBox('01  ミドルパート')
        middle_layout = QVBoxLayout(middle_box)
        self.middle_state = QLabel('未設定')
        middle_layout.addWidget(self.middle_state)
        self.ai_middle_button = self._button('AI候補を検出 / 現在の代表フレーム', self.detect_middle, middle_layout)
        self.middle_edit = self._button('ミドルパートを表示・編集', lambda: self.set_mode('middle'), middle_layout)
        self._button('ミドルパートをリセット', self.reset_middle, middle_layout)
        explanation = QLabel('初回は映像上をドラッグして四角形を作成。\n四隅を補正し、マーカー以外をクリックして確定。\n確定領域内をダブルクリックすると再編集。')
        explanation.setWordWrap(True)
        middle_layout.addWidget(explanation)
        side.addWidget(middle_box)
        lanes_box = QGroupBox('02  レーン')
        lanes_layout = QVBoxLayout(lanes_box)
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel('レーン数'))
        self.lane_count = QSpinBox()
        self.lane_count.setRange(2, 4)
        count_row.addWidget(self.lane_count)
        lanes_layout.addLayout(count_row)
        self.ai_lane_button = self._button('AIレーン候補を検出', self.detect_lanes, lanes_layout)
        self.split_button = self._button('手動均等分割で初期配置', self.configure_lanes, lanes_layout)
        self.lane_select = QComboBox()
        self.lane_select.currentIndexChanged.connect(self.select_lane)
        lanes_layout.addWidget(self.lane_select)
        self.lane_name = QLineEdit()
        self.lane_name.setPlaceholderText('レーン名（編集後 Enter）')
        self.lane_name.editingFinished.connect(self.rename_lane)
        lanes_layout.addWidget(self.lane_name)
        self._button('選択レーンの色…', self.choose_lane_color, lanes_layout)
        lanes_hint = QLabel('各レーンを選択して四隅を補正・確定します。\n外側の四隅も移動可能。ゲート側との余白を設定できます。\n内部境界は隣接レーンと共有します。\n均等分割は手動初期配置であり、AI検出ではありません。')
        lanes_hint.setWordWrap(True)
        lanes_layout.addWidget(lanes_hint)
        self.lanes_state = QLabel('未設定')
        self.lanes_state.setWordWrap(True)
        lanes_layout.addWidget(self.lanes_state)
        side.addWidget(lanes_box)
        settings_box = QGroupBox('表示・セッション設定')
        form = QFormLayout(settings_box)
        self.opacity = QSpinBox()
        self.opacity.setRange(5, 100)
        self.opacity.setSuffix(' %')
        self.opacity.setValue(30)
        self.opacity.valueChanged.connect(self.change_opacity)
        form.addRow('領域の不透明度', self.opacity)
        color_button = QPushButton('色を変更…')
        color_button.clicked.connect(self.choose_middle_color)
        form.addRow('ミドルパート', color_button)
        self.mirror = QCheckBox('左右反転表示')
        self.mirror.toggled.connect(self.change_mirror)
        form.addRow(self.mirror)
        self.start_time = QDoubleSpinBox()
        self.start_time.setRange(0, 8640000)
        self.start_time.setDecimals(3)
        self.start_time.setSuffix(' 秒')
        self.start_time.valueChanged.connect(self.change_start)
        form.addRow('解析開始時刻', self.start_time)
        current_button = QPushButton('現在時刻を解析開始位置にする')
        current_button.clicked.connect(lambda: self.start_time.setValue(self.current_frame.timestamp_ms/1000) if self.current_frame else None)
        form.addRow(current_button)
        side.addWidget(settings_box)
        side.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(325)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(scroll,'領域設定')
        self.course_map = CourseMap(self)
        self.landmarks_panel = LandmarkPanel(self)
        self.right_tabs.addTab(self.landmarks_panel,'ゲート・コブ')
        self.right_tabs.addTab(self.course_map,'コースマップ')
        self.analysis_panel = AnalysisPanel(self)
        self.right_tabs.addTab(self.analysis_panel,'分析・結果')
        self.course_map.selected.connect(self.landmarks_panel.select_record)
        self.right_tabs.currentChanged.connect(self.right_tab_changed)
        splitter.addWidget(self.right_tabs)
        splitter.setSizes([1000, 345])
        root.addWidget(splitter, 1)
        footer = QHBoxLayout()
        self.prerequisites = QLabel('')
        self.prerequisites.setWordWrap(True)
        footer.addWidget(self.prerequisites, 1)
        self.analyze_button = QPushButton('分析開始')
        self.analyze_button.setToolTip('ミドルパートとレーンを確定すると人数分析できます。ゲート・コブ登録は不要です。')
        self.analyze_button.clicked.connect(self.start_analysis)
        footer.addWidget(self.analyze_button)
        root.addLayout(footer)
        self.setCentralWidget(central)
        for shortcut, slot in [(QKeySequence.Save, self.save), (QKeySequence.Undo, self.undo)]:
            action = QAction(self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            self.addAction(action)
        self.setStyleSheet('''
            QMainWindow, QDialog { background: #18212d; color: #e6edf4; }
            QWidget { font-family: "Yu Gothic UI"; font-size: 12px; color: #e6edf4; }
            QGroupBox { border: 1px solid #3c4c60; border-radius: 7px; margin-top: 12px; padding: 12px 8px 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QPushButton { background: #293b50; border: 1px solid #49617a; border-radius: 5px; padding: 8px; }
            QPushButton:hover { background: #3b5670; }
            QPushButton:disabled { color: #7b8795; background: #252d37; border-color: #35404e; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #222f40; border: 1px solid #4c6077; padding: 5px; }
            QScrollArea { border: none; background: #18212d; }
            QScrollArea > QWidget > QWidget { background: #18212d; }
            QTabWidget::pane { background: #18212d; border: 1px solid #3c4c60; }
            QTabBar::tab { background: #293b50; color: #e6edf4; padding: 8px; }
            QTabBar::tab:selected { background: #3b5670; }
            QListWidget { background: #222f40; color: #e6edf4; border: 1px solid #4c6077; }
            QListWidget::item:selected { background: #3b5670; }
            QMessageBox { background: #263446; }
        ''')

    def initial_source(self):
        self.analysis_panel.refresh_history()
        if self.database.exists() and QMessageBox.question(self, '前回の設定', '前回保存したキャリブレーションを復元しますか？') == QMessageBox.Yes:
            self.restore(self.database)
        else:
            self.select_source()

    def maybe_save(self):
        self.view.click_timer.stop()
        if not self.workflow.dirty:
            return True
        result = QMessageBox.question(self, '未保存の変更', '設定の変更を保存しますか？', QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if result == QMessageBox.Cancel:
            return False
        return self.save() if result == QMessageBox.Save else True

    def select_source(self):
        if self.analysis_active:
            self.notice.setText('分析を停止してから映像ソースを変更してください。')
            return
        dialog = VideoSourceSelector(self)
        if dialog.exec() and self.maybe_save():
            self.open_source(dialog.spec, reset=True)

    def open_source(self, spec, reset=False):
        if self.analysis_active:
            self.notice.setText('分析を停止してから映像を開いてください。')
            return
        self.analysis_panel.review_analysis_id = None
        self.pending_preview=None
        self.preview_timer.stop()
        self.analysis_panel.review_end_ms = None
        self.view.analysis_tracks = []
        self.view.analysis_events = []
        self.course_map.analysis_events = []
        self.preview_tracking = PreviewTracking()
        self.request_id += 1
        self.view.click_timer.stop()
        self.source_ok = False
        self.metadata = {}
        self.frames.clear()
        self.last_sample = 0
        self.current_frame = None
        self.playing = False
        self.play_button.setText('▶ 再生')
        self.recording_active = self.pending_recording = False
        self.record_button.setText('● MP4録画')
        self.view.clear_frame()
        if reset:
            self.analysis_panel.reset_result()
            self.workflow.calibration = Calibration(source=spec,rotation=spec.get('rotation',0))
            self.workflow.history.clear()
            self.workflow.journal.clear()
            self.session_id = None
            self.workflow.dirty = True
            self.view.set_mode('middle')
            self.landmarks_panel.selected = None
            self.view.selected_landmark = None
            self.right_tabs.setCurrentIndex(0)
            self.sync_settings()
        self.auto_detect_pending = not bool(self.workflow.calibration.middle)
        self.input_label.setText('映像ソースに接続中…')
        self.worker.command('open', dict(spec, _request_id=self.request_id))
        self.refresh()

    def on_opened(self, metadata):
        if metadata.get('_request_id') != self.request_id:
            return
        self.metadata = metadata
        c = self.workflow.calibration
        recordings = c.source.get('recordings', [])
        c.source = {k:v for k,v in metadata.items() if not k.startswith('_')}
        if recordings:
            c.source['recordings'] = recordings
        self.seek.setRange(0, int(metadata['duration_ms']))
        self.loading = True
        self.start_time.setMaximum(max(0, metadata['duration_ms']/1000) if not metadata['live'] else 8640000)
        self.start_time.setValue(c.start_ms/1000)
        self.loading = False
        self.input_label.setText(f"{'UVCライブ' if metadata['live'] else Path(metadata['path']).name}  |  {metadata['width']} × {metadata['height']}  |  {metadata['fps']:g} fps")
        if not metadata['live'] and c.start_ms:
            self.worker.command('seek', c.start_ms)
        self.refresh()

    def queue_preview_frame(self,frame):
        if frame.source_id!=self.request_id or self.closing:
            return
        if self.analysis_active and self.analysis_job.queued:
            if not self.metadata.get('live') and self.analysis_waiting_seek:
                if abs(frame.timestamp_ms-self.workflow.calibration.start_ms)>1000/max(1,self.metadata.get('fps',30)):
                    return
                self.analysis_waiting_seek=False
            self.analysis_job.submit(frame)
        self.pending_preview=frame
        if self.current_frame is None or not self.playing:
            self.flush_preview()
            return
        if not self.preview_timer.isActive():
            self.preview_timer.start(16)

    def flush_preview(self):
        frame=self.pending_preview
        self.pending_preview=None
        if frame is not None:
            self.on_frame(frame)

    def on_frame(self, frame):
        if self.analysis_active:
            if self.analysis_job.queued and frame.source_id == self.request_id:
                self.view.analysis_tracks = self.preview_tracking.frame(
                    self.display_frame(frame.image),frame.timestamp_ms,frame.index)
                self.present_frame(frame)
                self.update_analysis_events(self.analysis_panel.data.get('runs',[]) if self.analysis_panel.data else [],frame.timestamp_ms)
                self.view.redraw_analysis()
                if self.right_tabs.currentIndex()==2:
                    self.course_map.update_course(self.workflow.calibration,self.landmarks_panel.selected,self.view.analysis_tracks)
            return
        had_tracks = bool(self.view.analysis_tracks)
        self.view.analysis_tracks = []
        self.present_frame(frame)
        if had_tracks:
            self.view.redraw()
            self.course_map.update_course(self.workflow.calibration,self.landmarks_panel.selected)
        panel = self.analysis_panel
        if panel.review_analysis_id and frame.source_id == self.request_id:
            self.view.analysis_tracks = AnalysisRepository(self.database).tracks_at(panel.review_analysis_id,frame.index)
            self.update_analysis_events(panel.data.get('runs',[]) if panel.data else [],frame.timestamp_ms)
            self.view.redraw()
            self.course_map.update_course(self.workflow.calibration,self.landmarks_panel.selected,self.view.analysis_tracks)
        if panel.review_end_ms is not None and frame.timestamp_ms >= panel.review_end_ms:
            self.worker.command('play',False)
            panel.review_end_ms = None

    def present_frame(self, frame):
        if frame.source_id != self.request_id:
            return
        was_healthy = self.source_ok
        self.current_frame = frame
        self.source_ok = True
        image = self.display_frame(frame.image)
        self.view.set_frame(image)
        self.update_input_label()
        if time.monotonic()-self.last_sample > .5:
            self.frames.append(image.copy())
            self.last_sample = time.monotonic()
        duration = self.metadata.get('duration_ms', 0)
        self.clock.setText(f'{timecode(frame.timestamp_ms)} / {"LIVE" if self.metadata.get("live") else timecode(duration)}')
        if not self.seek.isSliderDown():
            self.seek.setValue(int(frame.timestamp_ms))
        if not was_healthy:
            self.refresh()
        if self.auto_detect_pending:
            self.auto_detect_pending = False
            QTimer.singleShot(0, self.detect_middle)

    def on_status(self, text, healthy, request_id):
        if request_id != self.request_id:
            return
        changed = self.source_ok != healthy
        self.source_ok = healthy and self.current_frame is not None
        if not healthy:
            if self.analysis_active and self.analysis_job.queued and not self.metadata.get('live'):
                self.stop_analysis()
            self.pending_recording = False
            self.notice.setText(text + (' ／ 3秒後に再接続を試みます' if self.metadata.get('live') else ''))
            self.input_label.setText('入力異常: ' + text[:200])
        elif changed:
            self.notice.setText('映像入力が正常です。領域設定を操作できます。')
        if changed:
            self.refresh()

    def on_playback(self, playing, request_id):
        if request_id != self.request_id:
            return
        self.playing = playing
        self.play_button.setText('Ⅱ 一時停止' if playing else '▶ 再生')

    def on_ended(self, request_id):
        if request_id == self.request_id:
            self.flush_preview()
            if self.analysis_active and self.analysis_job.queued and not self.metadata.get('live') and not self.analysis_waiting_seek:
                self.analysis_job.end_input()
            self.notice.setText('動画の終端です。シークまたは先頭に戻して再生できます。')

    def toggle_play(self):
        self.worker.command('play', not self.playing)

    def step(self, direction):
        self.worker.command('play', False)
        self.worker.command('step', direction)

    def toggle_record(self):
        if self.recording_active:
            self.worker.command('stop_record')
        elif not self.pending_recording:
            path, _ = QFileDialog.getSaveFileName(self, 'MP4録画先', 'recording.mp4', 'MP4 (*.mp4)')
            if path:
                if not path.lower().endswith('.mp4'):
                    path += '.mp4'
                self.pending_recording = True
                self.worker.command('record', path)

    def on_recording(self, text, request_id):
        if request_id != self.request_id:
            return
        self.pending_recording = False
        self.recording_active = text.startswith('録画中:')
        self.record_label.setText(text)
        self.record_button.setText('■ 録画停止' if self.recording_active else '● MP4録画')
        if text.startswith('録画中:'):
            self.workflow.calibration.source.setdefault('recordings', []).append(text.split(': ', 1)[1])
            self.workflow.dirty = True
        self.refresh()

    def set_mode(self, mode):
        self.view.set_mode(mode, max(0, self.lane_select.currentIndex()))
        self.refresh()

    def select_lane(self, index):
        if index >= 0 and index < len(self.workflow.calibration.lanes):
            self.view.set_mode('lane', index)
            self.lane_name.setText(self.workflow.calibration.lanes[index].name)
        self.refresh_state()

    def refresh_state(self):
        c = self.workflow.calibration
        self.middle_state.setText('確定済み ✓' if c.middle_confirmed else ('四隅を編集中' if c.middle else '未設定：映像上をドラッグ'))
        self.lanes_state.setText(' / '.join(f'{x.id} {"確定 ✓" if x.confirmed else "未確定"}' for x in c.lanes) or 'ミドルパートを確定してから設定')
        self.dirty_label.setText('● 未保存' if self.workflow.dirty else '保存済み / 変更なし')
        reasons = self.workflow.reasons(self.source_ok)
        try:
            if c.middle_confirmed:
                course_position(c,c.middle[0])
        except ValueError as exc:
            reasons.append(str(exc))
        if not Path(self.analysis_panel.model.text()).is_file():
            reasons.append('人物モデル未導入です（分析・結果タブを確認）')
        self.analyze_button.setEnabled(not reasons and not self.analysis_active and not self.landmarks_panel.busy)
        self.analyze_button.setText('分析中…' if self.analysis_active else '分析開始')
        self.prerequisites.setText('分析中です。右側の「分析停止」で停止・保存できます。' if self.analysis_active else ('分析開始できない理由：' + ' ／ '.join(reasons) if reasons else '分析開始条件を満たしています'))

    def refresh(self):
        c = self.workflow.calibration
        self.update_lane_totals()
        index = min(self.view.lane_index, max(0, len(c.lanes)-1))
        self.lane_select.blockSignals(True)
        self.lane_select.clear()
        for lane in c.lanes:
            self.lane_select.addItem(f'{lane.id} — {lane.name}' + (' ✓' if lane.confirmed else ''))
        if c.lanes:
            self.lane_select.setCurrentIndex(index)
            self.lane_name.setText(c.lanes[index].name)
        else:
            self.lane_name.clear()
        self.lane_select.blockSignals(False)
        for button in (self.ai_lane_button, self.split_button, self.lane_count):
            button.setEnabled(c.middle_confirmed and self.source_ok)
        self.ai_middle_button.setEnabled(self.source_ok)
        live = self.metadata.get('live', False)
        for control in (self.play_button, self.back_button, self.step_button, self.speed, self.seek):
            control.setEnabled(self.source_ok and not live)
        self.record_button.setEnabled((live and self.source_ok) or self.recording_active)
        self.lane_select.setEnabled(bool(c.lanes))
        self.lane_name.setEnabled(bool(c.lanes))
        self.start_time.setEnabled(not live)
        self.view.redraw()
        self.landmarks_panel.refresh()
        self.course_map.update_course(c,self.landmarks_panel.selected,self.view.analysis_tracks)
        self.refresh_state()
        self.source_button.setEnabled(not self.analysis_active)
        self.rotation.setEnabled(not self.analysis_active)
        self.mirror.setEnabled(not self.analysis_active)
        self.start_time.setEnabled(not self.analysis_active and not live)
        for index in (0,1):
            self.right_tabs.setTabEnabled(index,not self.analysis_active)
        if self.analysis_active:
            for control in (self.play_button,self.back_button,self.step_button,self.speed,self.seek):
                control.setEnabled(False)

    def right_tab_changed(self,index):
        if index==2:
            self.course_map.update_course(self.workflow.calibration,self.landmarks_panel.selected,self.view.analysis_tracks)
        if self.analysis_active:
            return
        if index == 1:
            self.landmarks_panel.arm_tool()
        elif index == 0 and self.view.mode in ('landmarks','analysis'):
            self.view.set_mode('lane' if self.workflow.calibration.lanes else 'middle',max(0,self.lane_select.currentIndex()))

    def detect_middle(self):
        if not self.source_ok or self.current_frame is None:
            return
        image = self.display_frame(self.current_frame.image)
        frames = list(self.frames) if self.metadata.get('live') else [image]
        result = MiddlePartDetector().detect(frames)
        self.workflow.checkpoint()
        self.workflow.calibration.candidate['middle'] = {'success': result.success, 'message': result.message, 'model': result.model,
                                                         'timestamp_ms': self.current_frame.timestamp_ms if self.current_frame else None}
        self.notice.setText(result.message)
        self.view.set_mode('middle')
        self.refresh()

    def detect_lanes(self):
        c = self.workflow.calibration
        result = LaneDetector().detect(list(self.frames), c.middle, self.lane_count.value())
        self.workflow.checkpoint()
        c.candidate['lanes'] = {'success': result.success, 'message': result.message, 'model': result.model}
        self.notice.setText(result.message)
        self.refresh()

    def configure_lanes(self):
        c = self.workflow.calibration
        if c.lanes and QMessageBox.question(self, 'レーン再設定', '現在のレーン補正を均等分割の初期配置に戻しますか？（Undo可）') != QMessageBox.Yes:
            return
        self.workflow.checkpoint()
        try:
            c.configure_lanes(self.lane_count.value())
            self.view.set_mode('lane', 0)
            self.notice.setText('手動の初期配置です。各レーンを選択し、境界を補正してクリックで確定してください。')
        except ValueError as exc:
            self.notice.setText(str(exc))
        self.refresh()

    def calibration_changed(self):
        c = self.workflow.calibration
        if c.middle_confirmed and 'lanes' not in c.candidate:
            self.detect_lanes()
        self.refresh()
        if self.view.mode=='lane' and layout_ready(c):
            self.right_tabs.setCurrentIndex(3)
            self.notice.setText('人数分析の準備ができました。「分析開始」へ進めます。ゲート・コブ登録は不要です。')

    def rename_lane(self):
        i = self.lane_select.currentIndex()
        lanes = self.workflow.calibration.lanes
        if 0 <= i < len(lanes) and self.lane_name.text().strip() and lanes[i].name != self.lane_name.text().strip():
            self.workflow.checkpoint()
            lanes[i].name = self.lane_name.text().strip()
            self.refresh()

    def choose_lane_color(self):
        i = self.lane_select.currentIndex()
        lanes = self.workflow.calibration.lanes
        if 0 <= i < len(lanes):
            color = QColorDialog.getColor(QColor(lanes[i].color), self, 'レーン表示色')
            if color.isValid():
                self.workflow.checkpoint()
                lanes[i].color = color.name()
                self.refresh()

    def choose_middle_color(self):
        color = QColorDialog.getColor(QColor(self.workflow.calibration.middle_color), self, 'ミドルパート表示色')
        if color.isValid():
            self.workflow.checkpoint()
            self.workflow.calibration.middle_color = color.name()
            self.refresh()

    def change_opacity(self, value):
        if not self.loading:
            self.workflow.checkpoint()
            self.workflow.calibration.opacity = value/100
            self.refresh()

    def change_start(self, value):
        if not self.loading:
            self.workflow.checkpoint()
            self.workflow.calibration.start_ms = value*1000
            self.refresh_state()

    def display_frame(self,image):
        c = self.workflow.calibration
        return oriented_frame(image,c.rotation,c.mirrored)

    def update_input_label(self):
        if self.metadata:
            m = self.metadata
            name = 'UVCライブ' if m.get('live') else Path(m.get('path','')).name
            self.input_label.setText(f'{name}  |  入力 {m["width"]} × {m["height"]}  →  表示 {self.view.width_px} × {self.view.height_px}  |  {m["fps"]:g} fps')

    def change_rotation(self):
        if self.loading or self.analysis_active:
            return
        rotation = self.rotation.currentData()
        if rotation == self.workflow.calibration.rotation:
            return
        self.view.click_timer.stop()
        self.view.drag = None
        self.analysis_panel.review_analysis_id = None
        self.view.analysis_tracks = []
        self.workflow.checkpoint()
        rotate_calibration(self.workflow.calibration,rotation)
        self.frames.clear()
        self.last_sample = 0
        if self.current_frame:
            self.on_frame(self.current_frame)
        self.view.fit_all()
        self.notice.setText('回転後の映像を設定・候補検出に使用します。既存の領域・ゲート・コブの座標も一緒に回転しました。')
        self.refresh()

    def change_mirror(self, checked):
        if not self.loading and not self.analysis_active:
            self.analysis_panel.review_analysis_id = None
            self.view.analysis_tracks = []
            self.workflow.checkpoint()
            c = self.workflow.calibration
            c.mirrored = checked
            self.frames.clear()
            self.last_sample = 0
            # Geometry stays in displayed coordinates; changing mirror invalidates positional setup.
            if c.middle:
                c.middle = [[1-c.middle[i][0], c.middle[i][1]] for i in (1, 0, 3, 2)]
                c.middle_confirmed = False
                c.clear_lanes()
            for gate in c.control_gates:
                gate.point = [1-gate.point[0],gate.point[1]]
                if gate.candidate_point is not None:
                    gate.candidate_point = [1-gate.candidate_point[0],gate.candidate_point[1]]
                gate.side = 'RIGHT' if gate.side == 'LEFT' else 'LEFT'
            for bump in c.bumps:
                bump.center = [1-bump.center[0],bump.center[1]]
                bump.polygon = [[1-p[0],p[1]] for p in reversed(bump.polygon)]
            c.invalidate_landmarks()
            if self.current_frame:
                self.on_frame(self.current_frame)
            self.notice.setText('反転を変更しました。領域・レーンの位置を再確認してください（Undo可）。')
            self.refresh()

    def reset_middle(self):
        if self.workflow.calibration.middle and QMessageBox.question(self, '領域リセット', 'ミドルパートと全レーンをリセットしますか？（Undo可）') != QMessageBox.Yes:
            return
        self.workflow.checkpoint()
        self.workflow.calibration.middle = []
        self.workflow.calibration.middle_confirmed = False
        self.workflow.calibration.clear_lanes()
        self.view.set_mode('middle')
        self.refresh()

    def sync_settings(self):
        self.loading = True
        c = self.workflow.calibration
        self.opacity.setValue(round(c.opacity*100))
        self.mirror.setChecked(c.mirrored)
        self.rotation.setCurrentIndex(self.rotation.findData(c.rotation))
        self.start_time.setValue(c.start_ms/1000)
        self.lane_count.setValue(len(c.lanes) or 2)
        self.analysis_panel.apply_config(c.analysis_config)
        self.loading = False

    def undo(self):
        if self.analysis_active:
            self.notice.setText('分析を停止してから設定を戻してください。')
            return
        self.view.click_timer.stop()
        self.view.drag = None
        previous_orientation = (self.workflow.calibration.rotation,self.workflow.calibration.mirrored)
        if self.workflow.undo():
            if previous_orientation != (self.workflow.calibration.rotation,self.workflow.calibration.mirrored):
                self.analysis_panel.review_analysis_id = None
                self.view.analysis_tracks = []
                self.frames.clear()
                self.last_sample = 0
            if not self.workflow.calibration.lanes:
                self.view.set_mode('middle')
            self.sync_settings()
            if self.current_frame:
                self.on_frame(self.current_frame)
            self.refresh()

    def save(self):
        self.view.click_timer.stop()
        try:
            self.database.parent.mkdir(parents=True, exist_ok=True)
            repository = EventRepository(self.database)
            self.session_id = repository.save(self.workflow.calibration, self.session_id, self.workflow.journal)
            self.workflow.journal.clear()
            self.workflow.dirty = False
            self.notice.setText('キャリブレーションと編集履歴を保存しました: ' + str(self.database.resolve()))
            self.refresh_state()
            return True
        except Exception as exc:
            QMessageBox.warning(self, '保存できません', str(exc))
            return False

    def restore_dialog(self):
        if self.analysis_active:
            self.notice.setText('分析を停止してから復元してください。')
            return
        path, _ = QFileDialog.getOpenFileName(self, 'キャリブレーションを復元', str(self.database.parent), 'SQLite (*.sqlite *.db);;すべてのファイル (*)')
        if path and self.maybe_save():
            self.restore(path)

    def restore(self, path):
        if self.analysis_active:
            self.notice.setText('分析を停止してから復元してください。')
            return False
        try:
            session_id, calibration = EventRepository(path).load()
            self.database = Path(path)
            self.session_id = session_id
            self.workflow.calibration = calibration
            self.workflow.history.clear()
            self.workflow.journal.clear()
            self.workflow.dirty = False
            self.analysis_panel.reset_result()
            self.analysis_panel.refresh_history()
            self.landmarks_panel.selected = None
            self.view.selected_landmark = None
            ready=layout_ready(calibration)
            self.view.set_mode('analysis' if ready else ('lane' if calibration.lanes else 'middle'))
            self.right_tabs.setCurrentIndex(3 if ready else 0)
            self.sync_settings()
            self.notice.setText('保存済みの設定を復元しました。')
            if calibration.source:
                self.open_source(calibration.source)
            self.refresh()
            return True
        except Exception as exc:
            QMessageBox.warning(self, '復元できません', str(exc))
            return False

    def start_analysis(self):
        if self.analysis_active or self.landmarks_panel.busy:
            return
        if not self.controller.can_start(self.source_ok):
            self.notice.setText(' ／ '.join(self.workflow.reasons(self.source_ok)))
            return
        if not Path(self.analysis_panel.model.text()).is_file():
            self.right_tabs.setCurrentIndex(3)
            self.notice.setText('人物モデルを導入してください。')
            return
        self.workflow.checkpoint()
        self.workflow.calibration.analysis_config = self.analysis_panel.config()
        if not self.save():
            return
        self.worker.command('play',False if not self.metadata.get('live') else True)
        self.view.click_timer.stop()
        self.view.analysis_tracks = []
        self.view.set_mode('analysis')
        self.view.analysis_events = []
        self.preview_tracking = PreviewTracking()
        self.lane_total_values = {}
        self.update_lane_totals()
        config = self.analysis_panel.config()
        if self.metadata.get('live'):
            config['live_start_after_ms'] = ((time.monotonic()-self.metadata['_live_epoch'])*1000
                                            if '_live_epoch' in self.metadata else self.current_frame.timestamp_ms)
        self.analysis_result = None
        self.analysis_waiting_seek = not self.metadata.get('live')
        self.analysis_panel.reset_result()
        self.analysis_panel.set_active(True)
        self.analysis_job = AnalysisWorker(self.workflow.calibration,self.database,self.session_id,config,self.request_id,self)
        self.analysis_job.phase.connect(self.analysis_panel.progress.setText)
        self.analysis_job.packet.connect(self.on_analysis_packet)
        self.analysis_job.completed.connect(self.on_analysis_completed)
        self.analysis_job.finished.connect(self.finish_analysis)
        self.analysis_job.start()
        if self.analysis_job.queued and not self.metadata.get('live'):
            self.speed.setCurrentIndex(2)
            self.worker.command('speed',1.0)
            self.worker.command('seek',self.workflow.calibration.start_ms)
            self.worker.command('play',True)
        self.right_tabs.setCurrentIndex(3)
        self.refresh()

    @property
    def analysis_active(self):
        # Remain locked until the GUI has consumed completion and cleanup signals.
        return self.analysis_job is not None

    def stop_analysis(self):
        if self.analysis_active:
            self.analysis_job.requestInterruption()
            if not self.metadata.get('live'):
                self.worker.command('play',False)
            self.analysis_panel.progress.setText('停止・保存中…')

    def on_analysis_packet(self,packet):
        if self.closing or packet['frame'].source_id != self.request_id:
            return
        queued=self.analysis_active and self.analysis_job.queued
        if queued:
            self.flush_preview()
        if not queued:
            self.present_frame(packet['frame'])
        overlays=packet.get('overlays',packet['tracks'])
        self.view.analysis_tracks = (self.preview_tracking.correct(overlays,packet['frame'].timestamp_ms,packet['frame'].index,self.display_frame(packet['frame'].image))
                                     if queued else overlays)
        self.update_analysis_events(packet['runs'],self.current_frame.timestamp_ms if self.current_frame else 0)
        self.view.redraw_analysis()
        if self.right_tabs.currentIndex()==2:
            self.course_map.update_course(self.workflow.calibration,self.landmarks_panel.selected,self.view.analysis_tracks)
        self.analysis_panel.show_packet(packet)

    def update_lane_totals(self,values=None,lanes=None):
        if values is not None:
            self.lane_total_values=values
        lanes=self.workflow.calibration.lanes if lanes is None else lanes
        signature=tuple((l.id,l.name,l.color) for l in lanes)
        if signature!=self.lane_total_signature:
            while self.lane_totals.count():
                item=self.lane_totals.takeAt(0)
                item.widget().deleteLater()
            self.lane_total_labels={}
            for lane in lanes:
                label=QLabel(); label.setAlignment(Qt.AlignCenter)
                label.setWordWrap(True)
                label.setStyleSheet(f'font-size:28px;font-weight:700;border:2px solid {lane.color};border-radius:6px;padding:8px;')
                label.setMinimumHeight(78)
                self.lane_totals.addWidget(label)
                self.lane_total_labels[lane.id]=label
            self.lane_total_signature=signature
        for lane in lanes:
            self.lane_total_labels[lane.id].setText(f'{lane.id} {lane.name}\n{self.lane_total_values.get(lane.id,0)}')

    def change_analysis_geometry(self,checked):
        self.view.show_analysis_geometry=checked
        self.view.redraw()

    def update_analysis_events(self,runs,timestamp):
        self.view.analysis_events=[e for run in runs if run.get('review')!='excluded'
            for e in run.get('events',[]) if e.get('review')!='excluded' and e['timestamp_ms']<=timestamp]
        self.course_map.analysis_events=self.view.analysis_events

    def on_analysis_completed(self,result):
        self.analysis_result = result
        if result['id']:
            try:
                AnalysisRepository(self.database).attach_recordings(result['id'],self.workflow.calibration.source.get('recordings',[]))
            except Exception as exc:
                self.notice.setText('録画関連付けエラー: '+str(exc))

    def finish_analysis(self):
        job = self.analysis_job
        if job.queued and not self.metadata.get('live'):
            self.worker.command('play',False)
        self.analysis_job = None
        job.deleteLater()
        self.analysis_panel.set_active(False)
        result = self.analysis_result or {'id':None,'status':'failed','error':'解析スレッドが予期せず終了しました'}
        self.analysis_panel.refresh_history()
        if result['id']:
            self.analysis_panel.load(result['id'])
            if not self.metadata.get('live') and self.current_frame is not None:
                self.analysis_panel.review_analysis_id = result['id']
                self.worker.command('seek',self.current_frame.timestamp_ms)
        elif result.get('error'):
            self.analysis_panel.progress.setText('分析エラー: '+result['error'])
        self.notice.setText(('分析完了。自動推定を確認・修正してください。' if result['status']=='completed' else '分析を停止しました。') if not result.get('error') else '分析エラー: '+result['error'])
        self.refresh()

    def closeEvent(self, event):
        if not self.maybe_save():
            event.ignore()
            return
        if self.analysis_active:
            self.stop_analysis()
        self.closing = True
        self.worker.requestInterruption()
        if self.landmarks_panel.busy:
            self.landmarks_panel.task.requestInterruption()
        if not self.worker.wait(100) or self.landmarks_panel.busy or self.analysis_active:
            # Keep the Qt event loop responsive while a pending native read finishes.
            self.setEnabled(False)
            self.notice.setText('映像入力と録画を終了しています…')
            event.ignore()
            self.close_timer = QTimer(self)
            self.close_timer.timeout.connect(self.finish_close)
            self.close_timer.start(100)
        else:
            event.accept()

    def finish_close(self):
        if not self.worker.isRunning() and not self.landmarks_panel.busy and not self.analysis_active:
            self.close_timer.stop()
            self.workflow.dirty = False
            self.close()
