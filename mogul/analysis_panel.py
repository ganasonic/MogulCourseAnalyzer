"""Analysis settings, progress, saved run review and audited manual correction."""
import bisect
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox,QComboBox,QDoubleSpinBox,QFileDialog,QFormLayout,
    QLabel,QLineEdit,QListWidget,QPushButton,QScrollArea,QSpinBox,QVBoxLayout,QWidget)
from .athlete_detector import DEFAULT_MODEL
from .analytics import summarize
from .analysis_repository import AnalysisRepository
from .models import Calibration

STATUS_LABELS={'running':'実行中','completed':'完了','stopped':'停止','failed':'失敗','interrupted':'中断'}


class AnalysisPanel(QScrollArea):
    def __init__(self,window):
        super().__init__(window)
        self.window=window
        self.data=None
        self.selected=None
        self.loading=False
        self.review_analysis_id=None
        self.review_end_ms=None
        self.setWidgetResizable(True)
        body=QWidget(); layout=QVBoxLayout(body)
        self.model=QLineEdit(str(DEFAULT_MODEL))
        self.model.setReadOnly(True)
        layout.addWidget(QLabel('人物モデル：YOLOX-s / OpenCV CPU'))
        layout.addWidget(self.model)
        self.choose_model=QPushButton('モデルファイルを指定…')
        self.choose_model.clicked.connect(self.browse_model)
        layout.addWidget(self.choose_model)
        self.model_hint=QLabel('未導入時: .venvのPythonで\npython -m tools.install_person_model\n公式の固定版モデルのみ対応。利用料はかかりません。')
        self.model_hint.setWordWrap(True)
        layout.addWidget(self.model_hint)
        form=QFormLayout()
        self.processing_mode=QComboBox()
        self.processing_mode.addItem('再生優先（1倍速・解析は間引き）','realtime')
        self.processing_mode.addItem('全フレーム解析（処理速度で再生）','all_frames')
        form.addRow('分析方式',self.processing_mode)
        mode_hint=QLabel('再生優先では未解析フレームが生じ、短い滑走やレーン変更を見逃す場合があります。\n人物枠は解析した同じフレームにだけ表示します。')
        mode_hint.setWordWrap(True)
        layout.addWidget(mode_hint)
        self.confidence=QDoubleSpinBox()
        self.confidence.setRange(.1,.95); self.confidence.setSingleStep(.05); self.confidence.setValue(.35)
        self.confidence.setToolTip('低いしきい値では小さい人物を拾いやすくなりますが、誤検出も増えます。確認・修正してください。')
        form.addRow('新規追跡の検出しきい値',self.confidence)
        self.tiles=QCheckBox('小さい人物向けの分割推論（処理時間増加）')
        self.tiles.setChecked(True)
        form.addRow(self.tiles)
        layout.addLayout(form)
        self.confidence.valueChanged.connect(self.settings_changed)
        self.processing_mode.currentIndexChanged.connect(self.settings_changed)
        self.tiles.toggled.connect(self.settings_changed)
        self.model.textChanged.connect(self.settings_changed)
        self.stop=QPushButton('■ 分析停止')
        self.stop.clicked.connect(window.stop_analysis)
        self.stop.setEnabled(False)
        layout.addWidget(self.stop)
        self.progress=QLabel('未実行。位置設定とモデルを確認して「分析開始」を押してください。')
        self.progress.setWordWrap(True)
        layout.addWidget(self.progress)
        self.summary=QLabel('集計なし')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        layout.addWidget(QLabel('自動推定は未確認を含みます。\n停止後に人数・レーン・追跡の統合を修正できます。'))
        self.history=QComboBox()
        layout.addWidget(self.history)
        refresh=QPushButton('保存済み解析の一覧を更新')
        refresh.clicked.connect(self.refresh_history)
        layout.addWidget(refresh)
        self.load_button=QPushButton('選択した解析結果を開く')
        self.load_button.clicked.connect(self.load_selected)
        layout.addWidget(self.load_button)
        self.runs=QListWidget()
        self.runs.setMinimumHeight(160)
        self.runs.currentItemChanged.connect(self.select_run)
        layout.addWidget(self.runs)
        self.details=QLabel('')
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        form=QFormLayout()
        self.counted=QCheckBox('滑走として集計する')
        form.addRow(self.counted)
        self.start_lane=QComboBox()
        form.addRow('滑走開始レーン',self.start_lane)
        self.review_state=QComboBox()
        for label,value in [('未確認の自動推定','unreviewed'),('確認済み','confirmed'),('除外','excluded')]:
            self.review_state.addItem(label,value)
        form.addRow('確認状態',self.review_state)
        self.changes=QSpinBox(); self.changes.setRange(0,9999)
        form.addRow('コースチェンジ件数',self.changes)
        self.passed_widget=QWidget()
        self.passed_layout=QVBoxLayout(self.passed_widget)
        self.passed_checks={}
        form.addRow('通過したレーン',self.passed_widget)
        self.target=QComboBox()
        form.addRow('同じ滑走として統合する先',self.target)
        self.manual_lane=QComboBox()
        form.addRow('見逃し滑走の追加レーン',self.manual_lane)
        self.manual_time=QDoubleSpinBox(); self.manual_time.setRange(0,8640000); self.manual_time.setDecimals(3)
        form.addRow('手動追加時刻（秒）',self.manual_time)
        self.reason=QLineEdit(); self.reason.setPlaceholderText('修正・統合・追加の理由（必須）')
        form.addRow(self.reason)
        layout.addLayout(form)
        self.save_review=QPushButton('選択滑走の修正を保存')
        self.save_review.clicked.connect(self.apply_review)
        layout.addWidget(self.save_review)
        self.merge=QPushButton('選択滑走を統合先と同一滑走にする')
        self.merge.clicked.connect(self.merge_run)
        layout.addWidget(self.merge)
        self.unmerge=QPushButton('選択滑走の統合を解除')
        self.unmerge.clicked.connect(self.unmerge_run)
        layout.addWidget(self.unmerge)
        self.add=QPushButton('見逃した滑走を手動追加')
        self.add.clicked.connect(self.add_run)
        layout.addWidget(self.add)
        self.video=QPushButton('選択滑走の関連動画を前後2秒から確認')
        self.video.clicked.connect(self.open_video)
        layout.addWidget(self.video)
        self.interrupted=QPushButton('未完了の保存記録を中断扱いにする')
        self.interrupted.clicked.connect(self.mark_interrupted)
        layout.addWidget(self.interrupted)
        layout.addStretch()
        self.setWidget(body)
        self.set_active(False)

    def config(self):
        return {'model_path':self.model.text(),'confidence':self.confidence.value(),'tiles':self.tiles.isChecked(),
                'processing_mode':self.processing_mode.currentData(),
                'tracker':'two-stage-predictive-IoU','backend':'OpenCV CPU'}

    def apply_config(self,config):
        self.loading=True
        self.processing_mode.setCurrentIndex(max(0,self.processing_mode.findData(config.get('processing_mode','realtime'))))
        self.model.setText(config.get('model_path',str(DEFAULT_MODEL)))
        self.confidence.setValue(config.get('confidence',.35))
        self.tiles.setChecked(config.get('tiles',True))
        self.loading=False

    def settings_changed(self):
        if not self.loading and not self.window.analysis_active:
            self.window.workflow.checkpoint()
            self.window.workflow.calibration.analysis_config=self.config()
            self.window.refresh_state()

    def browse_model(self):
        path,_=QFileDialog.getOpenFileName(self,'固定版YOLOXモデルを指定',str(DEFAULT_MODEL.parent),'ONNX (*.onnx)')
        if path:
            self.model.setText(path)

    def set_active(self,active):
        self.stop.setEnabled(active)
        for widget in [self.processing_mode,self.model,self.choose_model,self.confidence,self.tiles,self.history,self.load_button,
                       self.save_review,self.merge,self.unmerge,self.add,self.video,self.interrupted]:
            widget.setEnabled(not active)

    def show_packet(self,packet):
        if self.data is None or self.data.get('id')!=packet['analysis_id']:
            self.data={'id':packet['analysis_id'],'calibration':self.window.workflow.calibration.to_dict(),'status':'running'}
        self.data['runs']=packet['runs']
        self.show_summary(packet['summary'])
        self.progress.setText(f"解析 {packet['processed']}フレーム / 未解析 {packet['dropped']}枚\n処理 {packet['inference_ms']:.0f}ms/枚 / 最終解析時刻 {packet['frame'].timestamp_ms/1000:.2f}秒\n再生時刻は映像下の時計を参照してください。")
        self.fill_runs()

    def show_summary(self,summary):
        lines=[f"ユニーク滑走 {summary['unique_runs']} / 集計内未確認 {summary['unreviewed']}",
               '開始人数: '+' / '.join(f'{key} {value}' for key,value in summary['starts'].items()),
               '通過人数: '+' / '.join(f'{key} {value}' for key,value in summary['passes'].items()),
               f"ブルー {summary['groups']['blue']} / レッド {summary['groups']['red']} / 中立 {summary['groups']['neutral']}",
               f"コースチェンジ {summary['course_changes']}件"]
        self.summary.setText('\n'.join(lines))

    def fill_runs(self):
        self.runs.blockSignals(True)
        self.runs.clear()
        for run in self.data.get('runs',[]):
            state='除外' if run.get('review')=='excluded' else ('集計' if run['qualified'] else '未集計')
            label=f"{run['id'][:8]} / T{','.join(map(str,run['track_ids'])) or '手動'} / {state}"
            label+=' / 確認済' if run.get('review')=='confirmed' else ' / 未確認'
            if run.get('relinked'):
                label+=' / 再接続'
            if run.get('merged_into'):
                label+=' / 統合済み'
            self.runs.addItem(label)
            item=self.runs.item(self.runs.count()-1)
            item.setData(Qt.UserRole,run['id'])
            if run['id']==self.selected:
                self.runs.setCurrentItem(item)
        self.runs.blockSignals(False)

    def refresh_history(self):
        if self.window.analysis_active:
            return
        self.history.clear()
        if self.window.database.exists():
            try:
                for entry in AnalysisRepository(self.window.database).list_analyses():
                    stamp=datetime.fromisoformat(entry['created_at']).astimezone(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')
                    self.history.addItem(f"{stamp} JST / {STATUS_LABELS.get(entry['status'],entry['status'])} / {entry['id'][:8]}",entry['id'])
            except Exception as exc:
                self.progress.setText('履歴読込エラー: '+str(exc))

    def load_selected(self):
        if self.history.currentData() and not self.window.analysis_active:
            self.load(self.history.currentData())

    def load(self,identifier):
        try:
            if self.review_analysis_id is not None and self.review_analysis_id!=identifier:
                self.review_analysis_id=None
                self.review_end_ms=None
                self.window.worker.command('play',False)
                self.window.view.analysis_tracks=[]
                self.window.view.redraw()
            self.data=AnalysisRepository(self.window.database).load_analysis(identifier)
            c=Calibration.from_dict(self.data['calibration'])
            self.show_summary(summarize(self.data['runs'],c.lanes))
            self.fill_runs()
            mode='再生優先' if self.data.get('config',{}).get('processing_mode')=='realtime' else '全フレーム／ライブ'
            self.progress.setText(f"{mode} / {STATUS_LABELS.get(self.data['status'],self.data['status'])} / 解析 {self.data['frames_processed']}枚 / 未解析 {self.data['dropped_frames']}枚"+('\n'+self.data['error'] if self.data.get('error') else ''))
            self.loading=True
            previous=self.manual_lane.currentData()
            self.manual_lane.clear()
            self.start_lane.clear(); self.start_lane.addItem('開始不明／途中から検出',None)
            while self.passed_layout.count():
                item=self.passed_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.passed_checks={}
            for lane in c.lanes:
                self.manual_lane.addItem(lane.id,lane.id)
                self.start_lane.addItem(lane.id,lane.id)
                check=QCheckBox(lane.id)
                self.passed_checks[lane.id]=check
                self.passed_layout.addWidget(check)
            self.manual_lane.setCurrentIndex(max(0,self.manual_lane.findData(previous)))
            self.target.clear()
            self.target.addItem('統合先を選択',None)
            for run in self.data['runs']:
                if not run.get('merged_into') and run.get('review')!='excluded':
                    self.target.addItem(run['id'][:8],run['id'])
            self.loading=False
            if self.selected:
                self.select_run(self.runs.currentItem())
        except Exception as exc:
            self.progress.setText('結果読込エラー: '+str(exc))

    def select_run(self,item,previous=None):
        if not item or not self.data:
            return
        self.selected=item.data(Qt.UserRole)
        run=next(r for r in self.data['runs'] if r['id']==self.selected)
        self.counted.setChecked(run['qualified'])
        self.start_lane.setCurrentIndex(max(0,self.start_lane.findData(run.get('start_lane'))))
        self.review_state.setCurrentIndex(self.review_state.findData(run.get('review','unreviewed')))
        for lane,check in self.passed_checks.items():
            check.setChecked(lane in run.get('passed_lanes',[]))
        self.changes.setValue(run.get('change_count_override') if run.get('change_count_override') is not None else len(run.get('changes',[])))
        self.manual_time.setValue(run['first_ms']/1000)
        calibration=self.data.get('calibration',{})
        bumps=(', '.join(b['label'] for b in run.get('bumps',[])) or '登録済み範囲で該当なし') if calibration.get('bumps_confirmed') else '未設定（人数分析には不要）'
        self.details.setText(f"滑走 {run['id'][:8]} / {run['first_ms']/1000:.2f}〜{run['last_ms']/1000:.2f}秒\n通過コブ: {bumps}\n"+('追跡ID再接続あり・要確認' if run.get('relinked') else ''))

    def edit(self,action):
        if self.window.analysis_active or not self.data:
            return
        try:
            action(AnalysisRepository(self.window.database))
            self.load(self.data['id'])
            if self.review_analysis_id==self.data['id'] and self.window.current_frame is not None:
                self.window.on_frame(self.window.current_frame)
            self.progress.setText('修正と理由を保存し、人数を再集計しました。')
        except Exception as exc:
            self.progress.setText('修正できません: '+str(exc))

    def apply_review(self):
        if self.selected:
            changes={'qualified':self.counted.isChecked(),'start_lane':self.start_lane.currentData(),
                     'passed_lanes':[lane for lane,check in self.passed_checks.items() if check.isChecked()],
                     'review':self.review_state.currentData(),'change_count_override':self.changes.value()}
            self.edit(lambda repo:repo.review(self.data['id'],self.selected,changes,self.reason.text()))

    def merge_run(self):
        if self.selected and self.target.currentData():
            self.edit(lambda repo:repo.review(self.data['id'],self.selected,{'merged_into':self.target.currentData(),'review':'confirmed'},self.reason.text()))

    def unmerge_run(self):
        if self.selected:
            self.edit(lambda repo:repo.review(self.data['id'],self.selected,{'merged_into':None,'review':'confirmed'},self.reason.text()))

    def reset_result(self):
        self.data=None
        self.selected=None
        self.review_analysis_id=None
        self.review_end_ms=None
        self.runs.clear()
        self.details.clear()
        self.reason.clear()
        self.summary.setText('集計なし')
        self.progress.setText('未実行。位置設定とモデルを確認して「分析開始」を押してください。')

    def add_run(self):
        def add(repo):
            self.selected=repo.add_manual(self.data['id'],self.manual_lane.currentData(),self.manual_time.value()*1000,self.reason.text())
        self.edit(add)

    def mark_interrupted(self):
        if self.data and self.data['status']=='running':
            self.edit(lambda repo:repo.finish(self.data['id'],'interrupted','ユーザーが未完了記録を中断扱いにしました'))

    def open_video(self):
        if not self.data or not self.selected or self.window.analysis_active:
            return
        run=next(r for r in self.data['runs'] if r['id']==self.selected)
        c=Calibration.from_dict(self.data['calibration'])
        start=max(0,run['first_ms']-2000)
        end=run['last_ms']+2000
        source=c.source
        if source.get('kind')=='uvc':
            source=None
            for path in self.data.get('recordings',[])+c.source.get('recordings',[]):
                sidecar=Path(path+'.timestamps.json')
                if not sidecar.is_file():
                    continue
                times=json.loads(sidecar.read_text(encoding='utf-8'))
                if times and times[0]<=run['first_ms']<=times[-1]:
                    fps=c.source.get('fps',30)
                    start=bisect.bisect_left(times,start)/fps*1000
                    end=bisect.bisect_right(times,end)/fps*1000
                    source={'kind':'file','path':path}
                    break
            if source is None:
                self.progress.setText('この時刻を含む関連録画がありません。録画中なら停止・保存してから確認してください。')
                return
        if not Path(source.get('path','')).is_file():
            self.progress.setText('関連動画が見つかりません。元のパスに戻してください。')
            return
        if not self.window.maybe_save():
            return
        c.source=source
        c.start_ms=start
        self.window.workflow.calibration=c
        self.window.workflow.history.clear(); self.window.workflow.journal.clear()
        self.window.workflow.dirty=False
        self.window.session_id=self.data['session_id']
        self.window.sync_settings()
        self.window.open_source(source)
        self.review_analysis_id=self.data['id'] if self.data['calibration']['source'].get('kind')=='file' else None
        self.review_end_ms=end
        self.window.worker.command('play',True)
