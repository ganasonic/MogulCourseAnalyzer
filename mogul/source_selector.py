"""Source-selection dialog. Device enumeration runs away from the GUI thread."""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget)
from .video import DeviceScanner


class VideoSourceSelector(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('映像ソースを選択')
        self.resize(580, 330)
        self.spec = None
        self.scanner = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Mogul Course Analyzer — 映像入力'))
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        file_tab = QWidget()
        file_layout = QVBoxLayout(file_tab)
        file_layout.addWidget(QLabel('録画済み動画からコースを設定します。'))
        row = QHBoxLayout()
        self.path = QLineEdit()
        self.path.setPlaceholderText('MP4 / MOV / AVI / MKV …')
        row.addWidget(self.path)
        browse = QPushButton('ファイルを選択…')
        browse.clicked.connect(self.browse)
        row.addWidget(browse)
        file_layout.addLayout(row)
        self.tabs.addTab(file_tab, '録画済み動画')
        camera_tab = QWidget()
        form = QFormLayout(camera_tab)
        self.devices = QComboBox()
        self.devices.addItem('一覧を更新してください', None)
        form.addRow('UVCデバイス', self.devices)
        self.resolution = QComboBox()
        self.resolution.addItems(['640 × 480', '1280 × 720', '1920 × 1080'])
        self.resolution.setCurrentIndex(1)
        form.addRow('解像度', self.resolution)
        self.fps = QComboBox()
        self.fps.addItems(['15', '25', '30', '60'])
        self.fps.setCurrentText('30')
        form.addRow('FPS', self.fps)
        self.refresh = QPushButton('接続済みデバイス一覧を更新')
        self.refresh.clicked.connect(self.scan)
        form.addRow(self.refresh)
        form.addRow(QLabel('指定モードが機器に対応している必要があります。\n開始後にライブプレビューで入力状態を確認できます。'))
        self.tabs.addTab(camera_tab, 'UVCライブ映像')
        orientation_form = QFormLayout()
        self.rotation = QComboBox()
        for label,value in [('回転なし（横長・縦長）',0),('右へ90°（横倒し撮影）',90),('180°',180),('左へ90°（横倒し撮影）',270)]:
            self.rotation.addItem(label,value)
        orientation_form.addRow('入力後の表示回転',self.rotation)
        layout.addLayout(orientation_form)
        self.message = QLabel('')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QDialogButtonBox()
        self.open_button = buttons.addButton('選択して開く', QDialogButtonBox.AcceptRole)
        buttons.addButton('キャンセル', QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept_source)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.tabs.currentChanged.connect(lambda i: self.scan() if i == 1 and self.scanner is None else None)

    def browse(self):
        path, _ = QFileDialog.getOpenFileName(self, '動画ファイルを開く', '', '動画 (*.mp4 *.mov *.avi *.mkv *.m4v);;すべてのファイル (*)')
        if path:
            self.path.setText(path)

    def scan(self):
        if self.scanner and self.scanner.isRunning():
            return
        self.refresh.setEnabled(False)
        self.message.setText('DirectShowデバイスを調査中…')
        self.scanner = DeviceScanner(self)
        self.scanner.found.connect(self.scanned)
        self.scanner.start()

    def scanned(self, devices, error):
        self.devices.clear()
        for device in devices:
            self.devices.addItem(device['name'], device['device'])
        if not devices:
            self.devices.addItem('利用可能なUVC映像デバイスなし', None)
        self.message.setText(error or f'{len(devices)}台の映像デバイスを検出しました。')
        self.refresh.setEnabled(True)

    def accept_source(self):
        if self.tabs.currentIndex() == 0:
            if not self.path.text().strip():
                self.message.setText('動画ファイルを選択してください。')
                return
            self.spec = {'kind': 'file', 'path': self.path.text().strip()}
        else:
            device = self.devices.currentData()
            if not device:
                self.message.setText('接続済みUVCデバイスを選択してください。')
                return
            width, height = map(int, self.resolution.currentText().split(' × '))
            self.spec = {'kind': 'uvc', 'device': device, 'width': width, 'height': height, 'fps': int(self.fps.currentText())}
        self.spec['rotation'] = self.rotation.currentData()
        self.accept()

    def done(self, result):
        if self.scanner and self.scanner.isRunning():
            self.message.setText('デバイス調査の終了を待っています。完了後に閉じてください。')
            return
        super().done(result)
