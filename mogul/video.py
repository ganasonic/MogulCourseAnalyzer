"""File/UVC sources share one worker and frame interface. No Qt widgets here."""
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


@dataclass
class VideoFrame:
    image: np.ndarray
    timestamp_ms: float
    index: int
    source_id: int = 0


class VideoSource(ABC):
    live = False

    @abstractmethod
    def read(self) -> VideoFrame | None: ...

    @abstractmethod
    def close(self): ...

    def seek(self, milliseconds):
        raise ValueError('ライブ映像はシークできません')


class FileVideoSource(VideoSource):
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self.capture = cv2.VideoCapture(self.path)
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError('動画を開けません。パス・コーデックを確認してください')
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.metadata = {'kind': 'file', 'path': self.path, 'fps': self.fps,
                         'width': int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         'height': int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         'duration_ms': self.count / self.fps * 1000, 'live': False}
        try:
            result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=duration:format=duration', '-of', 'json', self.path],
                capture_output=True, timeout=8, check=True, creationflags=CREATE_FLAGS)
            probe = json.loads(result.stdout)
            duration = probe.get('streams', [{}])[0].get('duration', probe.get('format', {}).get('duration'))
            if duration and float(duration) > 0:
                self.metadata['duration_ms'] = float(duration)*1000
        except (OSError, ValueError, subprocess.SubprocessError, IndexError):
            pass  # OpenCV estimate remains available if ffprobe cannot inspect the file.

    def read(self):
        ok, image = self.capture.read()
        if not ok:
            return None
        index = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))-1)
        timestamp = self.capture.get(cv2.CAP_PROP_POS_MSEC)
        if not np.isfinite(timestamp) or (timestamp <= 0 and index > 0):
            timestamp = index/self.fps*1000
        return VideoFrame(image, timestamp, index)

    def seek(self, milliseconds):
        milliseconds = max(0, min(milliseconds, max(0, self.metadata['duration_ms']-1000/self.fps)))
        if not self.capture.set(cv2.CAP_PROP_POS_MSEC, milliseconds):
            raise ValueError('この動画ではシークできません')

    def close(self):
        self.capture.release()


def enumerate_uvc_devices():
    """Actual DirectShow video names, not speculative OpenCV index probing."""
    result = subprocess.run(['ffmpeg', '-hide_banner', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
                            capture_output=True, timeout=12, creationflags=CREATE_FLAGS)
    text = result.stderr.decode('utf-8', errors='replace')
    devices = []
    in_video = False
    for line in text.splitlines():
        if 'DirectShow video devices' in line:
            in_video = True
        if 'DirectShow audio devices' in line:
            in_video = False
        match = re.search(r'"(.*?)"', line)
        if not match:
            continue
        if 'Alternative name' in line:
            if devices and in_video:
                devices[-1]['device'] = match.group(1)
        elif '(video)' in line or (in_video and 'Alternative name' not in line):
            devices.append({'name': match.group(1), 'device': match.group(1)})
    return devices


class UVCVideoSource(VideoSource):
    """FFmpeg DirectShow selects the device by name (not an unstable numeric index)."""
    live = True

    def __init__(self, device, width, height, fps):
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.metadata = {'kind': 'uvc', 'device': device, 'width': self.width, 'height': self.height,
                         'fps': self.fps, 'duration_ms': 0, 'live': True}
        self.frames = queue.Queue(maxsize=2)
        self.errors = deque(maxlen=20)
        self.stopping = threading.Event()
        self.started = time.monotonic()
        self.index = 0
        self.process = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-f', 'dshow', '-video_size', f'{width}x{height}', '-framerate', str(fps),
            '-i', f'video={device}', '-an', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=CREATE_FLAGS)
        self.reader = threading.Thread(target=self._read_pipe, daemon=True)
        self.error_reader = threading.Thread(target=self._read_errors, daemon=True)
        self.reader.start()
        self.error_reader.start()

    def _read_errors(self):
        for line in iter(self.process.stderr.readline, b''):
            self.errors.append(line.decode('utf-8', errors='replace').strip())

    def _read_pipe(self):
        size = self.width*self.height*3
        while not self.stopping.is_set():
            data = bytearray()
            while len(data) < size:
                chunk = self.process.stdout.read(size-len(data))
                if not chunk:
                    return
                data.extend(chunk)
            frame = VideoFrame(np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3).copy(),
                               (time.monotonic()-self.started)*1000, self.index)
            self.index += 1
            if self.frames.full():
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
            self.frames.put_nowait(frame)

    def read(self):
        try:
            return self.frames.get(timeout=2)
        except queue.Empty:
            raise OSError('UVC入力なし／切断／非対応モード: ' + ' '.join(self.errors)[-700:])

    def close(self):
        self.stopping.set()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        self.reader.join(timeout=2)
        self.error_reader.join(timeout=2)
        self.process.stdout.close()
        self.process.stderr.close()


class VideoRecorder:
    def __init__(self, path, fps, size):
        self.path = str(path)
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise OSError('MP4録画を開始できません')
        self.timestamps = []

    def write(self, frame):
        self.writer.write(frame.image)
        self.timestamps.append(frame.timestamp_ms)

    def close(self):
        self.writer.release()
        Path(self.path + '.timestamps.json').write_text(json.dumps(self.timestamps), encoding='utf-8')


class VideoWorker(QThread):
    frame_ready = Signal(object)
    opened = Signal(dict)
    status = Signal(str, bool, int)
    recording = Signal(str, int)
    ended = Signal(int)
    playback = Signal(bool, int)

    def __init__(self, parent=None, factory=None):
        super().__init__(parent)
        self.commands = queue.Queue()
        self.factory = factory or self._factory

    @staticmethod
    def _factory(spec):
        return FileVideoSource(spec['path']) if spec['kind'] == 'file' else UVCVideoSource(
            spec['device'], spec['width'], spec['height'], spec['fps'])

    def command(self, name, value=None):
        self.commands.put((name, value))

    def run(self):
        source = recorder = None
        spec = None
        playing = False
        speed = 1.0
        last = None
        retry_at = 0
        request_id = 0
        live_epoch = 0
        live_index = 0
        anchor_wall = anchor_ms = 0
        pending = None

        def publish(frame):
            nonlocal live_index
            frame.source_id = request_id
            if source.live:
                frame.timestamp_ms = (time.monotonic()-live_epoch)*1000
                frame.index = live_index
                live_index += 1
            if recorder:
                recorder.write(frame)
            self.frame_ready.emit(frame)

        def announce_source():
            metadata = dict(source.metadata)
            metadata['_request_id'] = request_id
            self.opened.emit(metadata)

        def stop_recording():
            nonlocal recorder
            if recorder:
                current = recorder
                recorder = None
                try:
                    current.close()
                    self.recording.emit('保存済み: ' + current.path, request_id)
                except OSError as exc:
                    self.recording.emit('録画保存エラー: ' + str(exc), request_id)

        try:
            while not self.isInterruptionRequested():
                try:
                    name, value = self.commands.get(timeout=.01)
                except queue.Empty:
                    name = None
                try:
                    if name == 'open':
                        stop_recording()
                        if source:
                            source.close()
                        source = None
                        spec = value
                        request_id = spec.get('_request_id', 0)
                        live_epoch = time.monotonic()
                        live_index = 0
                        retry_at = 0
                        last = None
                        pending = None
                        source = self.factory(spec)
                        announce_source()
                        playing = source.live
                        self.playback.emit(playing, request_id)
                        last = source.read()
                        if last:
                            publish(last)
                            anchor_wall, anchor_ms = time.monotonic(), last.timestamp_ms
                            self.status.emit('入力正常', True, request_id)
                        else:
                            raise OSError('有効なフレームがありません')
                    elif name == 'play' and source:
                        playing = bool(value)
                        self.playback.emit(playing, request_id)
                        anchor_wall, anchor_ms = time.monotonic(), last.timestamp_ms if last else 0
                    elif name == 'speed':
                        speed = float(value)
                        anchor_wall, anchor_ms = time.monotonic(), last.timestamp_ms if last else 0
                    elif name in ('seek', 'step') and source and not source.live:
                        position = float(value) if name == 'seek' else ((last.timestamp_ms if last else 0) + int(value)*1000/source.fps)
                        source.seek(position)
                        pending = None
                        frame = source.read()
                        if frame:
                            last = frame
                            publish(frame)
                            anchor_wall, anchor_ms = time.monotonic(), frame.timestamp_ms
                    elif name == 'record' and source and source.live:
                        stop_recording()
                        try:
                            recorder = VideoRecorder(value, source.fps, (source.metadata['width'], source.metadata['height']))
                            self.recording.emit('録画中: ' + str(value), request_id)
                        except OSError as exc:
                            self.recording.emit('録画開始エラー: ' + str(exc), request_id)
                    elif name == 'stop_record':
                        stop_recording()

                    if source is None and spec and spec['kind'] == 'uvc' and time.monotonic() >= retry_at:
                        retry_at = time.monotonic()+3
                        source = self.factory(spec)
                        announce_source()
                        playing = True
                    if source and playing:
                        if pending is None:
                            pending = source.read()
                        frame = pending
                        if frame is None:
                            playing = False
                            self.playback.emit(False, request_id)
                            self.ended.emit(request_id)
                        elif source.live or time.monotonic() >= anchor_wall + max(0, frame.timestamp_ms-anchor_ms)/1000/speed:
                            last = frame
                            publish(frame)
                            pending = None
                            self.status.emit('入力正常', True, request_id)
                except Exception as exc:
                    self.status.emit(str(exc), False, request_id)
                    playing = False
                    self.playback.emit(False, request_id)
                    try:
                        stop_recording()
                    except OSError as recording_error:
                        self.recording.emit('録画保存エラー: ' + str(recording_error), request_id)
                    if source:
                        source.close()
                        source = None
                    pending = None
                    if spec and spec['kind'] == 'uvc':
                        retry_at = time.monotonic()+3
                    else:
                        spec = None
        finally:
            try:
                stop_recording()
            finally:
                if source:
                    source.close()


class DeviceScanner(QThread):
    found = Signal(list, str)

    def run(self):
        try:
            self.found.emit(enumerate_uvc_devices(), '')
        except Exception as exc:
            self.found.emit([], str(exc))
