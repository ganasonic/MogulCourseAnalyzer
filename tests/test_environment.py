"""Development environment smoke checks; no application implementation."""

import subprocess

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel


def test_qt_event_loop(qtbot):
    widget = QLabel("MogulCourseAnalyzer environment check")
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitExposed(widget)
    qtbot.wait(100)
    assert widget.isVisible()
    widget.close()


def test_opencv_image_roundtrip(tmp_path):
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:, :, 1] = 127
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), frame)
    np.testing.assert_array_equal(cv2.imread(str(path)), frame)


def test_excel_roundtrip(tmp_path):
    expected = pd.DataFrame({"frame": [0, 1], "value": [1.5, 2.5]})
    path = tmp_path / "check.xlsx"
    expected.to_excel(path, index=False, engine="openpyxl")
    pd.testing.assert_frame_equal(pd.read_excel(path, engine="openpyxl"), expected)


def test_ffmpeg_video_roundtrip(tmp_path):
    path = tmp_path / "check.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=size=64x48:rate=5", "-t", "1", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True, timeout=30,
    )
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        assert ok
        assert frame.shape == (48, 64, 3)
    finally:
        capture.release()
