import argparse
from pathlib import Path
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from .app import MainWindow


def main():
    parser = argparse.ArgumentParser(description='Mogul Course Analyzer v1.8 Phase 1')
    parser.add_argument('--video', help='起動時に読み込む動画')
    parser.add_argument('--database', default='data/calibration.sqlite')
    parser.add_argument('--smoke-seconds', type=float, default=0, help='起動確認用の自動終了秒数')
    parser.add_argument('--screenshot', help='起動確認時のPNG出力先')
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName('Mogul Course Analyzer')
    window = MainWindow(args.database, startup=not bool(args.video or args.smoke_seconds))
    window.show()
    if args.video:
        window.open_source({'kind': 'file', 'path': args.video})
    if args.smoke_seconds:
        def finish():
            if args.screenshot:
                Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
                window.grab().save(args.screenshot)
            window.workflow.dirty = False
            window.close()
        QTimer.singleShot(int(args.smoke_seconds*1000), finish)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
