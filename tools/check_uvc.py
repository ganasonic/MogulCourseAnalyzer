"""Explicit hardware check. Does not save camera images unless --record is set."""
import argparse
import json
import time
from mogul.video import UVCVideoSource, VideoRecorder, enumerate_uvc_devices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, help='一覧の0から始まる番号。省略時は一覧のみ')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--seconds', type=float, default=2)
    parser.add_argument('--record', help='任意のMP4保存先')
    args = parser.parse_args()
    devices = enumerate_uvc_devices()
    print(json.dumps(devices, ensure_ascii=False))
    if args.device is None:
        return 0
    if not 0 <= args.device < len(devices):
        parser.error('デバイス番号が一覧の範囲外です')
    source = recorder = None
    try:
        source = UVCVideoSource(devices[args.device]['device'], args.width, args.height, args.fps)
        if args.record:
            recorder = VideoRecorder(args.record, args.fps, (args.width, args.height))
        started = time.monotonic()
        count = 0
        first = last = None
        while time.monotonic()-started < args.seconds or count == 0:
            frame = source.read()
            count += 1
            first = frame.timestamp_ms if first is None else first
            last = frame.timestamp_ms
            if recorder:
                recorder.write(frame)
        print(json.dumps({'result':'PASS', 'frames': count, 'shape': list(frame.image.shape), 'first_ms':first, 'last_ms':last}))
        return 0
    except Exception as exc:
        print(json.dumps({'result':'FAIL','error':str(exc)}, ensure_ascii=False))
        return 1
    finally:
        if recorder:
            recorder.close()
        if source:
            source.close()


if __name__ == '__main__':
    raise SystemExit(main())
