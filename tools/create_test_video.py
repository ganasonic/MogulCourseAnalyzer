"""Generate a clearly labelled synthetic course video. Not AI/tracking output."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from mogul.geometry import interpolate


def create(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (960, 540))
    if not writer.isOpened():
        raise OSError('テスト動画を生成できません')
    try:
        for index in range(150):
            frame = np.zeros((540,960,3), np.uint8)
            frame[:] = (78, 63, 46)
            snow = np.array([[350,50],[610,50],[905,510],[55,510]], np.int32)
            cv2.fillConvexPoly(frame, snow, (228,230,228))
            for row in range(1, 10):
                t = row/10
                left = interpolate(snow[0], snow[3], t)
                right = interpolate(snow[1], snow[2], t)
                for lane in range(4):
                    p = interpolate(left, right, (lane+.35+(row%2)*.3)/4)
                    cv2.ellipse(frame, tuple(map(int,p)), (int(8+12*t), int(3+4*t)), 0, 0, 360, (193,201,204), -1)
            for row in range(1, 7):
                t = row/7
                for a,b,color in [(snow[0],snow[3],(235,130,60)), (snow[1],snow[2],(80,80,230))]:
                    p = np.array(interpolate(a,b,t),int)
                    cv2.line(frame, tuple(p), tuple(p-[0,18]), color, 2)
                    cv2.circle(frame, tuple(p-[0,18]), 4, color, -1)
            t = (index%125)/125
            p = interpolate([440,70], [350,490], t)
            cv2.circle(frame, tuple(map(int,p)), 6, (30,130,220), -1)
            cv2.putText(frame, 'SYNTHETIC TEST VIDEO - NO AI DETECTION', (18,28), cv2.FONT_HERSHEY_SIMPLEX, .65, (240,240,240), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Frame {index:03d} / {index/25:.2f}s', (20,520), cv2.FONT_HERSHEY_SIMPLEX, .6, (240,240,240), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/synthetic-course.mp4')
    args = parser.parse_args()
    print(create(args.output))
