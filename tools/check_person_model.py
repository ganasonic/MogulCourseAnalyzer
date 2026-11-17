"""Real CPU model smoke check and timing; does not save input images."""
import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import time
import cv2
import numpy as np
from mogul.athlete_detector import AthleteDetector,DEFAULT_MODEL


def peak_memory_mb():
    if not hasattr(ctypes,'WinDLL'):
        return None
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ['peak_ws','working_set','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private']]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32')
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi')
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        return round(counters.peak_ws/1024**2,1)
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image')
    parser.add_argument('--model',default=str(DEFAULT_MODEL))
    parser.add_argument('--tiles',action='store_true')
    args=parser.parse_args()
    image=(cv2.imdecode(np.fromfile(Path(args.image),np.uint8),cv2.IMREAD_COLOR) if args.image else np.zeros((360,640,3),np.uint8))
    if image is None:
        raise SystemExit('画像を読み込めません')
    model=AthleteDetector(args.model,threshold=.1)
    elapsed=[]
    for _ in range(3):
        started=time.perf_counter()
        detections=model.detect(image,[[0,0],[1,0],[1,1],[0,1]],tiled=args.tiles)
        elapsed.append(round((time.perf_counter()-started)*1000,1))
    print(json.dumps({'backend':'OpenCV CPU','model_sha256':model.sha256,'input_shape':list(image.shape),
        'tiled':args.tiles,'inference_ms':elapsed,'person_candidates':len(detections),
        'confidence':[round(d.confidence,3) for d in detections],'peak_process_working_set_mb':peak_memory_mb(),
        'note':'No precision/recall or skier-count accuracy measured.'},ensure_ascii=False))


if __name__=='__main__':
    main()
