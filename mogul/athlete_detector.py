"""YOLOX-s person inference using OpenCV CPU. No torch/CUDA dependency."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/'models/yolox_manifest.json').read_text(encoding='utf-8'))
DEFAULT_MODEL=ROOT/MANIFEST['path']


@dataclass
class Detection:
    bbox: list  # normalized x1,y1,x2,y2
    confidence: float

    @property
    def foot(self):
        return [(self.bbox[0]+self.bbox[2])/2,self.bbox[3]]


class AthleteDetector:
    def __init__(self,path=DEFAULT_MODEL,threshold=.15,nms=.45):
        path=Path(path)
        if not path.is_file():
            raise ValueError('人物モデル未導入です。python -m tools.install_person_model を実行してください')
        self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        if self.sha256!=MANIFEST['sha256']:
            raise ValueError('モデルのSHA256が公式の固定版と一致しません。モデルを再導入してください')
        self.net=cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # Net's default target is CPU; OpenCV 5's new graph engine is CPU-only
        # and warns on setPreferableTarget even for CPU. Do not request GPU targets.
        self.threshold,self.nms=threshold,nms
        anchors=[]
        strides=[]
        for stride in [8,16,32]:
            yy,xx=np.mgrid[:640//stride,:640//stride]
            anchors.append(np.column_stack([xx.ravel(),yy.ravel()]))
            strides.extend([stride]*(640//stride)**2)
        self.anchors=np.concatenate(anchors)
        self.strides=np.asarray(strides)[:,None]

    def detect_course(self,image,region,tiled=False):
        crop,(x0,y0)=region.crop(image)
        height,width=image.shape[:2]
        ch,cw=crop.shape[:2]
        detections=self.detect(crop,[[0,0],[1,0],[1,1],[0,1]],tiled=tiled)
        result=[]
        for detection in detections:
            x1,y1,x2,y2=detection.bbox
            bbox=[(x0+x1*cw)/width,(y0+y1*ch)/height,(x0+x2*cw)/width,(y0+y2*ch)/height]
            mapped=Detection(bbox,detection.confidence)
            if region.can_observe(mapped.foot):
                result.append(mapped)
        return result

    def detect(self,image,middle=None,tiled=False):
        result=self._single(image,middle)
        if tiled and middle:
            points=np.asarray(middle)
            low=points.min(axis=0); high=points.max(axis=0)
            for u,v in [(0,0),(1,0),(0,1),(1,1)]:
                span=(high-low)*.65
                lo=low+(high-low-span)*[u,v]
                hi=lo+span
                region=[[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]]
                result.extend(self._single(image,region))
        if not result:
            return []
        boxes=[[d.bbox[0],d.bbox[1],d.bbox[2]-d.bbox[0],d.bbox[3]-d.bbox[1]] for d in result]
        keep=cv2.dnn.NMSBoxes(boxes,[d.confidence for d in result],self.threshold,self.nms)
        return [result[i] for i in np.asarray(keep).reshape(-1)]

    def _single(self,image,middle=None):
        height,width=image.shape[:2]
        x0=y0=0
        x1,y1=width,height
        if middle:
            points=np.asarray(middle)*[width,height]
            margin=np.array([width,height])*.04
            lo=np.maximum(0,np.floor(points.min(axis=0)-margin)).astype(int)
            hi=np.minimum([width,height],np.ceil(points.max(axis=0)+margin)).astype(int)
            x0,y0=lo
            x1,y1=hi
        crop=image[y0:y1,x0:x1]
        if crop.size==0:
            return []
        ratio=min(640/crop.shape[0],640/crop.shape[1])
        rw,rh=max(1,int(crop.shape[1]*ratio)),max(1,int(crop.shape[0]*ratio))
        canvas=np.full((640,640,3),114,np.float32)
        canvas[:rh,:rw]=cv2.cvtColor(cv2.resize(crop,(rw,rh)),cv2.COLOR_BGR2RGB)
        self.net.setInput(canvas.transpose(2,0,1)[None])
        output=np.asarray(self.net.forward()).reshape(-1,85).copy()
        return self.decode(output,ratio,(x0,y0),(width,height))

    def decode(self,output,ratio,offset,size):
        if output.shape!=(8400,85) or not np.isfinite(output).all():
            raise ValueError('人物モデルの出力形式が不正です')
        scores=output[:,4]*output[:,5]  # COCO class 0, person
        indices=np.flatnonzero((scores>=self.threshold)&(output[:,5]>=output[:,5:].max(axis=1)))
        if not len(indices):
            return []
        raw=output[indices]
        centers=(raw[:,:2]+self.anchors[indices])*self.strides[indices]
        extent=np.exp(np.clip(raw[:,2:4],-15,15))*self.strides[indices]
        boxes=np.column_stack([centers-extent/2,extent])/ratio
        boxes[:,:2]+=offset
        keep=cv2.dnn.NMSBoxes(boxes.tolist(),scores[indices].tolist(),self.threshold,self.nms)
        result=[]
        width,height=size
        for index in np.asarray(keep).reshape(-1)[:100]:
            x,y,w,h=boxes[index]
            bbox=np.clip([x/width,y/height,(x+w)/width,(y+h)/height],0,1).tolist()
            if bbox[2]>bbox[0] and bbox[3]>bbox[1]:
                result.append(Detection(bbox,float(scores[indices[index]])))
        return result
