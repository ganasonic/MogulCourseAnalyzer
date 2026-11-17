"""Gate and bump registration independent of Qt. Coordinates are normalized."""
from dataclasses import dataclass, field
from copy import deepcopy
from uuid import uuid4
import cv2
import numpy as np
from .geometry import contains, cross, inside, valid_quad


@dataclass
class ControlGate:
    id: str
    side: str
    point: list
    number: int | None = None
    confidence: float | None = None
    confirmed: bool = False
    origin: str = 'manual'
    candidate_point: list | None = None


@dataclass
class Bump:
    id: str
    lane_id: str
    center: list
    polygon: list
    gate_number: int = 1
    sequence: int = 1
    side: str = 'UNKNOWN'  # skier's left/right, independent of display mirroring
    side_confidence: float | None = None
    confirmed: bool = False
    origin: str = 'manual'

    @property
    def label(self):
        suffix = '' if self.side == 'UNKNOWN' else '-' + self.side
        return f'{self.lane_id}-{self.gate_number}-{self.sequence}{suffix}'


def valid_point(point):
    try:
        p = np.asarray(point, float)
        return p.shape == (2,) and bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all())
    except (TypeError, ValueError):
        return False


def course_position(calibration, point):
    source = np.asarray(calibration.middle, np.float32)
    if not valid_quad(source) or not cv2.isContourConvex(source):
        raise ValueError('このミドルパートは透視変換できません。凸四角形に補正してください')
    matrix = cv2.getPerspectiveTransform(source, np.array([[0,0],[1,0],[1,1],[0,1]], np.float32))
    result = cv2.perspectiveTransform(np.asarray([[point]], np.float32), matrix)[0,0]
    if not np.isfinite(result).all():
        raise ValueError('コース座標へ変換できません')
    return result.tolist()


def layout_ready(c):
    return c.middle_confirmed and c.lanes_valid() and all(lane.confirmed for lane in c.lanes)


def gate_position_valid(c, gate):
    if not c.lanes or not valid_point(gate.point) or not c.middle or not contains(c.middle, gate.point):
        return False
    # Only the two OUTER lane edges are control gate boundaries. Internal lane
    # separators are never used. No fixed distance from a lane is imposed.
    if gate.side == 'LEFT':
        polygon = c.lane_polygon(0)
        edge = [polygon[0],polygon[3]]
        return cross(edge[0], edge[1], gate.point) > 1e-7
    if gate.side == 'RIGHT':
        polygon = c.lane_polygon(len(c.lanes)-1)
        edge = [polygon[1],polygon[2]]
        return cross(edge[0], edge[1], gate.point) < -1e-7
    return False


def bump_position_valid(c, bump):
    index = next((i for i,lane in enumerate(c.lanes) if lane.id == bump.lane_id), None)
    return (index is not None and valid_point(bump.center) and valid_quad(bump.polygon)
            and contains(bump.polygon, bump.center) and inside(c.lane_polygon(index), bump.polygon))


def gate_errors(c, require_confirmed=True):
    errors = []
    if not layout_ready(c):
        errors.append('ミドルパートと全レーンを先に確定してください')
    if not c.control_gates:
        errors.append('映っているゲートを1本以上登録してください')
    keys = set()
    for gate in c.control_gates:
        if not gate_position_valid(c, gate):
            errors.append('ゲートはミドルパート内かつ最外側レーンの外側に配置してください')
        if type(gate.number) is not int or not 1 <= gate.number <= 9:
            errors.append('ゲートにコース全体の番号（1〜9）を指定してください')
        key = (gate.side, gate.number)
        if key in keys:
            errors.append('同じ側のゲート番号が重複しています')
        keys.add(key)
        if require_confirmed and not gate.confirmed:
            errors.append('未確認のゲートがあります')
    # Perspective depth, not screen Y. Never pair left and right gates by Y.
    for side in ('LEFT','RIGHT'):
        gates = [g for g in c.control_gates if g.side == side and type(g.number) is int and 1 <= g.number <= 9]
        gates.sort(key=lambda g:g.number)
        if len(gates) > 1:
            try:
                depth = [course_position(c,g.point)[1] for g in gates]
                if any(b <= a for a,b in zip(depth,depth[1:])):
                    errors.append('ゲート番号を上流から下流の順に確認してください')
            except ValueError as exc:
                errors.append(str(exc))
    return list(dict.fromkeys(errors))


def bump_errors(c, require_confirmed=True):
    errors = []
    if not layout_ready(c):
        errors.append('ミドルパートと全レーンを先に確定してください')
    keys = set()
    if not c.bumps:
        errors.append('登録したコブがありません（人数分析には不要です）')
    for bump in c.bumps:
        if not bump_position_valid(c, bump):
            errors.append('コブの中心と領域を所属レーン内に配置してください')
        if type(bump.gate_number) is not int or not 1 <= bump.gate_number <= 9 or type(bump.sequence) is not int or bump.sequence < 1:
            errors.append('基準ゲート番号とコブ順序を確認してください')
        if bump.side not in ('L','R','UNKNOWN'):
            errors.append('コブ左右はL/R/UNKNOWNで指定してください')
        key = (bump.lane_id,bump.gate_number,bump.sequence)
        if key in keys:
            errors.append('レーン・ゲート番号・コブ順序が重複しています')
        keys.add(key)
        if require_confirmed and not bump.confirmed:
            errors.append('未確認のコブがあります')
    return list(dict.fromkeys(errors))


class ControlGateEditor:
    def __init__(self, calibration):
        self.c = calibration

    def add(self, side, point, **attributes):
        gate = ControlGate(str(uuid4()),side,list(point),**attributes)
        if not layout_ready(self.c) or not gate_position_valid(self.c,gate):
            raise ValueError('ゲートは確定済みミドルパート内、最外側レーンの外側に追加してください。レーン外側に余白が必要です')
        self.c.control_gates.append(gate)
        self.c.invalidate_landmarks(gates=True, bumps=True, records=False)
        return gate

    def move(self, identifier, point):
        gate = next(g for g in self.c.control_gates if g.id == identifier)
        previous = gate.point
        gate.point = list(point)
        if not gate_position_valid(self.c,gate):
            gate.point = previous
            return False
        gate.confirmed = False
        self.c.invalidate_landmarks(gates=True, bumps=True, records=False)
        return True

    def update(self, identifier, side, number):
        gate = next(g for g in self.c.control_gates if g.id == identifier)
        proposed = deepcopy(gate)
        proposed.side, proposed.number = side, number
        if not gate_position_valid(self.c,proposed) or type(number) is not int or not 1 <= number <= 9:
            raise ValueError('左右・位置・全体ゲート番号（1〜9）を確認してください')
        if any(g.id != identifier and (g.side,g.number) == (side,number) for g in self.c.control_gates):
            raise ValueError('同じ側のゲート番号が重複しています')
        gate.side, gate.number, gate.confirmed = side, number, True
        self.c.invalidate_landmarks(gates=True, bumps=True, records=False)

    def remove(self, identifier):
        self.c.control_gates = [g for g in self.c.control_gates if g.id != identifier]
        self.c.invalidate_landmarks(gates=True, bumps=True, records=False)

    def confirm(self):
        errors = gate_errors(self.c)
        if errors:
            raise ValueError(' ／ '.join(errors))
        self.c.gates_confirmed = True


class BumpRegistry:
    def __init__(self, calibration):
        self.c = calibration

    def add(self, lane_id, polygon, gate_number=1):
        sequence = max([b.sequence for b in self.c.bumps if b.lane_id == lane_id and b.gate_number == gate_number], default=0)+1
        bump = Bump(str(uuid4()),lane_id,np.mean(polygon,axis=0).tolist(),deepcopy(polygon),gate_number,sequence)
        if not layout_ready(self.c) or not bump_position_valid(self.c,bump):
            raise ValueError('選択したレーン内をドラッグしてコブの領域を指定してください')
        self.c.bumps.append(bump)
        self.c.bumps_confirmed = False
        return bump

    def add_center(self,lane_id,point,gate_number=1):
        index=next((i for i,lane in enumerate(self.c.lanes) if lane.id==lane_id),None)
        if index is None or not valid_point(point) or not contains(self.c.lane_polygon(index),point):
            raise ValueError('選択レーン内のコブの中心をクリックしてください')
        lane=np.asarray(self.c.lane_polygon(index),np.float32)
        unit=np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)
        matrix=cv2.getPerspectiveTransform(lane,unit)
        inverse=cv2.getPerspectiveTransform(unit,lane)
        u,v=cv2.perspectiveTransform(np.asarray([[point]],np.float32),matrix)[0,0]
        for fraction in [1,.5,.25,.125]:
            width,height=.12*fraction,.024*fraction
            left=float(np.clip(u-width/2,0,1-width)); top=float(np.clip(v-height/2,0,1-height))
            box=np.array([[[left,top],[left+width,top],[left+width,top+height],[left,top+height]]],np.float32)
            polygon=cv2.perspectiveTransform(box,inverse)[0].tolist()
            proposed=Bump('',lane_id,list(point),polygon,gate_number)
            if bump_position_valid(self.c,proposed):
                bump=self.add(lane_id,polygon,gate_number)
                bump.center=list(point)
                bump.origin='manual-center'
                return bump
        raise ValueError('中心の周囲に仮領域を作れません。少し内側をクリックしてください')

    def add_line(self,lane_id,start,end,count,gate_number=1):
        if type(count) is not int or not 2<=count<=250 or np.linalg.norm(np.asarray(start)-end)<.001:
            raise ValueError('異なる2点と2〜250個を指定してください')
        index=next((i for i,lane in enumerate(self.c.lanes) if lane.id==lane_id),None)
        if index is None:
            raise ValueError('レーンを選択してください')
        lane=np.asarray(self.c.lane_polygon(index),np.float32)
        unit=np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)
        matrix=cv2.getPerspectiveTransform(lane,unit)
        inverse=cv2.getPerspectiveTransform(unit,lane)
        ends=cv2.perspectiveTransform(np.asarray([[start,end]],np.float32),matrix)[0]
        coordinates=np.asarray([[ends[0]*(1-t)+ends[1]*t for t in np.linspace(0,1,count)]],np.float32)
        points=cv2.perspectiveTransform(coordinates,inverse)[0]
        previous=deepcopy(self.c.bumps)
        confirmed=self.c.bumps_confirmed
        try:
            result=[self.add_center(lane_id,p.tolist(),gate_number) for p in points]
            for bump in result:
                bump.origin='manual-batch'
            return result
        except ValueError:
            self.c.bumps=previous
            self.c.bumps_confirmed=confirmed
            raise

    def confirm_all(self):
        errors=bump_errors(self.c,require_confirmed=False)
        if errors:
            raise ValueError(' ／ '.join(errors))
        for bump in self.c.bumps:
            bump.confirmed=True
        self.c.bumps_confirmed=True

    def move(self, identifier, point):
        bump = next(b for b in self.c.bumps if b.id == identifier)
        delta = np.asarray(point)-bump.center
        candidate = deepcopy(bump)
        candidate.center = list(point)
        candidate.polygon = (np.asarray(bump.polygon)+delta).tolist()
        if not bump_position_valid(self.c,candidate):
            return False
        bump.center, bump.polygon, bump.confirmed = candidate.center,candidate.polygon,False
        self.c.bumps_confirmed = False
        return True

    def resize(self, identifier, corner, point):
        bump = next(b for b in self.c.bumps if b.id == identifier)
        candidate = deepcopy(bump)
        candidate.polygon[corner] = list(point)
        candidate.center = np.mean(candidate.polygon,axis=0).tolist()
        if not bump_position_valid(self.c,candidate):
            return False
        bump.polygon, bump.center, bump.confirmed = candidate.polygon,candidate.center,False
        self.c.bumps_confirmed = False
        return True

    def update(self, identifier, lane_id, gate_number, sequence, side):
        bump = next(b for b in self.c.bumps if b.id == identifier)
        candidate = deepcopy(bump)
        candidate.lane_id, candidate.gate_number, candidate.sequence, candidate.side = lane_id,gate_number,sequence,side
        if not bump_position_valid(self.c,candidate) or type(gate_number) is not int or not 1 <= gate_number <= 9 or type(sequence) is not int or sequence < 1 or side not in ('L','R','UNKNOWN'):
            raise ValueError('所属レーン・位置・ゲート番号・コブ順序・左右を確認してください')
        if any(b.id != identifier and (b.lane_id,b.gate_number,b.sequence) == (lane_id,gate_number,sequence) for b in self.c.bumps):
            raise ValueError('同じコブラベルが既に登録されています')
        bump.lane_id, bump.gate_number, bump.sequence, bump.side = lane_id,gate_number,sequence,side
        bump.side_confidence = 1.0 if side != 'UNKNOWN' else None
        bump.confirmed = True
        self.c.bumps_confirmed = False

    def remove(self, identifier):
        self.c.bumps = [b for b in self.c.bumps if b.id != identifier]
        self.c.bumps_confirmed = False

    def confirm(self):
        errors = bump_errors(self.c)
        if errors:
            raise ValueError(' ／ '.join(errors))
        self.c.bumps_confirmed = True


@dataclass
class GateDetectionResult:
    candidates: list = field(default_factory=list)
    message: str = ''


class ControlGateDetector:
    """HSV color candidates, explicitly NOT an AI model or a calibrated probability."""
    def detect(self, frames, calibration, cancelled=lambda:False):
        candidates = []
        for frame in frames:
            if cancelled():
                return GateDetectionResult([], '候補抽出を中止しました')
            h,w = frame.shape[:2]
            scale = min(1,960/max(h,w))
            image = cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))))
            hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            roi = np.zeros(image.shape[:2],np.uint8)
            vertices = np.rint(np.asarray(calibration.middle)*[image.shape[1],image.shape[0]]).astype(np.int32)
            cv2.fillPoly(roi,[vertices],255)
            masks = [cv2.inRange(hsv,np.array([90,100,60]),np.array([135,255,255])),
                     cv2.inRange(hsv,np.array([0,100,60]),np.array([12,255,255])) | cv2.inRange(hsv,np.array([170,100,60]),np.array([179,255,255]))]
            for mask in masks:
                mask = cv2.bitwise_and(mask,roi)
                contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    fraction = cv2.contourArea(contour)/(image.shape[0]*image.shape[1])
                    if not .000015 <= fraction <= .01:
                        continue
                    moments = cv2.moments(contour)
                    if not moments['m00']:
                        continue
                    point = [moments['m10']/moments['m00']/image.shape[1], moments['m01']/moments['m00']/image.shape[0]]
                    side = next((side for side in ('LEFT','RIGHT') if gate_position_valid(calibration,ControlGate('',side,point))),None)
                    if side is None or any(side == c['side'] and np.linalg.norm(np.array(point)-c['point']) < .025 for c in candidates):
                        continue
                    candidates.append({'side':side,'point':point,'confidence':None,'origin':'HSV','candidate_point':point[:]})
                    if len(candidates) >= 40:
                        return GateDetectionResult(candidates,'色抽出候補が40件に達しました。誤検出を削除し、番号・位置を手動確認してください（AI未導入）。')
        return GateDetectionResult(candidates,f'色抽出（AI未導入）: {len(candidates)}候補。番号・位置は未確定です。検出精度は未評価のため必ず確認してください。')


class BumpDetector:
    def detect(self, frames, calibration):
        return [], 'コブAIモデル未導入。手動登録してください。'
