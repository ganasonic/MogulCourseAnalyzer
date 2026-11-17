"""Calibration data and validation, independent of Qt and capture devices."""
from dataclasses import asdict, dataclass, field
from copy import deepcopy
import numpy as np
from .geometry import valid_quad, inside, overlap, signed_area, interpolate
from .landmarks import ControlGate, Bump, valid_point, gate_errors, bump_errors

NAMES = {2: ['ブルー', 'レッド'], 3: ['ブルー', 'センター', 'レッド'],
         4: ['ブルーアウト', 'ブルーイン', 'レッドイン', 'レッドアウト']}


@dataclass
class Lane:
    id: str
    name: str
    group: str
    color: str
    confirmed: bool = False


@dataclass
class Calibration:
    middle: list = field(default_factory=list)
    middle_confirmed: bool = False
    boundaries: list = field(default_factory=list)  # shared [[top_x, top_y], [bottom_x, bottom_y]]
    boundary_format: int = 2
    lanes: list[Lane] = field(default_factory=list)
    opacity: float = .3
    middle_color: str = '#78d6eb'
    source: dict = field(default_factory=dict)
    start_ms: float = 0
    mirrored: bool = False
    rotation: int = 0  # clockwise, applied after decoder metadata orientation
    analysis_config: dict = field(default_factory=dict)
    control_gates: list[ControlGate] = field(default_factory=list)
    bumps: list[Bump] = field(default_factory=list)
    gates_confirmed: bool = False
    bumps_confirmed: bool = False
    candidate: dict = field(default_factory=dict)

    def lane_polygon(self, index):
        top_left, bottom_left = self.boundaries[index]
        top_right, bottom_right = self.boundaries[index+1]
        return deepcopy([top_left, top_right, bottom_right, bottom_left])

    def configure_lanes(self, count):
        if not self.middle_confirmed or count not in NAMES:
            raise ValueError('ミドルパート確定後、2〜4レーンを指定してください')
        a, b, c, d = self.middle
        self.boundaries = [[interpolate(a, b, i/count), interpolate(d, c, i/count)] for i in range(count+1)]
        self.lanes = []
        for i, name in enumerate(NAMES[count]):
            group = 'neutral' if count == 3 and i == 1 else ('blue' if i < count/2 else 'red')
            self.lanes.append(Lane(chr(65+i), name, group,
                                   {'blue': '#64b5ff', 'red': '#ff8c96', 'neutral': '#c4cbd0'}[group]))
        if not self.lanes_valid():
            self.lanes = []
            self.boundaries = []
            raise ValueError('この形状はレーンに分割できません。ミドルパートの四隅を補正してください')
        self.invalidate_landmarks()

    def lanes_valid(self):
        if not valid_quad(self.middle) or len(self.lanes) not in NAMES or len(self.boundaries) != len(self.lanes)+1:
            return False
        try:
            points = np.asarray(self.boundaries, dtype=float)
        except (TypeError, ValueError):
            return False
        if points.shape != (len(self.lanes)+1, 2, 2) or not np.isfinite(points).all():
            return False
        polygons = [self.lane_polygon(i) for i in range(len(self.lanes))]
        if any(not valid_quad(p) or not inside(self.middle, p) for p in polygons):
            return False
        if any(overlap(p, q) > 2e-6 for i, p in enumerate(polygons) for q in polygons[i+1:]):
            return False
        # Shared full edges prevent gaps between adjacent lanes. The union need not
        # fill the middle part: the space between bumps and gates is intentional.
        return all(signed_area(p)*signed_area(self.middle) > 0 for p in polygons)

    def move_lane_corner(self, index, corner, point):
        if not (0 <= index < len(self.lanes) and corner in (0, 1, 2, 3)):
            return False
        boundary = index + (corner in (1, 2))
        bottom = corner in (2, 3)
        old = deepcopy(self.boundaries)
        self.boundaries[boundary][int(bottom)] = list(point)
        if not self.lanes_valid():
            self.boundaries = old
            return False
        for neighbor in (boundary-1, boundary):
            if 0 <= neighbor < len(self.lanes):
                self.lanes[neighbor].confirmed = False
        self.invalidate_landmarks()
        return True

    def invalidate_landmarks(self, gates=True, bumps=True, records=True):
        if gates:
            self.gates_confirmed = False
            if records:
                for gate in self.control_gates:
                    gate.confirmed = False
        if bumps:
            self.bumps_confirmed = False
            if records:
                for bump in self.bumps:
                    bump.confirmed = False

    def clear_lanes(self):
        self.lanes = []
        self.boundaries = []
        self.invalidate_landmarks()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = deepcopy(data)
        version = data.get('boundary_format', 1)
        if version not in (1, 2):
            raise ValueError('未対応のレーン座標形式です')
        if version == 1 and data.get('boundaries'):
            # Read old snapshots without modifying the original SQLite records.
            try:
                a, b, c, d = data['middle']
                fractions = np.asarray(data['boundaries'], dtype=float)
                if fractions.shape != (len(data.get('lanes', []))+1, 2) or not np.isfinite(fractions).all() or (fractions < 0).any() or (fractions > 1).any():
                    raise ValueError('旧レーン座標が不正です')
                data['boundaries'] = [[interpolate(a, b, t), interpolate(d, c, u)] for t, u in fractions]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('旧レーン座標が不正です') from exc
        data['boundary_format'] = 2
        data['lanes'] = [Lane(**x) for x in data.get('lanes', [])]
        data['control_gates'] = [ControlGate(**x) for x in data.get('control_gates', [])]
        data['bumps'] = [Bump(**x) for x in data.get('bumps', [])]
        result = cls(**data)
        if type(result.rotation) is not int or result.rotation not in (0,90,180,270):
            raise ValueError('保存された回転角が不正です')
        if not isinstance(result.analysis_config,dict):
            raise ValueError('解析設定が不正です')
        if result.middle and not valid_quad(result.middle):
            raise ValueError('保存されたミドルパートが不正です')
        if result.middle_confirmed and not result.middle:
            raise ValueError('ミドルパートがありません')
        if result.lanes and (not result.middle_confirmed or not result.lanes_valid()):
            raise ValueError('保存されたレーンが不正です')
        if not 0.05 <= result.opacity <= 1:
            raise ValueError('不透明度が不正です')
        if not isinstance(result.start_ms, (int, float)) or result.start_ms < 0:
            raise ValueError('解析開始時刻が不正です')
        identifiers = [item.id for item in result.control_gates + result.bumps]
        if any(not isinstance(i,str) or not i for i in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError('ゲート・コブの内部IDが不正です')
        for gate in result.control_gates:
            if not valid_point(gate.point) or gate.side not in ('LEFT','RIGHT') or (gate.number is not None and (type(gate.number) is not int or not 1 <= gate.number <= 9)):
                raise ValueError('保存されたゲートが不正です')
            if gate.candidate_point is not None and not valid_point(gate.candidate_point):
                raise ValueError('ゲート候補位置が不正です')
        for bump in result.bumps:
            if not valid_point(bump.center) or not valid_quad(bump.polygon) or bump.side not in ('L','R','UNKNOWN') or type(bump.gate_number) is not int or not 1 <= bump.gate_number <= 9 or type(bump.sequence) is not int or bump.sequence < 1:
                raise ValueError('保存されたコブが不正です')
        for confidence in [g.confidence for g in result.control_gates] + [b.side_confidence for b in result.bumps]:
            if confidence is not None and (not isinstance(confidence,(float,int)) or not np.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError('信頼度が不正です')
        if result.gates_confirmed and gate_errors(result):
            result.gates_confirmed = False
        if result.bumps_confirmed and bump_errors(result):
            result.bumps_confirmed = False
        return result
