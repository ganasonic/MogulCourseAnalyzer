"""Calibration data and validation, independent of Qt and capture devices."""
from dataclasses import asdict, dataclass, field
from copy import deepcopy
import numpy as np
from .geometry import valid_quad, inside, overlap, signed_area, interpolate

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
    # Explicit Phase 2 prerequisites: cannot be bypassed by Phase 1 GUI.
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
        self.gates_confirmed = self.bumps_confirmed = False
        return True

    def clear_lanes(self):
        self.lanes = []
        self.boundaries = []
        self.gates_confirmed = self.bumps_confirmed = False

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
        result = cls(**data)
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
        # Phase 1 never restores unsupported Phase 2 confirmations.
        result.gates_confirmed = result.bumps_confirmed = False
        return result
