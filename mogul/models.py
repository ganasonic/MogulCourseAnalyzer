"""Calibration data and validation, independent of Qt and capture devices."""
from dataclasses import asdict, dataclass, field
from copy import deepcopy
from .geometry import valid_quad, inside, overlap, area, interpolate, projection

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
    boundaries: list = field(default_factory=list)  # shared [top fraction, bottom fraction]
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
        a, b, c, d = self.middle
        t0, b0 = self.boundaries[index]
        t1, b1 = self.boundaries[index+1]
        return [interpolate(a, b, t0), interpolate(a, b, t1),
                interpolate(d, c, b1), interpolate(d, c, b0)]

    def configure_lanes(self, count):
        if not self.middle_confirmed or count not in NAMES:
            raise ValueError('ミドルパート確定後、2〜4レーンを指定してください')
        self.boundaries = [[i/count, i/count] for i in range(count+1)]
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
        if self.boundaries[0] != [0, 0] or self.boundaries[-1] != [1, 1]:
            return False
        if any(not (0 <= p[0] <= 1 and 0 <= p[1] <= 1) for p in self.boundaries):
            return False
        if any(self.boundaries[i][j] >= self.boundaries[i+1][j] for i in range(len(self.lanes)) for j in (0, 1)):
            return False
        polygons = [self.lane_polygon(i) for i in range(len(self.lanes))]
        if any(not valid_quad(p) or not inside(self.middle, p) for p in polygons):
            return False
        if any(overlap(p, q) > 2e-6 for i, p in enumerate(polygons) for q in polygons[i+1:]):
            return False
        return abs(sum(area(p) for p in polygons)-area(self.middle)) < 2e-6

    def move_lane_corner(self, index, corner, point):
        boundary = index + (corner in (1, 2))
        bottom = corner in (2, 3)
        if boundary in (0, len(self.lanes)):
            return False  # outer border belongs to the middle-part editor
        edge = (self.middle[3], self.middle[2]) if bottom else (self.middle[0], self.middle[1])
        old = deepcopy(self.boundaries)
        self.boundaries[boundary][int(bottom)] = projection(*edge, point)
        if not self.lanes_valid():
            self.boundaries = old
            return False
        for neighbor in (boundary-1, boundary):
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
