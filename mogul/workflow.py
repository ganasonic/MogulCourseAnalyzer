"""Undo, validation and analysis prerequisites."""
from copy import deepcopy
from .geometry import valid_quad
from .landmarks import gate_errors, bump_errors


class CalibrationWorkflow:
    def __init__(self, calibration):
        self.calibration = calibration
        self.history = []
        self.journal = []
        self.dirty = False

    def checkpoint(self):
        self.history.append(deepcopy(self.calibration))
        self.journal.append(deepcopy(self.calibration))
        self.history = self.history[-100:]
        self.dirty = True

    def undo(self):
        if self.history:
            self.journal.append(deepcopy(self.calibration))
            self.calibration = self.history.pop()
            self.dirty = True
            return True
        return False

    def middle_changed(self, points):
        if not valid_quad(points):
            return False
        self.calibration.middle = deepcopy(points)
        self.calibration.middle_confirmed = False
        self.calibration.clear_lanes()
        self.calibration.candidate.pop('lanes', None)
        self.dirty = True
        return True

    def reasons(self, source_ok, require_landmarks=False):
        c = self.calibration
        reasons = []
        if not source_ok:
            reasons.append('映像ソースが正常ではありません')
        if not c.middle_confirmed:
            reasons.append('ミドルパートが未確定です')
        if not c.lanes:
            reasons.append('レーン数・位置が未設定です')
        elif not c.lanes_valid() or not all(x.confirmed for x in c.lanes):
            reasons.append('全レーンを確定してください')
        if require_landmarks and (not c.gates_confirmed or gate_errors(c)):
            reasons.append('ゲート設定が未確認です')
        if require_landmarks and (not c.bumps_confirmed or bump_errors(c)):
            reasons.append('コブ位置設定が未確認です')
        return reasons


class AnalysisController:
    def __init__(self, workflow):
        self.workflow = workflow

    def can_start(self, source_ok, require_landmarks=False):
        return not self.workflow.reasons(source_ok,require_landmarks=require_landmarks)
