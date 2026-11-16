"""Logical source orientation, shared by display and candidate detection."""
import numpy as np


def oriented_frame(image, rotation=0, mirrored=False):
    if rotation not in (0,90,180,270):
        raise ValueError('回転角は0/90/180/270度で指定してください')
    result = np.rot90(image,-rotation//90)
    if mirrored:
        result = result[:,::-1]
    return np.ascontiguousarray(result)


def rotated_point(point, delta, mirrored=False):
    x,y = point
    if mirrored:
        x = 1-x
    for _ in range((delta % 360)//90):
        x,y = 1-y,x
    if mirrored:
        x = 1-x
    return [x,y]


def rotate_calibration(c, rotation):
    if rotation not in (0,90,180,270):
        raise ValueError('回転角が不正です')
    delta = (rotation-c.rotation) % 360
    transform = lambda p:rotated_point(p,delta,c.mirrored)
    c.middle = [transform(p) for p in c.middle]
    c.boundaries = [[transform(p) for p in boundary] for boundary in c.boundaries]
    for gate in c.control_gates:
        gate.point = transform(gate.point)
        if gate.candidate_point is not None:
            gate.candidate_point = transform(gate.candidate_point)
    for bump in c.bumps:
        bump.center = transform(bump.center)
        bump.polygon = [transform(p) for p in bump.polygon]
    c.rotation = rotation
