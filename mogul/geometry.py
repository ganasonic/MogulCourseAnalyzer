"""Resolution-independent geometry. Coordinates are in [0, 1]."""
import cv2
import numpy as np

EPS = 1e-7


def cross(a, b, c):
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    return float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))


def area(points):
    p = np.asarray(points, float)
    return abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))) / 2


def on_segment(a, b, p):
    return abs(cross(a, b, p)) < EPS and all(min(a[i], b[i])-EPS <= p[i] <= max(a[i], b[i])+EPS for i in (0, 1))


def intersects(a, b, c, d):
    cs = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    return (cs[0]*cs[1] < 0 and cs[2]*cs[3] < 0) or any([
        on_segment(a, b, c), on_segment(a, b, d), on_segment(c, d, a), on_segment(c, d, b)])


def valid_quad(points):
    p = np.asarray(points, float)
    if p.shape != (4, 2) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        return False
    if area(p) < 1e-5 or any(np.linalg.norm(p[i]-p[(i+1) % 4]) < 1e-4 for i in range(4)):
        return False
    if intersects(p[0], p[1], p[2], p[3]) or intersects(p[1], p[2], p[3], p[0]):
        return False
    return all(abs(cross(p[i-1], p[i], p[(i+1) % 4])) > EPS for i in range(4))


def contains(polygon, point):
    # cv2 contours are float32; shared-edge interpolation needs a small distance tolerance.
    return cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(map(float, point)), True) >= -EPS


def triangles(quad):
    p = np.asarray(quad, float)
    # Choose an interior diagonal, including for a concave quadrilateral.
    for a, b in [(0, 2), (1, 3)]:
        t = [p[[a, (a+1) % 4, b]], p[[a, b, (b+1) % 4]]]
        if abs(sum(area(x) for x in t)-area(p)) < EPS:
            return t
    raise ValueError('不正な四角形です')


def overlap(a, b):
    # Double-precision clipping avoids OpenCV float32 intersection instability when
    # a triangle shares an almost-collinear boundary with another triangle.
    def clipped_area(subject, clip):
        output = list(subject)
        sign = 1 if cross(*clip) > 0 else -1
        for i in range(3):
            if not output:
                return 0.0
            edge_a, edge_b = clip[i], clip[(i+1) % 3]
            inputs, output = output, []
            previous = inputs[-1]
            dp = sign*cross(edge_a, edge_b, previous)
            for current in inputs:
                dc = sign*cross(edge_a, edge_b, current)
                if (dc >= -EPS) != (dp >= -EPS) and abs(dp-dc) > 1e-15:
                    output.append(previous + (current-previous)*(dp/(dp-dc)))
                if dc >= -EPS:
                    output.append(current)
                previous, dp = current, dc
        return area(output) if len(output) >= 3 else 0.0

    return sum(clipped_area(x, y) for x in triangles(a) for y in triangles(b))


def inside(outer, inner):
    return all(contains(outer, p) for p in inner) and abs(overlap(outer, inner)-area(inner)) < 2e-6


def interpolate(a, b, t):
    return (np.asarray(a)*(1-t) + np.asarray(b)*t).tolist()


def projection(a, b, point):
    a, b, point = np.asarray(a), np.asarray(b), np.asarray(point)
    return float(np.clip(np.dot(point-a, b-a)/np.dot(b-a, b-a), 0, 1))


def to_pixels(points, width, height):
    return np.asarray(points) * [width, height]
