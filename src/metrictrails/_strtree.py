"""Nearest segment as returned by sf::st_nearest_feature (GEOS STRtree).

When a point is exactly equidistant to several segments (on the outer side
of a vertex, or where a loop passes twice), GEOS returns the first leaf
popped from its priority queue. Which one that is depends on how the
sort-tile-recursive tree is packed (node capacity 10, as in sf) and on the
tie order of the C++ standard library heap (libstdc++). Both are reproduced
here so that the port picks the same segment as R.
"""

import math

import numpy as np

from . import _libstdcpp as stl

NODE_CAPACITY = 10


class _Node:
    __slots__ = ("minx", "miny", "maxx", "maxy", "children", "item")

    def __init__(self, minx, miny, maxx, maxy, children=None, item=-1):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.children = children
        self.item = item


def _parent(nodes):
    return _Node(min(n.minx for n in nodes), min(n.miny for n in nodes),
                 max(n.maxx for n in nodes), max(n.maxy for n in nodes), children=nodes)


def _less_x(a, b):
    return a.minx + a.maxx < b.minx + b.maxx


def _less_y(a, b):
    return a.miny + a.maxy < b.miny + b.maxy


def _build(leaves):
    """Sort-tile-recursive packing as in GEOS TemplateSTRtree (libstdc++ sorts)."""
    level = list(leaves)
    while len(level) > 1:
        n = len(level)
        n_slices = int(math.ceil(math.sqrt(math.ceil(n / NODE_CAPACITY))))
        per_slice = int(math.ceil(n / n_slices))
        stl.sort(level, 0, n, _less_x)
        parents = []
        start = 0
        for _ in range(n_slices):
            end_slice = start + min(n - start, per_slice)
            stl.sort(level, start, end_slice, _less_y)
            for c in range(start, end_slice, NODE_CAPACITY):
                parents.append(_parent(level[c:min(c + NODE_CAPACITY, end_slice)]))
            start = end_slice
        level = parents
    return level[0]


def _env_distance(a, px, py):
    """GEOS Envelope::distance between a node envelope and a point."""
    if a.minx <= px <= a.maxx and a.miny <= py <= a.maxy:
        return 0.0
    dx = 0.0
    if a.maxx < px:
        dx = px - a.maxx
    elif a.minx > px:
        dx = a.minx - px
    dy = 0.0
    if a.maxy < py:
        dy = py - a.maxy
    elif a.miny > py:
        dy = a.miny - py
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return math.sqrt(dx * dx + dy * dy)


def point_segment_distance(px, py, ax, ay, bx, by):
    """GEOS Distance::pointToSegment."""
    if ax == bx and ay == by:
        return math.sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay))
    len2 = (bx - ax) * (bx - ax) + (by - ay) * (by - ay)
    r = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / len2
    if r <= 0.0:
        return math.sqrt((px - ax) * (px - ax) + (py - ay) * (py - ay))
    if r >= 1.0:
        return math.sqrt((px - bx) * (px - bx) + (py - by) * (py - by))
    s = ((ay - py) * (bx - ax) - (ax - px) * (by - ay)) / len2
    return abs(s) * math.sqrt(len2)


# ---- libstdc++ std::priority_queue with comp(a, b) = a.distance > b.distance ----

def _heap_push(h, v):
    h.append(v)
    hole = len(h) - 1
    parent = (hole - 1) // 2
    while hole > 0 and h[parent][0] > v[0]:
        h[hole] = h[parent]
        hole = parent
        parent = (hole - 1) // 2
    h[hole] = v


def _heap_pop(h):
    last = len(h) - 1
    if last > 0:
        value = h[last]
        h[last] = h[0]
        length = last
        hole = 0
        second = 0
        while second < (length - 1) // 2:
            second = 2 * (second + 1)
            if h[second][0] > h[second - 1][0]:
                second -= 1
            h[hole] = h[second]
            hole = second
        if (length & 1) == 0 and second == (length - 2) // 2:
            second = 2 * (second + 1)
            h[hole] = h[second - 1]
            hole = second - 1
        parent = (hole - 1) // 2
        while hole > 0 and h[parent][0] > value[0]:
            h[hole] = h[parent]
            hole = parent
            parent = (hole - 1) // 2
        h[hole] = value
    h.pop()


class SegmentIndex:
    """STRtree of the segments of a polyline, queried like sf::st_nearest_feature."""

    def __init__(self, rx, ry):
        self.rx = np.asarray(rx, dtype=float)
        self.ry = np.asarray(ry, dtype=float)
        leaves = []
        for i in range(len(rx) - 1):
            ax, ay, bx, by = self.rx[i], self.ry[i], self.rx[i + 1], self.ry[i + 1]
            leaves.append(_Node(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by), item=i))
        self.root = _build(leaves)

    def _leaf_distance(self, node, px, py):
        i = node.item
        return point_segment_distance(px, py, self.rx[i], self.ry[i], self.rx[i + 1], self.ry[i + 1])

    def nearest(self, px, py):
        """0-based index of the nearest segment of point (px, py)."""
        def dist(node):
            if node.children is None:
                return self._leaf_distance(node, px, py)
            return _env_distance(node, px, py)

        heap = []
        _heap_push(heap, (dist(self.root), self.root))
        lower = math.inf
        best = None
        while heap and lower > 0:
            d, node = heap[0]
            if best is not None and d >= lower:
                break
            _heap_pop(heap)
            if node.children is None:
                lower = d
                best = node
            else:
                for child in node.children:
                    cd = dist(child)
                    if cd < lower:
                        _heap_push(heap, (cd, child))
        return best.item


class PointIndex:
    """STRtree of points, queried like sf::st_nearest_feature on points."""

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        leaves = [_Node(self.x[i], self.y[i], self.x[i], self.y[i], item=i) for i in range(self.x.size)]
        self.root = _build(leaves)

    def nearest(self, px, py):
        """0-based index of the nearest point to (px, py)."""
        def dist(node):
            if node.children is None:
                dx = px - self.x[node.item]
                dy = py - self.y[node.item]
                return math.sqrt(dx * dx + dy * dy)
            return _env_distance(node, px, py)

        heap = []
        _heap_push(heap, (dist(self.root), self.root))
        lower = math.inf
        best = None
        while heap and lower > 0:
            d, node = heap[0]
            if best is not None and d >= lower:
                break
            _heap_pop(heap)
            if node.children is None:
                lower = d
                best = node
            else:
                for child in node.children:
                    cd = dist(child)
                    if cd < lower:
                        _heap_push(heap, (cd, child))
        return best.item
