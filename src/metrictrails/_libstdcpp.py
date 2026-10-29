"""std::sort and std::nth_element as implemented in libstdc++ (GCC).

GEOS packs its STRtree with these algorithms, which are not stable: the
order of nodes with equal keys depends on the exact sequence of swaps. They
are reproduced here so that ties are broken as in R (sf/GEOS on Linux).
All functions work in place on a list `a` over [first, last) with a
strict-weak-order `less(x, y)`.
"""

THRESHOLD = 16


def _lg(n):
    return n.bit_length() - 1


def _adjust_heap(a, first, hole, length, value, less):
    top = hole
    second = hole
    while second < (length - 1) // 2:
        second = 2 * (second + 1)
        if less(a[first + second], a[first + second - 1]):
            second -= 1
        a[first + hole] = a[first + second]
        hole = second
    if (length & 1) == 0 and second == (length - 2) // 2:
        second = 2 * (second + 1)
        a[first + hole] = a[first + second - 1]
        hole = second - 1
    parent = (hole - 1) // 2
    while hole > top and less(a[first + parent], value):
        a[first + hole] = a[first + parent]
        hole = parent
        parent = (hole - 1) // 2
    a[first + hole] = value


def _make_heap(a, first, last, less):
    length = last - first
    if length < 2:
        return
    parent = (length - 2) // 2
    while True:
        _adjust_heap(a, first, parent, length, a[first + parent], less)
        if parent == 0:
            return
        parent -= 1


def _pop_heap(a, first, last, result, less):
    value = a[result]
    a[result] = a[first]
    _adjust_heap(a, first, 0, last - first, value, less)


def _heap_select(a, first, middle, last, less):
    _make_heap(a, first, middle, less)
    for i in range(middle, last):
        if less(a[i], a[first]):
            _pop_heap(a, first, middle, i, less)


def _sort_heap(a, first, last, less):
    while last - first > 1:
        last -= 1
        _pop_heap(a, first, last, last, less)


def _move_median_to_first(a, result, x, y, z, less):
    if less(a[x], a[y]):
        if less(a[y], a[z]):
            a[result], a[y] = a[y], a[result]
        elif less(a[x], a[z]):
            a[result], a[z] = a[z], a[result]
        else:
            a[result], a[x] = a[x], a[result]
    elif less(a[x], a[z]):
        a[result], a[x] = a[x], a[result]
    elif less(a[y], a[z]):
        a[result], a[z] = a[z], a[result]
    else:
        a[result], a[y] = a[y], a[result]


def _unguarded_partition(a, first, last, pivot, less):
    while True:
        while less(a[first], a[pivot]):
            first += 1
        last -= 1
        while less(a[pivot], a[last]):
            last -= 1
        if not first < last:
            return first
        a[first], a[last] = a[last], a[first]
        first += 1


def _unguarded_partition_pivot(a, first, last, less):
    mid = first + (last - first) // 2
    _move_median_to_first(a, first, first + 1, mid, last - 1, less)
    return _unguarded_partition(a, first + 1, last, first, less)


def _unguarded_linear_insert(a, last, less):
    val = a[last]
    nxt = last - 1
    while less(val, a[nxt]):
        a[last] = a[nxt]
        last = nxt
        nxt -= 1
    a[last] = val


def _insertion_sort(a, first, last, less):
    if first == last:
        return
    for i in range(first + 1, last):
        if less(a[i], a[first]):
            val = a[i]
            a[first + 1:i + 1] = a[first:i]
            a[first] = val
        else:
            _unguarded_linear_insert(a, i, less)


def _introsort_loop(a, first, last, depth, less):
    while last - first > THRESHOLD:
        if depth == 0:
            _heap_select(a, first, last, last, less)
            _sort_heap(a, first, last, less)
            return
        depth -= 1
        cut = _unguarded_partition_pivot(a, first, last, less)
        _introsort_loop(a, cut, last, depth, less)
        last = cut


def sort(a, first, last, less):
    """std::sort(a + first, a + last, less)."""
    if first == last:
        return
    _introsort_loop(a, first, last, _lg(last - first) * 2, less)
    if last - first > THRESHOLD:
        _insertion_sort(a, first, first + THRESHOLD, less)
        for i in range(first + THRESHOLD, last):
            _unguarded_linear_insert(a, i, less)
    else:
        _insertion_sort(a, first, last, less)


def nth_element(a, first, nth, last, less):
    """std::nth_element(a + first, a + nth, a + last, less)."""
    if first == last or nth == last:
        return
    depth = _lg(last - first) * 2
    while last - first > 3:
        if depth == 0:
            _heap_select(a, first, nth + 1, last, less)
            a[first], a[nth] = a[nth], a[first]
            return
        depth -= 1
        cut = _unguarded_partition_pivot(a, first, last, less)
        if cut <= nth:
            first = cut
        else:
            last = cut
    _insertion_sort(a, first, last, less)
