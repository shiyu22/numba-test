import numpy
import time
from towhee import register, ops
from numba import njit


@register(name='inner_distance')
def inner_distance(query, data):
    dists = []
    for vec in data:
        dist = 0
        for i in range(len(vec)):
            dist += vec[i] * query[i]
        dists.append(dist)
    return dists


# 0.26415419578552246
# 0.0016889572143554688
if __name__ == '__main__':
    data = numpy.random.random((10000, 128))
    query = numpy.random.random(128)

    # op = njit(inner_distance, nogil=True)
    time1 = time.time()
    # op(query, data)

    dists = ops.inner_distance()(query, data)
    print(time.time() - time1)

    time1 = time.time()
    # op(query, data)

    dists = ops.inner_distance()(query, data)
    print(time.time() - time1)
