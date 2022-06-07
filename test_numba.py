from numba import njit
import time
import numpy
from towhee import register, ops
from towhee.operator import Operator


@njit()
def calEuclideanDistance(query, data):
    dists = []
    for vec in data:
        dist = numpy.sqrt(numpy.sum(numpy.square(query - vec)))
        dists.append(round(dist, 3))
    return dists


@njit()
def calInnerDistance(query, data):
    dists = []
    for vec in data:
        dist = 0
        for i in range(len(vec)):
            dist += vec[i] * query[i]
        dists.append(dist)
    return dists


@njit()
def get_topk(dists, topk):
    sorted_id = numpy.argsort(dists)
    return sorted(dists)[0:topk], sorted_id[0:topk]


@register(name='similarity_search_test', output_schema=['dis', 'ids'])
class SimilaritySearchTest(Operator):
    def __init__(self, data: list, cal: str = 'L2', topk: int = 5, name: str = 'feature_vector'):
        self.data = data
        self.name = name
        self.cal = cal
        self.topk = topk

    def __call__(self, query: list):
        try:
            query = getattr(query, self.name)
        except AttributeError:
            pass
        if self.cal == 'L2':
            dists = calEuclideanDistance(query, self.data)
        elif self.cal == 'IP':
            dists = calInnerDistance(query, self.data)
        dists = numpy.array(dists)
        dis, ids = get_topk(dists, self.topk)
        return dis, ids


# 3.808941125869751
# 0.008603811264038086
# 0.010421037673950195
if __name__ == '__main__':
    data = numpy.random.random((10000, 128))
    query = numpy.random.random(128)

    time1 = time.time()
    dists_l2 = calEuclideanDistance(query, data)
    dists_ip = calInnerDistance(query, data)
    dists_l2 = numpy.array(dists_l2)
    dists_ip = numpy.array(dists_ip)
    dis_l2, ids_l2 = get_topk(dists_l2, topk=10)
    dis_ip, ids_ip = get_topk(dists_ip, topk=10)
    print(time.time()-time1)

    time1 = time.time()
    dists_l2 = calEuclideanDistance(query, data)
    dists_ip = calInnerDistance(query, data)
    dists_l2 = numpy.array(dists_l2)
    dists_ip = numpy.array(dists_ip)
    dis_l2, ids_l2 = get_topk(dists_l2, topk=10)
    dis_ip, ids_ip = get_topk(dists_ip, topk=10)
    print(time.time() - time1)

    time1 = time.time()
    search = ops.similarity_search_test(data=data, cal='L2')
    distance, ids = search(query=query)
    search = ops.similarity_search_test(data=data, cal='IP')
    distance, ids = search(query=query)
    print(time.time() - time1)
