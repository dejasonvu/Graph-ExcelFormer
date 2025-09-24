# cython: language_level=3, boundscheck=False, wraparound=False, language=c++

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libc.math cimport pow
from cython.parallel import prange, threadid

cpdef object compute_mtam_knn(float[:, :] dist_matrix, int base_k, float threshold, int num_threads):
    cdef vector[int] neighbors
    cdef vector[pair[int, int]] edge_list
    cdef vector[vector[int]] adjacency = vector[vector[int]](dist_matrix.shape[0])
    cdef vector[vector[pair[int, int]]] thread_edge_list = vector[vector[pair[int, int]]](num_threads)
    cdef int i, j, num_samples, k, check, tid, z, left, right, temp, rec
    num_samples = dist_matrix.shape[0]

    for i in prange(num_samples, num_threads=num_threads, nogil=True):
        neighbors = vector[int]()
        for j in range(num_samples):
            if dist_matrix[i, j] <= threshold:
                neighbors.push_back(j)

        k = <int>(pow(base_k * neighbors.size(), 0.35))

        if k < neighbors.size():
            left = 0
            right = neighbors.size() - 1
            
            while left < right:
                rec = left
                for j in range(left, right):
                    if dist_matrix[i, neighbors[j]] < dist_matrix[i, neighbors[right]]:
                        temp = neighbors[j]
                        neighbors[j] = neighbors[rec]
                        neighbors[rec] = temp 
                        rec = rec + 1
                temp = neighbors[right]
                neighbors[right] = neighbors[rec]
                neighbors[rec] = temp
                
                if k == rec:
                    break
                elif k < rec:
                    right = rec - 1
                else:
                    left = rec + 1

            for j in range(k):
                adjacency[i].push_back(neighbors[j])
        else:
            adjacency[i] = neighbors

    for i in prange(num_samples, num_threads=num_threads, nogil=True):
        tid = threadid()
        for j in range(adjacency[i].size()):
            check = adjacency[i][j]
            for z in range(adjacency[check].size()):
                if adjacency[check][z] == i:
                    thread_edge_list[tid].push_back(pair[int, int](i, check))
                    break

    for i in range(num_threads):
        for j in range(thread_edge_list[i].size()):
            edge_list.push_back(thread_edge_list[i][j])
    return edge_list
