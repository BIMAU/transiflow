from fvm import CrsMatrix

import numpy

def get_test_matrix():
    coA = numpy.array([1, 2, 3, 4, 5], dtype=int)
    jcoA = numpy.array([0, 2, 1, 1, 2], dtype=int)
    begA = numpy.array([0, 2, 3, 5], dtype=int)
    return CrsMatrix(coA, jcoA, begA)

def get_random_test_matrix():
    m = 5
    n = 3
    coA = numpy.array(numpy.random.random(m * n), dtype=int)
    jcoA = numpy.repeat(numpy.array([numpy.arange(n)]), m)
    begA = numpy.arange(m * n + 1, step=n)
    return CrsMatrix(coA, jcoA, begA, m=m, n=n)

def test_getitem():
    A = get_test_matrix()
    assert A[0, 1] == 0
    assert A[1, 1] == 3

def test_getitem_range():
    A = get_test_matrix()
    B = A[0:2, 1:3]
    print(str(B))
    assert B[0, 0] == A[0, 1]
    assert B[0, 1] == A[0, 2]
    assert B[1, 0] == A[1, 1]
    assert B[1, 1] == A[1, 2]


def test_getitem_array():
    A = get_test_matrix()
    B = A[[0, 1], [0, 2]]
    print(str(B))
    assert B[0, 0] == A[0, 0]
    assert B[0, 1] == A[0, 2]
    assert B[1, 0] == A[1, 0]
    assert B[1, 1] == A[1, 2]

def test_to_dense():
    A = get_test_matrix()
    B = A.to_dense()
    for i in range(A.m):
        for j in range(A.n):
            assert A[i, j] == B[i, j]

    A = get_random_test_matrix()
    B = A.coA.copy().reshape(A.m, A.n)
    for i in range(A.m):
        for j in range(A.n):
            assert A[i, j] == B[i, j]
