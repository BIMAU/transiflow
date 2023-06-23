from transiflow import CrsMatrix

import numpy
import pytest

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

    assert B[0, 0] == A[0, 1]
    assert B[0, 1] == A[0, 2]
    assert B[1, 0] == A[1, 1]
    assert B[1, 1] == A[1, 2]


def test_getitem_array():
    A = get_test_matrix()
    B = A[[0, 1], [0, 2]]

    assert B[0, 0] == A[0, 0]
    assert B[0, 1] == A[0, 2]
    assert B[1, 0] == A[1, 0]
    assert B[1, 1] == A[1, 2]

def test_setitem():
    A = get_test_matrix()

    B = CrsMatrix(m=A.m, n=A.n)
    B[0, 0] = 1
    B[0, 2] = 2
    B[1, 1] = 3
    B[2, 1] = 4
    B[2, 2] = 5
    B.assemble()

    for i in range(A.m):
        for j in range(A.n):
            assert A[i, j] == B[i, j]

def test_setitem_range():
    A = get_test_matrix()

    B = CrsMatrix(m=5, n=6)
    B[0:3, 1:4] = A
    B.assemble()

    assert A.begA[-1] == B.begA[-1]
    for i in range(A.m):
        for j in range(A.n):
            assert A[i, j] == B[i, j + 1]

def test_setitem_array():
    A = get_test_matrix()

    iidx = [1, 3, 4]
    jidx = [0, 2, 1]

    B = CrsMatrix(m=5, n=6)
    B[iidx, jidx] = A
    B.assemble()

    assert A.begA[-1] == B.begA[-1]
    for i in range(A.m):
        for j in range(A.n):
            assert A[i, j] == B[iidx[i], jidx[j]]

def test_setitem_to_array():
    A = get_test_matrix()
    A2 = A.to_dense()

    iidx = [1, 3, 4]
    jidx = [0, 2, 1]

    B = CrsMatrix(m=5, n=6)
    B[iidx, jidx] = A2
    B.assemble()

    assert A.begA[-1] == B.begA[-1]
    for i in range(A.m):
        for j in range(A.n):
            assert A[i, j] == B[iidx[i], jidx[j]]

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

def test_matvec():
    numpy.random.seed(1234)

    A = get_test_matrix()
    A2 = A.to_dense()

    x = numpy.random.random(A.n)
    b = A.matvec(x)
    b2 = numpy.matmul(A2, x)

    assert b == pytest.approx(b2)

    x = numpy.random.random((A.n, 4))
    b = A.matvec(x)
    b2 = numpy.matmul(A2, x)

    assert b == pytest.approx(b2)

def test_matmul():
    numpy.random.seed(1234)

    A = get_test_matrix()
    A2 = A.to_dense()

    x = numpy.random.random(A.n)
    b = A @ x
    b2 = A2 @ x

    assert b == pytest.approx(b2)

    x = numpy.random.random((A.n, 4))
    b = A @ x
    b2 = A2 @ x

    assert b == pytest.approx(b2)

def test_transpose():
    A = get_test_matrix()
    B = A.transpose()

    for i in range(A.m):
        for j in range(A.n):
            assert B[i, j] == A[j, i]

def test_add():
    A = get_test_matrix()
    B = A.to_dense()

    A2 = A + A
    B2 = B + B

    for i in range(A.m):
        for j in range(A.n):
            assert A2[i, j] == B2[i, j]
            assert A2[i, j] == A[i, j] + A[i, j]

def test_iadd():
    A = get_test_matrix()
    A2 = get_test_matrix()
    B = A.to_dense()

    A2 += A
    B2 = B + B

    for i in range(A.m):
        for j in range(A.n):
            assert A2[i, j] == B2[i, j]
            assert A2[i, j] == A[i, j] + A[i, j]

def test_sub():
    A = get_test_matrix()
    B = A.to_dense()

    A2 = A - A.transpose()
    B2 = B - B.T

    for i in range(A.m):
        for j in range(A.n):
            assert A2[i, j] == B2[i, j]
            assert A2[i, j] == A[i, j] - A[j, i]

def test_isub():
    A = get_test_matrix()
    A2 = A.transpose()
    B = A.to_dense()

    A2 -= A
    B2 = B.T - B

    for i in range(A.m):
        for j in range(A.n):
            assert A2[i, j] == B2[i, j]
            assert A2[i, j] == A[j, i] - A[i, j]
