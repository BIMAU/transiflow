import os

import numpy
import pytest

from fvm import Continuation, utils


def read_matrix(fname):
    from petsc4py import PETSc

    dirname = os.path.dirname(__file__)
    rows = []
    cols = []
    vals = []
    with open(os.path.join(dirname, fname), "r") as f:
        for i in f.readlines():
            r, c, v = [j.strip() for j in i.strip().split(" ") if j]
            r = int(r) - 1
            c = int(c) - 1
            v = float(v)
            rows.append(r)
            cols.append(c)
            vals.append(v)

    size = max(rows) + 1
    A = PETSc.Mat().createAIJ((size, size))
    A.setUp()

    for r, c, v in zip(rows, cols, vals):
        A.setValue(r, c, v)
    A.assemble()

    return A


def read_vector(fname):
    from fvm.interface import PETSc as PETScInterface

    values = []
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), "r") as f:
        for i, v in enumerate(f.readlines()):
            values.append(float(v.strip()))

    vec = PETScInterface.Vector.from_array(values)

    return vec


def extract_sorted_row(A, i):
    indices, values = A.getRow(i)
    idx = sorted(range(len(indices)), key=lambda i: indices[i])
    return [indices[i] for i in idx], [values[i] for i in idx]


def test_ldc():
    try:
        from fvm.interface import PETSc as PETScInterface
    except ImportError:
        pytest.skip("PETSc not found")

    nx = 4
    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {"Reynolds Number": 100}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i + 1

    interface = PETScInterface.Interface(parameters, nx, ny, nz, dim, dof)

    state = PETScInterface.Vector.from_array(state)

    A = interface.jacobian(state)
    rhs = interface.rhs(state)

    B = read_matrix("ldc_%sx%sx%s.txt" % (nx, ny, nz))
    rhs_B = read_vector("ldc_rhs_%sx%sx%s.txt" % (nx, ny, nz))

    for i in range(n):
        indices_A, values_A = extract_sorted_row(A, i)
        indices_B, values_B = extract_sorted_row(B, i)

        print("Expected:")
        print(indices_B)
        print(values_B)

        print("Got:")
        print(indices_A)
        print(values_A)

        assert len(indices_A) == len(indices_B)
        for j in range(len(indices_A)):
            assert indices_A[j] == indices_B[j]
            assert values_A[j] == pytest.approx(values_B[j])

    for i in range(n):
        print(i, rhs[i], rhs_B[i])
        assert rhs_B[i] == pytest.approx(rhs[i])


def test_norm():
    try:
        from fvm.interface import PETSc as PETScInterface
    except ImportError:
        pytest.skip("PETSc not found")

    nx = 4
    ny = nx
    nz = nx
    dof = 4
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i + 1

    state_dist = PETScInterface.Vector.from_array(state)
    assert utils.norm(state) == utils.norm(state_dist)


def test_PETSc(nx=4):
    try:
        from fvm.interface import PETSc as PETScInterface
    except ImportError:
        pytest.skip("PETSc not found")

    numpy.random.seed(1234)

    dim = 3
    dof = 4
    ny = nx
    nz = nx
    parameters = {"Reynolds Number": 0}

    interface = PETScInterface.Interface(parameters, nx, ny, nz, dim, dof)

    continuation = Continuation(interface, parameters)

    n = nx * ny * nz * dof
    x0 = PETScInterface.Vector().createMPI(n)
    x0 = continuation.newton(x0)

    start = 0
    target = 2000
    ds = 100
    x = continuation.continuation(x0, "Reynolds Number", start, target, ds)[0]

    assert utils.norm(x) > 0
