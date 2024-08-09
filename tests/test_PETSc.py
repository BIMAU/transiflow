import os

import numpy
import pytest

from transiflow import Continuation, Discretization, utils


def read_matrix(fname, ao):
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

    for r, c, v in zip(ao.app2petsc(rows), ao.app2petsc(cols), vals):
        A.setValue(r, c, v)
    A.assemble()

    return A


def read_vector(fname, m, ao):
    from petsc4py import PETSc

    from transiflow.interface import PETSc as PETScInterface

    values = []
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), "r") as f:
        for v in f.readlines():
            values.append(float(v.strip()))

    indices_nat = ao.petsc2app(m.indices)
    vec = PETScInterface.Vector.from_array(
        m, numpy.array(values, dtype=PETSc.ScalarType)[indices_nat]
    )
    return vec


def extract_sorted_row(A, i):
    if i in range(*A.getOwnershipRange()):
        indices, values = A.getRow(i)
        idx = sorted(range(len(indices)), key=lambda i: indices[i])
        return [indices[i] for i in idx], [values[i] for i in idx]
    else:
        return [], []


def extract_sorted_local_row(A, i, ao):
    indices = A.jcoA[A.begA[i]: A.begA[i + 1]]
    values = A.coA[A.begA[i]: A.begA[i + 1]]
    indices = ao.app2petsc(indices.astype(numpy.int32))
    idx = sorted(range(len(indices)), key=lambda i: indices[i])
    return [indices[i] for i in idx], [values[i] for i in idx]


def test_ldc():
    try:
        from transiflow.interface import PETSc as PETScInterface
    except ImportError:
        pytest.skip("PETSc not found")

    nx = 4
    ny = nx
    nz = nx
    dof = 4
    parameters = {"Reynolds Number": 100}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i + 1

    interface = PETScInterface.Interface(parameters, nx, ny, nz, dof=dof)
    state = interface.vector_from_array(state)

    for i in interface.map.indices:
        i_nat = interface.index_ordering.petsc2app(i)
        print(i, i_nat, state[i] - 1)
        assert i_nat == state[i] - 1

    A = interface.jacobian(state)
    rhs = interface.rhs(state)

    B = read_matrix("data/ldc_%sx%sx%s.txt" % (nx, ny, nz), interface.index_ordering_assembly)
    rhs_B = read_vector(
        "data/ldc_rhs_%sx%sx%s.txt" % (nx, ny, nz), interface.map, interface.index_ordering
    )

    for i in range(n):
        if i not in interface.map.indices:
            continue

        indices_A, values_A = extract_sorted_row(A, i)
        indices_B, values_B = extract_sorted_row(B, i)

        print("Expected:")
        print(indices_B)
        print(values_B)

        print("Got:")
        print(indices_A)
        print(values_A)

        assert len(indices_A) == len(indices_B) > 0

        for j in range(len(indices_A)):
            assert indices_A[j] == indices_B[j]
            assert values_A[j] == pytest.approx(values_B[j])

    for i in range(n):
        if i not in interface.map.indices:
            continue

        print(i, rhs[i], rhs_B[i])
        assert rhs_B[i] == pytest.approx(rhs[i])


def test_ldc_stretched_file():
    try:
        from petsc4py import PETSc

        from transiflow.interface import PETSc as PETScInterface
    except ImportError:
        pytest.skip("PETSc not found")

    nx = 4
    ny = nx
    nz = nx
    dof = 4
    parameters = {"Reynolds Number": 100, "Grid Stretching": True}
    n = nx * ny * nz * dof

    state = numpy.zeros(n, dtype=PETSc.ScalarType)
    for i in range(n):
        state[i] = i + 1

    interface = PETScInterface.Interface(parameters, nx, ny, nz, dof=dof)
    state = interface.vector_from_array(state)

    A = interface.jacobian(state)
    rhs = interface.rhs(state)

    B = read_matrix(
        "data/ldc_stretched_%sx%sx%s.txt" % (nx, ny, nz), interface.index_ordering_assembly
    )
    rhs_B = read_vector(
        "data/ldc_stretched_rhs_%sx%sx%s.txt" % (nx, ny, nz), interface.map, interface.index_ordering
    )

    for i in range(n):
        if i not in interface.map.indices:
            continue

        indices_A, values_A = extract_sorted_row(A, i)
        indices_B, values_B = extract_sorted_row(B, i)

        print("Expected:")
        print(indices_B)
        print(values_B)

        print("Got:")
        print(indices_A)
        print(values_A)

        assert len(indices_A) == len(indices_B) > 0

        for j in range(len(indices_A)):
            assert indices_A[j] == indices_B[j]
            assert values_A[j] == pytest.approx(values_B[j])

    for i in range(n):
        if i not in interface.map.indices:
            continue

        print(i, rhs[i], rhs_B[i])
        assert rhs_B[i] == pytest.approx(rhs[i])


def test_ldc_stretched(nx=4):
    try:
        from petsc4py import PETSc

        from transiflow.interface import PETSc as PETScInterface
    except ImportError:
        pytest.skip("PETSc not found")

    ny = nx
    nz = nx
    dof = 4
    parameters = {"Reynolds Number": 100, "Grid Stretching": True}
    n = nx * ny * nz * dof

    state = numpy.zeros(n, dtype=PETSc.ScalarType)
    for i in range(n):
        state[i] = i + 1

    discretization = Discretization(parameters, nx, ny, nz, dof=dof)
    B = discretization.jacobian(state)
    rhs_B = discretization.rhs(state)

    interface = PETScInterface.Interface(parameters, nx, ny, nz, dof=dof)

    state_numpy = state.copy()

    state = interface.vector_from_array(state)

    for i in interface.map.indices:
        i_nat = interface.index_ordering.petsc2app(i)
        print(i, i_nat, state[i] - 1)
        assert i_nat == state[i] - 1

    assert numpy.allclose(state.array, state_numpy[interface.map_natural.indices])

    for i in range(n):
        if i not in interface.map.indices:
            continue

        i_nat = interface.index_ordering.petsc2app(i)
        print(i, state[i], i_nat, state_numpy[i_nat])
        assert state_numpy[i_nat] == pytest.approx(state[i])

    # check ghosts mapping and assembly index ordering
    with state.localForm() as lf:
        for i, ghost in enumerate(interface.ghosts):
            ghost_nat = interface.index_ordering_assembly.petsc2app(ghost)
            print(i, ghost, lf.array[state.local_size + i])
            print(i, ghost_nat, state_numpy[ghost_nat])
            assert state_numpy[ghost_nat] == lf.array[state.local_size + i]

    A = interface.jacobian(state)
    rhs = interface.rhs(state)

    for i in range(n):
        if i not in interface.map.indices:
            continue

        i_nat = interface.index_ordering.petsc2app(i)
        print(i, rhs[i], i_nat, rhs_B[i_nat])
        assert rhs_B[i_nat] == pytest.approx(rhs[i])

    for i in range(n):
        if i not in interface.map.indices:
            continue

        indices_A, values_A = extract_sorted_row(A, i)

        i_nat = interface.index_ordering_assembly.petsc2app(i)

        # the assembly & solve map index orderings map i to the same PETSc index
        j_nat = interface.index_ordering.petsc2app(i)
        print(i, i_nat, j_nat)
        assert i_nat == j_nat

        indices_B, values_B = extract_sorted_local_row(
            B, i_nat, interface.index_ordering_assembly
        )

        print("Expected:")
        print(indices_B)
        print(values_B)

        print("Got:")
        print(indices_A)
        print(values_A)

        assert len(indices_A) == len(indices_B) > 0

        for j in range(len(indices_A)):
            assert indices_A[j] == indices_B[j]
            assert values_A[j] == pytest.approx(values_B[j])


def test_ldc8_stretched():
    test_ldc_stretched(8)


def test_norm():
    try:
        from transiflow.interface import PETSc as PETScInterface
    except ImportError:
        pytest.skip("PETSc not found")

    nx = 4
    ny = nx
    nz = nx
    dof = 4
    n = nx * ny * nz * dof
    parameters = {}

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i + 1

    interface = PETScInterface.Interface(parameters, nx, ny, nz, dof=dof)

    state_dist = PETScInterface.Vector.from_array(
        interface.map, state[interface.map_natural.indices]
    )
    state_dist2 = interface.vector_from_array(state)

    assert utils.norm(state) == utils.norm(state_dist)
    assert utils.norm(state) == utils.norm(state_dist2)


def test_PETSc(nx=4):
    try:
        from transiflow.interface import PETSc as PETScInterface
    except ImportError:
        pytest.skip("PETSc not found")

    numpy.random.seed(1234)

    ny = nx
    nz = nx
    parameters = {"Reynolds Number": 0, "Verbose": True}

    interface = PETScInterface.Interface(parameters, nx, ny, nz)

    continuation = Continuation(interface)

    x0 = interface.vector()
    x0 = continuation.newton(x0)

    start = 0
    target = 2000
    ds = 100
    x = continuation.continuation(x0, "Reynolds Number", start, target, ds)[0]

    assert utils.norm(x) > 0

def test_vector(nx=8):
    try:
        from transiflow.interface import PETSc as PETScInterface
    except ImportError:
        pytest.skip("PETSc not found")

    ny = nx
    nz = nx
    dim = 3
    dof = 4
    parameters = {"Reynolds Number": 100, "Grid Stretching": True}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i + 1

    print(state)

    interface = PETScInterface.Interface(parameters, nx, ny, nz, dim, dof)
    state_vec = interface.vector_from_array(state)
    state_gathered = interface.array_from_vector(state_vec)

    print(state_gathered)

    assert state == pytest.approx(state_gathered)
