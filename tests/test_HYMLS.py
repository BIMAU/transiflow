import pytest
import numpy
import os

from transiflow import Continuation
from transiflow import Discretization
from transiflow import utils


def read_matrix(fname, m):
    from PyTrilinos import Epetra

    A = Epetra.CrsMatrix(Epetra.Copy, m, 27)

    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'r') as f:
        for i in f.readlines():
            r, c, v = [j.strip() for j in i.strip().split(' ') if j]
            r = int(r) - 1
            c = int(c) - 1
            v = float(v)
            if A.MyGlobalRow(r):
                A[r, c] = v
        A.FillComplete()

    return A

def read_vector(fname, m):
    from transiflow.interface import HYMLS as HYMLSInterface

    vec = HYMLSInterface.Vector(m)

    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'r') as f:
        for i, v in enumerate(f.readlines()):
            lid = m.LID(i)
            if lid != -1:
                vec[lid] = float(v.strip())
    return vec

def write_vector(vec, fname):
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'w') as f:
        for i in range(len(vec)):
            f.write('%.16e\n' % vec[i])

def read_value(fname):
    val = 0
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'r') as f:
        for v in f.readlines():
            val = float(v.strip())
    return val

def write_value(val, fname):
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, fname), 'w') as f:
        f.write('%.16e\n' % val)

def extract_sorted_row(A, i):
    values, indices = A.ExtractGlobalRowCopy(i)
    idx = sorted(range(len(indices)), key=lambda i: indices[i])
    return [indices[i] for i in idx], [values[i] for i in idx]

def extract_sorted_local_row(A, i):
    indices = A.jcoA[A.begA[i]:A.begA[i+1]]
    values = A.coA[A.begA[i]:A.begA[i+1]]
    idx = sorted(range(len(indices)), key=lambda i: indices[i])
    return [indices[i] for i in idx], [values[i] for i in idx]

def test_ldc():
    try:
        from transiflow.interface import HYMLS as HYMLSInterface
    except ImportError:
        pytest.skip("HYMLS not found")

    nx = 4
    ny = nx
    nz = nx
    dof = 4
    parameters = {'Reynolds Number': 100}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    interface = HYMLSInterface.Interface(parameters, nx, ny, nz, dof=dof)
    state = interface.vector_from_array(state)

    A = interface.jacobian(state)
    rhs = interface.rhs(state)

    B = read_matrix('data/ldc_%sx%sx%s.txt' % (nx, ny, nz), interface.solve_map)
    rhs_B = read_vector('data/ldc_rhs_%sx%sx%s.txt' % (nx, ny, nz), interface.map)

    for i in range(n):
        lid = interface.solve_map.LID(i)
        if lid == -1:
            continue

        print(i, lid)

        indices_A, values_A = extract_sorted_row(A, i)
        indices_B, values_B = extract_sorted_row(B, i)

        print('Expected:')
        print(indices_B)
        print(values_B)

        print('Got:')
        print(indices_A)
        print(values_A)

        assert len(indices_A) == len(indices_B)
        for j in range(len(indices_A)):
            assert indices_A[j] == indices_B[j]
            assert values_A[j] == pytest.approx(values_B[j])

    for i in range(n):
        lid = interface.map.LID(i)
        if lid == -1:
            continue

        print(i, lid, rhs[lid], rhs_B[lid])

        assert rhs_B[lid] == pytest.approx(rhs[lid])

def test_ldc_stretched_file():
    try:
        from transiflow.interface import HYMLS as HYMLSInterface
    except ImportError:
        pytest.skip("HYMLS not found")

    nx = 4
    ny = nx
    nz = nx
    dof = 4
    parameters = {'Reynolds Number': 100, 'Grid Stretching': True}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    interface = HYMLSInterface.Interface(parameters, nx, ny, nz, dof=dof)
    state = interface.vector_from_array(state)

    A = interface.jacobian(state)
    rhs = interface.rhs(state)

    B = read_matrix('data/ldc_stretched_%sx%sx%s.txt' % (nx, ny, nz), interface.solve_map)
    rhs_B = read_vector('data/ldc_stretched_rhs_%sx%sx%s.txt' % (nx, ny, nz), interface.map)

    for i in range(n):
        lid = interface.solve_map.LID(i)
        if lid == -1:
            continue

        print(i, lid)

        indices_A, values_A = extract_sorted_row(A, i)
        indices_B, values_B = extract_sorted_row(B, i)

        print('Expected:')
        print(indices_B)
        print(values_B)

        print('Got:')
        print(indices_A)
        print(values_A)

        assert len(indices_A) == len(indices_B)
        for j in range(len(indices_A)):
            assert indices_A[j] == indices_B[j]
            assert values_A[j] == pytest.approx(values_B[j])

    for i in range(n):
        lid = interface.map.LID(i)
        if lid == -1:
            continue

        print(i, lid, rhs[lid], rhs_B[lid])

        assert rhs_B[lid] == pytest.approx(rhs[lid])

def test_ldc_stretched(nx=4):
    try:
        from transiflow.interface import HYMLS as HYMLSInterface
    except ImportError:
        pytest.skip("HYMLS not found")

    ny = nx
    nz = nx
    dof = 4
    parameters = {'Reynolds Number': 100, 'Grid Stretching': True}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    discretization = Discretization(parameters, nx, ny, nz, dof=dof)
    B = discretization.jacobian(state)
    rhs_B = discretization.rhs(state)

    interface = HYMLSInterface.Interface(parameters, nx, ny, nz, dof=dof)
    state = interface.vector_from_array(state)

    A = interface.jacobian(state)
    rhs = interface.rhs(state)

    for i in range(n):
        lid = interface.solve_map.LID(i)
        if lid == -1:
            continue

        print(i, lid)

        indices_A, values_A = extract_sorted_row(A, i)
        indices_B, values_B = extract_sorted_local_row(B, i)

        print('Expected:')
        print(indices_B)
        print(values_B)

        print('Got:')
        print(indices_A)
        print(values_A)

        assert len(indices_A) == len(indices_B)
        for j in range(len(indices_A)):
            assert indices_A[j] == indices_B[j]
            assert values_A[j] == pytest.approx(values_B[j])

    for i in range(n):
        lid = interface.map.LID(i)
        if lid == -1:
            continue

        print(i, lid, rhs[lid], rhs_B[lid])

        assert rhs_B[i] == pytest.approx(rhs[lid])

def test_ldc8_stretched():
    test_ldc_stretched(8)

def test_prec(nx=4, parameters=None):
    try:
        from transiflow.interface import HYMLS as HYMLSInterface
        from PyTrilinos import Epetra
    except ImportError:
        pytest.skip("HYMLS not found")

    ny = nx
    nz = nx
    dof = 4
    n = nx * ny * nz * dof

    if not parameters:
        parameters = {'Reynolds Number': 100, 'Preconditioner': {'Number of Levels': 0}}

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    interface = HYMLSInterface.Interface(parameters, nx, ny, nz, dof=dof)
    state = interface.vector_from_array(state)

    if interface.comm.NumProc() > 1:
        if interface.comm.MyPID() == 0:
            pytest.skip("The preconditioner won't be exactly the same on multiple cores")
        return

    interface.jacobian(state)
    rhs = interface.rhs(state)

    rhs_sol = HYMLSInterface.Vector(interface.solve_map)
    prec_rhs = HYMLSInterface.Vector(interface.solve_map)
    rhs_sol.Import(rhs, interface.solve_importer, Epetra.Insert)

    interface.preconditioner.Compute()
    interface.preconditioner.ApplyInverse(rhs_sol, prec_rhs)

    # write_vector(prec_rhs, 'data/ldc_prec_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    prec_rhs_B = read_vector('data/ldc_prec_rhs_%sx%sx%s.txt' % (nx, ny, nz), interface.solve_map)

    for i in range(n):
        lid = interface.solve_map.LID(i)
        if lid == -1:
            continue

        print(i, lid, prec_rhs[lid], prec_rhs_B[lid])

        assert prec_rhs_B[lid] == pytest.approx(prec_rhs[lid])

def test_multilevel_prec():
    parameters = {'Reynolds Number': 100, 'Preconditioner': {'Number of Levels': 2, 'Separator Length': 4}}
    test_prec(8, parameters)

def test_prec_stretched(nx=4, parameters=None):
    try:
        from transiflow.interface import HYMLS as HYMLSInterface
        from PyTrilinos import Epetra
    except ImportError:
        pytest.skip("HYMLS not found")

    ny = nx
    nz = nx
    dof = 4
    n = nx * ny * nz * dof

    if not parameters:
        parameters = {'Reynolds Number': 100, 'Grid Stretching': True, 'Preconditioner': {'Number of Levels': 0}}

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    interface = HYMLSInterface.Interface(parameters, nx, ny, nz, dof=dof)
    state = interface.vector_from_array(state)

    if interface.comm.NumProc() > 1:
        if interface.comm.MyPID() == 0:
            pytest.skip("The preconditioner won't be exactly the same on multiple cores")
        return

    interface.jacobian(state)
    rhs = interface.rhs(state)

    rhs_sol = HYMLSInterface.Vector(interface.solve_map)
    prec_rhs = HYMLSInterface.Vector(interface.solve_map)
    rhs_sol.Import(rhs, interface.solve_importer, Epetra.Insert)

    interface.preconditioner.Compute()
    interface.preconditioner.ApplyInverse(rhs_sol, prec_rhs)

    # write_vector(prec_rhs, 'data/ldc_stretched_prec_rhs_%sx%sx%s.txt' % (nx, ny, nz))

    prec_rhs_B = read_vector('data/ldc_stretched_prec_rhs_%sx%sx%s.txt' % (nx, ny, nz), interface.solve_map)

    for i in range(n):
        lid = interface.solve_map.LID(i)
        if lid == -1:
            continue

        print(i, lid, prec_rhs[lid], prec_rhs_B[lid])

        assert prec_rhs_B[lid] == pytest.approx(prec_rhs[lid])

def test_multilevel_prec_stretched():
    parameters = {'Reynolds Number': 100, 'Grid Stretching': True,
                  'Preconditioner': {'Number of Levels': 2, 'Separator Length': 4}}
    test_prec_stretched(8, parameters)

def test_bordered_prec():
    try:
        from transiflow.interface import HYMLS as HYMLSInterface
        from PyTrilinos import Epetra
    except ImportError:
        pytest.skip("HYMLS not found")

    nx = 4
    ny = nx
    nz = nx
    dof = 4
    n = nx * ny * nz * dof

    parameters = {'Reynolds Number': 100, 'Preconditioner': {'Number of Levels': 0}}

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    V = numpy.zeros(n)
    for i in range(n):
        state[i] = i+2

    W = numpy.zeros(n)
    for i in range(n):
        state[i] = i+3

    C = 42

    rhs_2 = 4

    interface = HYMLSInterface.Interface(parameters, nx, ny, nz, dof=dof)
    state = interface.vector_from_array(state)
    V = interface.vector_from_array(V)
    W = interface.vector_from_array(W)

    interface.jacobian(state)
    rhs = interface.rhs(state)

    rhs_2_sol = Epetra.SerialDenseMatrix(1, 1)
    rhs_2_sol[0, 0] = rhs_2

    prec_rhs_2 = Epetra.SerialDenseMatrix(1, 1)

    V_sol = HYMLSInterface.Vector(interface.solve_map)
    V_sol.Import(V, interface.solve_importer, Epetra.Insert)

    W_sol = HYMLSInterface.Vector(interface.solve_map)
    W_sol.Import(W, interface.solve_importer, Epetra.Insert)

    C_sol = Epetra.SerialDenseMatrix(1, 1)
    C_sol[0, 0] = C

    rhs_sol = HYMLSInterface.Vector(interface.solve_map)
    prec_rhs = HYMLSInterface.Vector(interface.solve_map)
    rhs_sol.Import(rhs, interface.solve_importer, Epetra.Insert)

    interface.preconditioner.SetBorder(V_sol, W_sol, C_sol)
    interface.preconditioner.Compute()
    interface.preconditioner.ApplyInverse(rhs_sol, rhs_2_sol, prec_rhs, prec_rhs_2)

    # write_vector(prec_rhs, 'ldc_bordered_prec_rhs_%sx%sx%s.txt' % (nx, ny, nz))
    # write_value(prec_rhs_2[0, 0], 'ldc_bordered_prec_rhs_2_%sx%sx%s.txt' % (nx, ny, nz))

    prec_rhs_B = read_vector('data/ldc_bordered_prec_rhs_%sx%sx%s.txt' % (nx, ny, nz), interface.solve_map)
    prec_rhs_2_B = read_value('data/ldc_bordered_prec_rhs_2_%sx%sx%s.txt' % (nx, ny, nz))

    for i in range(n):
        lid = interface.solve_map.LID(i)
        if lid == -1:
            continue

        print(i, lid, prec_rhs[lid], prec_rhs_B[lid])

        assert prec_rhs_B[lid] == pytest.approx(prec_rhs[lid])

    assert prec_rhs_2_B == prec_rhs_2[0, 0]

def test_norm():
    try:
        from transiflow.interface import HYMLS as HYMLSInterface
    except ImportError:
        pytest.skip("HYMLS not found")

    nx = 4
    ny = nx
    nz = nx
    dof = 4
    parameters = {}
    n = nx * ny * nz * dof

    state = numpy.zeros(n)
    for i in range(n):
        state[i] = i+1

    interface = HYMLSInterface.Interface(parameters, nx, ny, nz, dof=dof)

    state_dist = HYMLSInterface.Vector.from_array(interface.map, state)
    assert utils.norm(state) == utils.norm(state_dist)

    state_dist = HYMLSInterface.Vector.from_array(interface.solve_map, state)
    assert utils.norm(state) == utils.norm(state_dist)

def test_HYMLS(nx=4):
    try:
        from transiflow.interface import HYMLS as HYMLSInterface
        from PyTrilinos import Teuchos
    except ImportError:
        pytest.skip("HYMLS not found")

    numpy.random.seed(1234)

    ny = nx
    nz = nx

    parameters = Teuchos.ParameterList()
    parameters.set('Reynolds Number', 0)

    interface = HYMLSInterface.Interface(parameters, nx, ny, nz)
    continuation = Continuation(interface)

    x0 = interface.vector()
    x0 = continuation.newton(x0)

    start = 0
    target = 100
    ds = 100
    x = continuation.continuation(x0, 'Reynolds Number', start, target, ds)[0]

    assert x.Norm2() > 0
