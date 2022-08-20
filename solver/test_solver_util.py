import pytest
from solver_util import *


testdata = ( (1,1,1), \
             (1,1,8), \
             (2,2,1), \
             (4,2,1), \
             (4,2,3), \
             (9,5,1), \
             (9,5,2), \
             (9,5,6) )

@pytest.mark.parametrize("nx,ny,dof", testdata)
def test_z_ordering_1_to_1(nx,ny,dof):

        z_idx = get_z_ordering(nx,ny,dof=dof)
        z_idx.sort()
        wrong = z_idx - range(nx*ny*dof)
        assert not wrong@wrong

@pytest.mark.parametrize("nx,ny,dof", testdata)
def test_inv_ordering(nx,ny,dof):

        z_idx = get_z_ordering(nx,ny,dof=dof)
        z_inv = get_inv_ordering(z_idx)

        N = nx*ny*dof
        x = numpy.array(range(N))+1
        wrong = x - x[z_idx][z_inv]
        assert not wrong@wrong

