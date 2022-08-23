import pytest
from solver_util import *


def test_flatten_list_of_lists():
    l = flatten([[1],[2,3],[4,5,6]])
    assert len(l)==6
    for i in range(6):
        assert l[i] == i+1

@pytest.mark.skip(reason='this case is not working, must have a proper list of lists')
def test_flatten_list_of_lists_and_other_stuff():
    l = flatten([[1],2,3,[4,5,6],"Hello World!"])
    assert len(l)==7
    for i in range(7):
        assert l[i] == i+1

    assert l[7] == "Hello World!"

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

