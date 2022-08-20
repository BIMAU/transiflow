import pytest
from solver_util import *

def test_z_ordering_1_to_1():

    for nx in range(1,6):
        for ny in range(1,6):
            for dof in range(1,4):
                z_idx = get_z_ordering(nx,ny,dof=dof)
                z_idx.sort()
                wrong = z_idx - range(nx*ny*dof)
                assert not wrong@wrong

if __name__ == '__main__':

    test_z_ordering_1_to_1()
