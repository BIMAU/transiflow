from solver_util import *

nx=16
ny=16

# note: this currently only works if
# nx is a multiple of sx, etc., because otherwise
# we'll get holes and we need to reindex the z-curve.
sx=4
sy=4
dof=1

V = get_separator_groups(nx,sx,ny,sy,dof=dof)
