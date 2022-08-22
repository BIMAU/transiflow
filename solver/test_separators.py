from matplotlib import pyplot as plt
from solver_util import *

nx=8
ny=6

# note: this currently only works if
# nx is a multiple of sx, etc., because otherwise
# we'll get holes and we need to reindex the z-curve.
sx=4
sy=4
dof=3

DD = StokesDD(nx,ny,sx,sy)

sd=0
i0, i1, i2, i3 =DD.indices(sd)

fig, axs = plt.subplots(2)
DD.plot_cgrid([i0],['yellow'],title='overlapping subdomain',markersize=12, ax=axs[0], plotgrid=True)
DD.plot_cgrid([i1,flatten(i2),i3],['white','red','yellow'],title='groups',ax=axs[1], plotgrid=True)

sd=1
i0, i1, i2, i3 =DD.indices(sd)
DD.plot_cgrid([i0],['blue'],markersize=10, ax=axs[0], plotgrid=False)

#DD.plot_cgrid(nx,ny,i1,'interior of subdomain')
#pDD.lot_cgrid(nx,ny,i2,'separators of subdomain')
#DD.plot_cgrid(nx,ny,i2,'pressures of subdomain')

#V2 = get_separator_groups(nx,sx,ny,sy,dof=dof)
#numpy.set_printoptions(precision=3, linewidth=128, threshold=1000,edgeitems=200)


#print('nx=%d, sx=%d'%(nx,sx))
#print('V2.shape='+str(V2.shape))
#print('V2:')
#print(V2.todense())

axs[0].set_aspect('equal', adjustable='box')
axs[1].set_aspect('equal', adjustable='box')
plt.show()
