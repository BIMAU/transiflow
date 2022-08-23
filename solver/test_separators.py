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

sda=0
i0a, i1a, i2a, i3a = DD.indices(sda)
sdb=1
i0b, i1b, i2b, i3b = DD.indices(sdb)
sdc=2
i0c, i1c, i2c, i3c = DD.indices(sdc)
sdd=3
i0d, i1d, i2d, i3d = DD.indices(sdd)

fig, axs = plt.subplots(2)
DD.plot_cgrid(i0a,'white',title='overlapping subdomain',markersize=15, ax=axs[0], plotgrid=True)
DD.plot_cgrid(i0b,'green',markersize=12, ax=axs[0], plotgrid=False)
DD.plot_cgrid(i0c,'blue',markersize=9, ax=axs[0], plotgrid=False)
DD.plot_cgrid(i0d,'yellow',markersize=6, ax=axs[0], plotgrid=False)

DD.plot_cgrid([i1a,flatten(i2a),i3a],['white','red','yellow'],markersize=[12,12,9], title='groups',ax=axs[1], plotgrid=True)


print(i0a)
print(i0b)


axs[0].set_aspect('equal', adjustable='box')
axs[1].set_aspect('equal', adjustable='box')
plt.show()
