#!/usr/bin/env python
#
# This examples shows how to plot the GW spectral functions A(w)
# See lesson tgw2_4
from abipy import *
import abipy.data as data

# Open the file with the GW results
sigma_file = SIGRES_File(data.ref_file("tgw2_4o_SIGRES.nc"))

# Plot A(w) for the first spin, the gamma point, and bands in [0,1,2,3]
sigma_file.plot_spectral_functions(spin=0, kpoint=(0,0,0), bands=range(0,4))
