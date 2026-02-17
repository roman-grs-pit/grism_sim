import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table, join
import os,sys
from astropy.io import fits
from grizli.model import GrismFLT
import grizli

os.environ['github_dir']='/global/common/software/m4943/grizli0/'
sys.path.append(os.environ['github_dir']+'/grism_sim/py/')
os.environ["STPSF_PATH"]="/global/cfs/cdirs/m4943/grismsim/stpsf-data"
os.environ["WEBBPSF_PATH"]="/global/cfs/cdirs/m4943/grismsim/webbpsf-data"

import grism_sim

dx = np.arange(-4000,4000)
sl = [0,4088]
beams = ['A','B','C']
results = []
for det in range(1,19):
    for beam in beams:
        minxt = 1e6
        maxxt = -1e6
        for i in range(0,2):
            for j in range(0,2):
                output = grism_sim.get_trace(sl[i],sl[j],dx,[beam],'/global/homes/a/ajross/RomanGRS/github/grism_sim/data/testbuild/TestBuild_rot_det'+str(det)+'.conf')
                dy, lam = output[beam]
                sel = lam > 9000
                sel &= lam < 21000
                minx,maxx = np.min(dx[sel]),np.max(dx[sel])
                if minx < minxt:
                    minxt = minx
                if maxx > maxxt:
                    maxxt = maxx
        print(det,beam,minxt,maxxt,min(lam[sel]),max(lam[sel]))
        results.append((det,beam,minxt,maxxt,min(lam[sel]),max(lam[sel])))
print(results)
