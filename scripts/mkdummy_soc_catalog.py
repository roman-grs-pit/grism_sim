from astropy.table import Table
import os
t = Table()

ral = [10.,10.1]
t['ra'] = ral
decl = [1.,1.02]
t['dec'] = decl
idl = [0,1]
t['sourceid'] = idl
magl = [23.4,23.2]
t['kron_f158_abmag'] = magl
mel = [0.1,0.1]
t['kron_f158_abmag_err'] = mel
rl = [0.35,0.128]
t['semimajor'] = rl
t['semiminor'] = rl
ol = [0,0]
t['orientation_sky'] = ol
el = [True,False]
t['is_extended_f158'] = el
t.write(os.getenv('HOME')+'/dummy_soc_catalog.fits')

