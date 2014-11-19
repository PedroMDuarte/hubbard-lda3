
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
 
import spi_calc 
from colorChooser import rgb_to_hex, cmapCycle


savedir = 'dataplots/PROF/' 
if not os.path.exists(savedir):
    os.makedirs(savedir) 
if not os.path.exists(savedir + 'Inhomog'):
    os.makedirs(savedir + 'Inhomog') 

for aS,mu,spiext in [ (200, 0.107, 19.6), (290,0.079,19.6),\
               (380, 0.029,18.6), (470,-0.027,17.0) ]:
    
    #if aS in [200., 290., 470.] : continue

    finegrid = True 
    GREEN = 3.70
    spis = spi_calc.single_spi( 
          params_g  = GREEN, \
          params_aS = aS, \
          params_Tspi  = 0.53, \
          params_Tdens = 0.60, \
          savedir=savedir,\
          mulist = [mu],  bestForce=0, \
          spiextents=spiext, 
          entextents=17.,\
          finegrid=finegrid )
    
    fname = savedir + 'aS{:03d}_gr{:0.3f}_mu{:0.3f}.pck'.format(aS,GREEN,mu) 
    pickle.dump( spis, open( fname, "wb" ) )        
            
            
