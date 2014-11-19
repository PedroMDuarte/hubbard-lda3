
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
 
import spi_calc 
from colorChooser import rgb_to_hex, cmapCycle




# Dictionary with chemical potential, spiextent, entextent

GREEN = 3.700

mudicts = {}

#        [  muplus, spiext, entext, done ]
mudicts[ 80] = { 
    0.53: [0.115, 20.0, 30, True] ,\
    0.55: [0.115, 20.5, 30, True] ,\
    0.57: [0.115, 21.0, 30, True] ,\
    0.61: [0.115, 22.0, 30, True] ,\
    0.68: [0.114, 23.0, 30, True] ,\
    0.80: [0.112, 24.0, 30, True] ,\
    1.09: [0.105, 24.0, 30, True] ,\
    1.99: [0.074, 24.0, 30, True] ,\
}
 
mudicts[120] = { 
    0.53: [0.115, 20.0, 30, True] ,\
    0.55: [0.115, 20.5, 30, True] ,\
    0.57: [0.115, 21.0, 30, True] ,\
    0.61: [0.115, 22.0, 30, True] ,\
    0.68: [0.113, 23.0, 30, True] ,\
    0.80: [0.111, 24.0, 30, True] ,\
    1.09: [0.104, 24.0, 30, True] ,\
    1.99: [0.072, 24.0, 30, True] ,\
} 

mudicts[160] = { 
    0.53: [0.112, 20.0, 30, True] ,\
    0.55: [0.112, 20.5, 30, True] ,\
    0.57: [0.112, 21.0, 30, True] ,\
    0.61: [0.111, 22.0, 30, True] ,\
    0.68: [0.110, 23.0, 30, True] ,\
    0.80: [0.108, 24.0, 30, True] ,\
    1.09: [0.100, 24.0, 30, True] ,\
    1.99: [0.068, 24.0, 30, True] ,\
} 

mudicts[200] = { 
    0.53: [0.107, 20.0, 30, True] ,\
    0.55: [0.107, 20.5, 30, True] ,\
    0.57: [0.107, 21.0, 30, True] ,\
    0.61: [0.107, 22.0, 30, True] ,\
    0.68: [0.105, 23.0, 30, True] ,\
    0.80: [0.103, 24.0, 30, True] ,\
    1.09: [0.095, 24.0, 30, True] ,\
    1.99: [0.062, 24.0, 30, True] ,\
} 

mudicts[245] = { 
    0.53: [0.095, 20.0, 30, True] ,\
    0.55: [0.095, 20.5, 30, True] ,\
    0.57: [0.095, 21.0, 30, True] ,\
    0.61: [0.094, 22.0, 30, True] ,\
    0.68: [0.093, 23.0, 30, True] ,\
    0.80: [0.090, 24.0, 30, True] ,\
    1.09: [0.082, 24.0, 30, True] ,\
    1.99: [0.049, 24.0, 30, True] ,\
}
 
mudicts[290] = { 
    0.53: [0.078, 20.0, 30, True] ,\
    0.55: [0.078, 20.5, 30, True] ,\
    0.57: [0.078, 21.0, 30, True] ,\
    0.61: [0.077, 22.0, 30, True] ,\
    0.68: [0.076, 23.0, 30, True] ,\
    0.80: [0.073, 24.0, 30, True] ,\
    1.09: [0.065, 24.0, 30, True] ,\
    1.99: [0.032, 24.0, 30, True] ,\
}

mudicts[335] = { 
    0.53: [0.054, 20.0, 30, True] ,\
    0.55: [0.054, 20.5, 30, True] ,\
    0.57: [0.054, 21.0, 30, True] ,\
    0.61: [0.054, 22.0, 30, True] ,\
    0.68: [0.053, 23.0, 30, True] ,\
    0.80: [0.050, 24.0, 30, True] ,\
    1.09: [0.042, 24.0, 30, True] ,\
    1.99: [0.009, 24.0, 30, True] ,\
}

mudicts[380] = { 
    0.53: [ 0.030, 19.0, 30, True] ,\
    0.55: [ 0.030, 19.5, 30, True] ,\
    0.57: [ 0.030, 20.0, 30, True] ,\
    0.61: [ 0.030, 21.0, 30, True] ,\
    0.68: [ 0.029, 22.0, 30, True] ,\
    0.80: [ 0.026, 23.0, 30, True] ,\
    1.09: [ 0.018, 24.0, 30, True] ,\
    1.99: [-0.013, 24.0, 30, True] ,\
}

mudicts[425] = { 
    0.53: [ 0.002, 18.5, 30, True] ,\
    0.55: [ 0.002, 19.0, 30, True] ,\
    0.57: [ 0.002, 20.0, 30, True] ,\
    0.61: [ 0.002, 21.0, 30, True] ,\
    0.68: [ 0.001, 22.0, 30, True] ,\
    0.80: [-0.001, 23.0, 30, True] ,\
    1.09: [-0.010, 24.0, 30, True] ,\
    1.99: [-0.040, 24.0, 30, True] ,\
}

mudicts[470] = { 
    0.53: [-0.027, 17.0, 30, True] ,\
    0.55: [-0.027, 18.0, 30, True] ,\
    0.57: [-0.027, 19.0, 30, True] ,\
    0.61: [-0.027, 21.0, 30, True] ,\
    0.68: [-0.028, 23.0, 30, True] ,\
    0.80: [-0.030, 24.0, 30, True] ,\
    1.09: [-0.039, 24.0, 30, True] ,\
    1.99: [-0.070, 24.0, 30, True] ,\
}
 
mudicts[515] = { 
    0.53: [-0.064, 17.0, 30, True] ,\
    0.55: [-0.064, 18.0, 30, True] ,\
    0.57: [-0.064, 19.0, 30, True] ,\
    0.61: [-0.062, 21.0, 30, True] ,\
    0.68: [-0.061, 23.0, 30, True] ,\
    0.80: [-0.064, 24.0, 30, True] ,\
    1.09: [-0.072, 24.0, 30, True] ,\
    1.99: [-0.104, 24.0, 30, True] ,\
}

mudicts[560] = { 
    0.53: [-0.102, 17.0, 30, True] ,\
    0.55: [-0.102, 18.0, 30, True] ,\
    0.57: [-0.102, 19.0, 30, True] ,\
    0.61: [-0.100, 21.0, 30, True] ,\
    0.68: [-0.093, 23.0, 30, True] ,\
    0.80: [-0.095, 24.0, 30, True] ,\
    1.09: [-0.103, 24.0, 30, True] ,\
    1.99: [-0.136, 24.0, 30, True] ,\
}


numpoly = np.loadtxt('atomNumberFit.dat')
def fnum(aS):
    x = aS * 11.1 / 290.
    return 1e5*( numpoly[0] + numpoly[1]*x + numpoly[2]*x**2 + numpoly[3]*x**3 )

if False:
    print fnum(80.)
    print fnum(200.)
    print fnum(290.)
    print fnum(380.)
    print fnum(470.)
    print fnum(560.)
    exit()




# 80, 120, 160, 200, 245, 290, 335, 380, 425, 470, 515, 560
# for aS in [290, 425, 470, 515, 560]:
# for aS in [200]:
for aS in [80, 120, 160, 200, 245, 290, 335, 380, 425, 470, 515, 560]:

    
    mu = mudicts[aS] 
    savedir = 'dataplots/THERM/{:03d}/'.format(int(aS)) 
 
    if not os.path.exists(savedir):
        os.makedirs(savedir) 
    if not os.path.exists(savedir + 'Inhomog'):
        os.makedirs(savedir + 'Inhomog')
    
    for T in sorted(mu.keys()):
    
        Tspi = T 
        if T < 0.60:
            Tdens = 0.60
        else:
            Tdens = Tspi
    
        muP = mu[T][0]
        spiextents = mu[T][1] 
        entextents = mu[T][2]
    
        # Skip over done
        # if mu[T][3] : continue 
    
        finegrid = False
        GREEN = 3.70
        spis = spi_calc.single_spi( 
              params_g  = GREEN, \
              params_aS = aS, \
              params_Tspi  = Tspi, \
              params_Tdens = Tdens, \
              savedir=savedir,\
              numlist = [fnum(aS)] ,\
              mulist = [muP],  bestForce=0, \
              spiextents=spiextents, \
              sthextents=30., \
              entextents=entextents,\
              finegrid=finegrid )
        
        fname = savedir + 'aS{:03d}_gr{:0.3f}_mu{:0.3f}_T{:0.3f}.pck'.format(aS,GREEN,muP,Tspi) 
        pickle.dump( spis, open( fname, "wb" ) )        
            
            
