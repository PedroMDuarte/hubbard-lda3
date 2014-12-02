
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 

import scubic
import lda
    
from colorChooser import rgb_to_hex, cmapCycle

def single_spi(**kwargs):

    savedir   = kwargs.pop('savedir', None)
    numlist   = kwargs.pop('numlist', [1.2e5, 1.3e5, 1.4e5, 1.5e5, 1.6e5] )  
    mulist    = kwargs.pop('mulist',  [-0.15, -0.075, 0., 0.10, 0.18] )  
    bestForce = kwargs.pop('bestForce',-1) 

    s       = kwargs.pop('params_s', 7.) 
    g       = kwargs.pop('params_g', 3.666)
    wIR     = kwargs.pop('params_wIR', 47.) 
    wGR     = kwargs.pop('params_wGR', 47./1.175) 
    direc   = '111'
    mu0     = 'halfMott'

    aS = kwargs.pop('params_aS', 300.)
    Tdens  = kwargs.pop('params_Tdens', 0.6 )
    Tspi   = kwargs.pop('params_Tspi',  0.6 )

    extents = kwargs.pop('extents', 30.)
    spiextents = kwargs.pop('spiextents', 25.) 
    sthextents = kwargs.pop('sthextents', 30.) 
    entextents = kwargs.pop('entextents', 25.) 
    finegrid = kwargs.pop('finegrid', False)

    sarr = np.array( [[s],[s],[s]] )
    bands = scubic.bands3dvec( sarr , NBand = 0 )
    t0 = np.mean( (bands[1] - bands[0])/12.)  # tunneling 0 in recoils 

    Tdens_Er = Tdens*t0
    Tspi_Er  = Tspi*t0

    print "========================================"
    print " Single Spi"
    print " gr={:0.3f}, aS={:03d}".format( g, int(aS) )
    print " Tdens={:0.2f}, Tspi={:0.2f}".format(Tdens, Tspi) 

    select = 'qmc'
    spis = [] 

    for tag, muPlus in enumerate(mulist):
        numgoal = numlist[tag] 
        print 
        print "num = %.3g, muPlus = %.3f"%(numgoal, muPlus)
        pot = scubic.sc(allIR=s, allGR=g, allIRw=wIR, allGRw=wGR)

        lda0 = lda.lda(potential = pot, Temperature=Tdens_Er, a_s=aS, \
                       extents=extents, \
                       Natoms=numgoal, halfMottPlus=muPlus,\
#                       globalMu=mu0, halfMottPlus=muPlus,\
                       verbose=True, \
                       select = select,\
                       ignoreExtents=False, ignoreSlopeErrors=True, \
                       ignoreMuThreshold=True)

        spibulk, spi, sthbulk, sth, r111, n111, U111, t111, mut111, \
        entrbulk, entr111,\
        lda_num, density111, k111, k111htse_list = \
            lda0.getBulkSpi(Tspi=Tspi, inhomog=True, \
               spiextents=spiextents, sthextents=sthextents, \
               entextents=entextents, do_k111=False)
    
        if finegrid: 
            r111_fine, spi111_fine, n111_fine, k111_fine, mu111_Er = \
                lda0.getSpiFineGrid( Tspi=Tspi, numpoints=320,\
                    inhomog=True, spiextents=spiextents, \
                    entextents=entextents )
        else:
            r111_fine, spi111_fine, n111_fine, k111_fine, mu111_Er = \
                 None, None, None, None, None


        spis.append( {
                      'gr':g,\
                      'muPlus':muPlus,\
                      'SpiBulk':spibulk,\
                      'spi111':spi,\
                      'SthBulk':sthbulk,\
                      'sth111':sth,\
                      'r111':r111,\
                      'n111':n111,\
                      'U111':U111,\
                      'mut111':mut111,\
                      't111':t111,\
                      'entrbulk':entrbulk,\
                      'entr111':entr111,\
                      'k111':k111,\
                      'k111htse_list':k111htse_list,\
                      'Number':lda0.Number,\
                      'ldanum':lda_num,\
                      # dens111 is the one obtained from QMC
                      'dens111':density111,\
                      'Tdens':Tdens,\
                      'Tspi':Tspi,\
                      'aS':aS,\
                      'savedir':savedir,\
                      'r111_fine':r111_fine,\
                      'spi111_fine':spi111_fine,\
                      'n111_fine':n111_fine,\
                      'k111_fine':k111_fine,\
                      'mu111_Er':mu111_Er,\
                      'v0111': lda0.pot.S0(lda0.X111, lda0.Y111, lda0.Z111)[0]
                      } ) 

        # Figure to check inhomogeneity only run if temperature is high
        if Tspi > 0.85 and Tdens > 0.85:
            fig111, binresult, peak_dens, radius1e, peak_t, output = \
                lda.CheckInhomog( lda0, closefig = True, n_ylim=(-0.1,2.0) ) ;

            figfname = savedir + 'Inhomog/{:0.3f}gr_{:03d}_{}_T{:0.4f}Er.png'.\
                       format(g,tag,select,Tspi)
    
            figfname = kwargs.pop( 'params_figfname', figfname) 
    
            fig111.savefig(figfname, dpi=300)

    print 
    print "Atom number = {:5.3g}".format(spis[0]['Number'])
    print "Entropy     = {:0.2f}".format(spis[0]['entrbulk'])

    plot_spis( spis, bestForce=bestForce, \
          # kwargs 
          **kwargs)
    
    return spis[bestForce] 



def plot_spis( spis, bestForce = -1, **kwargs):
    """ This function makes a nice plot of the results of 
    the studies of Spi_vs_n""" 

    from matplotlib import rc
    rc('font', **{'family':'serif'})
    rc('text', usetex=True)

    fig = plt.figure(figsize=(9.,6.8))
    gs = matplotlib.gridspec.GridSpec(4,4,\
            wspace=0.45, hspace=0.1,\
            left=0.07, right=0.96, bottom=0.08, top=0.9)

    

    axn = fig.add_subplot( gs[0,0])
    axSpi = fig.add_subplot( gs[0,1]) 
    axEnt = fig.add_subplot( gs[0,2])
    axEntP = fig.add_subplot( gs[3,2])

    axVol  = fig.add_subplot( gs[1,0] ) 
    axVolSpi = fig.add_subplot( gs[1,1] ) 
    axVolEnt = fig.add_subplot( gs[1,2] ) 


    axT = fig.add_subplot( gs[3,0])
    axU = fig.add_subplot( gs[3,1])

    axSpiB = fig.add_subplot(gs[0,3])
    axEntB = fig.add_subplot(gs[1,3])
    axFrac = fig.add_subplot(gs[3,3])

    axSPI = fig.add_subplot( gs[2,0]) 
    axSTH = fig.add_subplot( gs[2,1]) 


    savedir = spis[0]['savedir']
    aS = spis[0]['aS'] 
    gr = spis[0]['gr'] 
    U0 = spis[0]['U111'].max()
    T0 = spis[0]['Tspi']
    t0 = spis[0]['t111'].min()

    T0dens = spis[0]['Tdens']
    
    titletext  = r'$g_{{0}}={:0.2f}\,E_{{r}} \ \ \ $'.format(gr)
    titletext += r'$\mathrm{{density\ at\ }} [T/t]_{{0}}={:0.2f}$\, \ \ \ '\
               .format(T0dens)
    titletext += r'$S_{{\pi}}\ \mathrm{{at}}\ \ [U/t]_{{0}}={:0.1f}\ , \  \ $'\
                 .format(U0) + \
                 r'$[T/t]_{{0}}={:0.2f}\ \ $'.format(T0) 

    fig.text(0.5, 0.98, titletext, ha='center', va='top', fontsize=13) 


    spiBs = []
    sthBs = []
    for spi in spis:
        spiBs.append( spi['SpiBulk'] ) 
        sthBs.append( spi['SthBulk'] ) 
    best = np.argmax( spiBs ) 
    bestTH = np.argmax( sthBs ) 

    if bestForce != -1:
        best = bestForce

    print 
    print "Best Spi result at ", best
    print "  Spi =", spiBs[best]
    print "  Sth =", sthBs[best]
        

    results = [] 
    for ii, spi in enumerate(spis):

        try:
            color = cmapCycle( matplotlib.cm.jet, float(ii), \
                    lbound=0, ubound=float(len(spis)-1))
        except:
            color = 'blue'
  
        spiB = spi['SpiBulk']
        sthB = spi['SthBulk']
        spi0 = spi['spi111'].max()
        entB = spi['entrbulk'] 

        # effective contributing fraction
        x = (spiB - 1.)/(spi0-1.)


        t0 = spi['t111'].min()
        #print "T (density) = ", spi['Tn']/t0
        Tspi = spi['Tspi']

        lw =1.5 
        axn.plot(spi['r111'], spi['n111'], color=color, lw=lw,\
            label=r'${:0.2f}({:0.3f})$'.\
                       format(spi['Number']/1e5, spi['muPlus']))
            #label='{:0.1f}({:0.1f})'.format(spi['Number']/1e5, spi['ldanum']/1e5))

        axn.plot(spi['r111'], spi['dens111'], 'o', color=color, lw=lw, \
            ms=3., alpha=0.3 ) 

        axSpi.plot(spi['r111'], spi['spi111'], color=color, lw=lw)

        axSPI.plot(spi['r111'], spi['spi111']*spi['n111'], color=color, lw=lw*2./3.)
        axSTH.plot(spi['r111'], spi['sth111'], color=color, lw=lw*2./3.)

        axEnt.plot(spi['r111'], spi['entr111'], color=color, lw=lw)
        axEntP.plot(spi['r111'], spi['entr111']/spi['dens111'], color=color, lw=lw)

        axVol.plot(spi['r111'], \
            4.*np.pi * np.power( spi['r111'],2) * spi['n111'] / 1e3 ,\
                   color=color, lw=lw)

        axVolSpi.plot(spi['r111'], \
            4.*np.pi * np.power( spi['r111'],2) * spi['spi111'] \
                      * spi['n111'] / 1e3 ,\
              color=color, lw=lw)
        axVolSpi.plot(spi['r111'], \
            4.*np.pi * np.power( spi['r111'],2) * spi['sth111'] / 1e3 ,\
              color=color, lw=lw*2./3., alpha=0.6)

        axVolEnt.plot(spi['r111'], \
            4.*np.pi * np.power( spi['r111'],2) * spi['entr111'] / 1e3,\
              color=color, lw=lw)
         

        axT.plot( spi['r111'], Tspi * t0 / spi['t111'], color='black', lw=lw  )
        axU.plot( spi['r111'], spi['U111'], color='black', lw=lw )

        pos = spi['r111'] > 0 
        indmax =  np.argmax( spi['spi111'][pos]  ) 
        rmax = spi['r111'][pos][indmax] 
        tbest = spi['t111'][pos][indmax] 
        Tbest = Tspi * t0 / tbest 
        Ubest = spi['U111'][pos][indmax]
        results.append( [spi['Number']/1e5, spiB, x*100., entB, tbest, Tbest, Ubest, sthB] )

        # Find the max of spi 
        if ii == best:
            axSpi.axvline( rmax, color=color, alpha=0.7  ) 
            axSPI.axvline( rmax, color=color, alpha=0.7  ) 
            axSTH.axvline( rmax, color=color, alpha=0.7  ) 
            axn.axvline( rmax, color=color, alpha=0.7  ) 
            axT.axvline( rmax, color=color, alpha=0.7  ) 
            axU.axvline( rmax, color=color, alpha=0.7 )

            axT.text( 0.95, 0.95,  '$[T/t]^{{*}}={:0.2f}$'.format(Tbest),\
                      ha='right', va='top', color='black',\
                      transform = axT.transAxes, fontsize=10, 
                      bbox= {'boxstyle':'round', 'facecolor':'white'}) 
            axU.text( 0.95, 0.95,  '$[U/t]^{{*}}={:0.1f}$'.format(Ubest),\
                      ha='right', va='top', color='black',\
                      transform = axU.transAxes, fontsize=10, 
                      bbox= {'boxstyle':'round', 'facecolor':'white'}) 
 

    results = np.array(results)
    

    fname = savedir + 'aS{:03d}_U{:02d}_Tspi{:0.2f}.dat'.format( int(aS), int(U0), T0 ) 
    np.savetxt( fname, results, fmt='%.4g',\
     header= '# N/1e5  Spi  Frac  S/N  t*  T/t*  U*/t*  Sth' )

    legend = axn.legend(title='$N/10^{5}$',bbox_to_anchor=(1.02,1.02), \
           loc='upper right', numpoints=1, \
           prop={'size':6}, handlelength=1.1, handletextpad=0.5)

    plt.setp(legend.get_title(),fontsize=6.5) 
 
    axSpiB.plot( results[:,0], results[:,1],'.-', color='black')
    axSpiB.plot( results[:,0], results[:,7],'.-', color='lightgray')
    axFrac.plot( results[:,0], results[:,2],'.-', color='black')
    axEntB.plot( results[:,0], results[:,3],'.-', color='black')

    axn.set_ylabel('$n$', rotation=0, labelpad=12, fontsize=13)
    axT.set_ylabel('$T/t$', rotation=0, labelpad=12, fontsize=13)
    axU.set_ylabel('$U/t$', rotation=0, labelpad=12, fontsize=13)
    axSpi.set_ylabel(r'$\frac{S_{\pi}}{n}$', rotation=0, labelpad=12, \
                     fontsize=13)

    axSPI.set_ylabel(r'$S_{\pi}$', rotation=0, labelpad=12, \
                     fontsize=13)
    axSTH.set_ylabel(r'$S_{\theta}$', rotation=0, labelpad=12, \
                     fontsize=13)

    axEnt.set_ylabel(r'$s$', rotation=0, labelpad=12, fontsize=13)
    axEntP.set_ylabel(r'$\frac{s}{n}$', rotation=0, labelpad=12, fontsize=14)
    axVol.set_ylabel(r'$(4\pi r^{2}) n $', rotation=90) 
    axVolSpi.set_ylabel(r'$(4\pi r^{2}) S_{\pi} $', rotation=90) 
    axVolEnt.set_ylabel(r'$(4\pi r^{2}) s $', rotation=90)


    def scale_text(ax):
        ax.text( 0.0, 1.0, r'$\times 10^{3}$', ha='left', va='bottom',\
             transform = ax.transAxes, fontsize=9) 
    for axV in [axVol, axVolSpi, axVolEnt]:
        scale_text(axV) 
    


    axSpiB.set_ylabel(r'$\bar{S}_{\pi}$', rotation=0, labelpad=12, \
                     fontsize=13)
    axEntB.set_ylabel(r'$\frac{S}{N}$', rotation=0, labelpad=12, \
                     fontsize=13)
    axFrac.set_ylabel('Effective fraction (\%)', fontsize=10)

    axFrac.set_xlabel('$N/10^{5}$')
   
    x1 = 29. 
    axn.set_xlim(0.,x1)
    axT.set_xlim(0.,x1)
    axU.set_xlim(0.,x1)
    axSpi.set_xlim(0.,x1) 
    axSPI.set_xlim(0.,x1) 
    axSTH.set_xlim(0.,x1) 

    axT.set_ylim(0.3, 1.00)
    axU.set_ylim(0., 20.00)
    axSpi.set_ylim(0.5, 6.0)
 
    axSPI.set_ylim(-0.1, 5.0) 
    axSTH.set_ylim(-0.1, 1.1) 

    axSpiB.set_xlim(0., 3.5)
    axFrac.set_xlim(0., 3.5)
    axSpiB.set_ylim(0.4,2.5)
    axFrac.set_ylim(0., 65.) 

    axEnt.set_xlim(0., x1) 
    axEntP.set_xlim(0., x1) 
    axVol.set_xlim(0., x1)
    axVolSpi.set_xlim(0., x1)
    axVolEnt.set_xlim(0., x1)  
    axEntB.set_xlim( 0., 3.5  )

    axEnt.set_ylim(0., 1.0) 
    axEntP.set_ylim(0.,4.0) 
    axVol.set_ylim(0., 3.0)
    axVolSpi.set_ylim(0., 15.)
    axVolEnt.set_ylim(0., 4.0)  
    axEntB.set_ylim( 0., 2.0  )
   

    for ax in [axVol, axVolSpi, axVolEnt, axEntB, axU, axT]:
        ax.yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(5) ) 

    for ax in [axSpiB, axFrac]:
        ax.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(5) ) 

    for ax in [axn, axSpi, axSPI, axSTH, axEnt, axVol, axVolSpi, \
               axVolEnt, axSpiB, axEntB]:
        ax.xaxis.set_ticklabels([])
    for ax in [axT, axU, axEntP]:
        ax.set_xlabel('$r_{111}\ (\mu\mathrm{m})$')


    for ax in [axn, axSpi, axSPI, axSTH, axT, axU, axSpiB, axFrac, \
             axEnt, axEntP, axVol, axVolEnt, axVolSpi, axEntB]:
        ax.grid(alpha=0.3)

  
    figfname2 =  savedir + 'Tn{:0.2f}_gr{:0.3f}_U{:04.1f}_T{:0.2f}'.\
                 format(T0dens,gr,U0,T0)  +  '.png'
    figfname2 = kwargs.pop( 'figfname2', figfname2) 
    fig.savefig( figfname2, dpi=300)


import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('spi_calc.py')

    parser.add_argument('SCATTLEN', action="store", type=float, \
        help='Scattering length') 

    parser.add_argument('GREEN', action="store", type=float, \
        help='Green compensation') 

    parser.add_argument('TSPI', action="store", type=float, \
        help='The value of [T/t]_0 used for Spi')

    parser.add_argument('TDENS', action="store", type=float, \
        help='The value of [T/t]_0 used for density')

    parser.add_argument('--savedir', action="store", type=str, \
        default='spicalc/',\
        help='Directory to put the results')

    parser.add_argument('--mu', nargs='+', type=float, \
        default=[0.],\
        help='List of chemical potentials to consider') 

    parser.add_argument('--best', action='store', type=int, default=-1,\
        help='Selects which of the mus to highlight in the plot')

    parser.add_argument('--extents', action='store', type=float, default=30.,\
        help='Sets the extents over which to calculate Spi')

    parser.add_argument('--spiextents', action='store', type=float, default=25.,\
        help='Sets the extents over which to calculate Spi')

    parser.add_argument('--entextents', action='store', type=float, default=25.,\
        help='Sets the extents over which to calculate Entropy')

    args = parser.parse_args()

    print args  

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir) 


    #Spi_inhomog = False
    #aS      = 200. 
    #Ts      = 0.9 
    single_spi(
              params_g  = args.GREEN ,\
              params_aS = args.SCATTLEN, \
              params_Tspi  = args.TSPI, \
              params_Tdens  = args.TDENS, \
              savedir = args.savedir,\
              mulist = args.mu,\
              bestForce = args.best,\
              spiextents = args.spiextents,\
              entextents = args.entextents
             ) 
