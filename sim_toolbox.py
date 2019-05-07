#!/usr/bin/env python3
# Copyright 2014-2018 by Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.


import numpy as np
import numpy.ma as ma
from scipy import interpolate as interp
from scipy.ndimage.filters import gaussian_filter


# Toolbox of functions used in the Simulator class to calculate key bioelectric properties.

def electroflux(cA,cB,Dc,d,zc,vBA,T,p,rho=1):
    """
    Electro-diffusion between two connected volumes. Note for cell work, 'b' is
    'inside', 'a' is outside, with a positive flux moving from a to b. The
    voltage is defined as Vb - Va (Vba), which is equivalent to Vmem.

    This function defaults to regular diffusion if Vba == 0.0.

    This function takes numpy matrix values as input. All inputs must be
    matrices of the same shape.

    This is the Goldman Flux/Current Equation (not to be confused with the
    Goldman Equation). Note: the Nernst-Planck equation has been trialed in
    place of this, and it does not reproduce proper reversal potentials.

    Parameters
    ----------
    cA          concentration in region A [moles/m3] (out)
    cB          concentration in region B [moles/m3] (in)
    Dc          Diffusion constant of c  [m2/s]
    d           Distance between region A and region B [m]
    zc          valence of ionic species c
    vBA         voltage difference between region B (in) and A (out) = Vmem
    p           an instance of the Parameters class

    Returns
    --------
    flux        Chemical flux magnitude between region A and B [mol/s]

    """

    # Reasonably small real number, preventing divide by zero errors.
    # Note that "betse.util.type.numeric.floats.FLOAT_MIN", the
    # smallest possible real number, is too small for this use case.
    FLOAT_NONCE = 1.0e-25

    vBA += FLOAT_NONCE

    zc += FLOAT_NONCE

    alpha = (zc*vBA*p.F)/(p.R*T)

    exp_alpha = np.exp(-alpha)

    deno = -np.expm1(-alpha)   # calculate the denominator for the electrodiffusion equation,..
    #
    # izero = (deno==0).nonzero()     # get the indices of the zero and non-zero elements of the denominator
    # inotzero = (deno!=0).nonzero()
    #
    # # initialize data matrices to the same shape as input data
    # flux = np.zeros(deno.shape)
    #
    # if len(deno[izero]):   # if there's anything in the izero array:
    #      # calculate the flux for those elements as standard diffusion [mol/m2s]:
    #     flux[izero] = -(Dc[izero]/d[izero])*(cB[izero] - cA[izero])
    #
    # if len(deno[inotzero]):   # if there's any indices in the inotzero array:

    # calculate the flux for those elements:
    flux = -((Dc*alpha)/d)*((cB -cA*exp_alpha)/deno)*rho

    # flux = flux*rho

    return flux

def pumpNaKATP(cNai,cNao,cKi,cKo,Vm,T,p,block, met = None):

    """
    Parameters
    ----------
    cNai            Concentration of Na+ inside the cell
    cNao            Concentration of Na+ outside the cell
    cKi             Concentration of K+ inside the cell
    cKo             Concentration of K+ outside the cell
    Vm              Voltage across cell membrane [V]
    p               An instance of Parameters object

    met             A "metabolism" vector containing concentrations of ATP, ADP and Pi


    Returns
    -------
    f_Na            Na+ flux (into cell +)
    f_K             K+ flux (into cell +)
    """

    deltaGATP_o = p.deltaGATP  # standard free energy of ATP hydrolysis reaction in J/(mol K)

    cATP = p.cATP
    cADP = p.cADP
    cPi  = p.cPi

    # calculate the reaction coefficient Q:
    Qnumo = (cADP*1e-3)*(cPi*1e-3)*((cNao*1e-3)**3)*((cKi*1e-3)**2)
    Qdenomo = (cATP*1e-3)*((cNai*1e-3)**3)*((cKo*1e-3)** 2)

    # ensure no chance of dividing by zero:
    inds_Z = (Qdenomo == 0.0).nonzero()
    Qdenomo[inds_Z] = 1.0e-15

    Q = Qnumo / Qdenomo


    # calculate the equilibrium constant for the pump reaction:
    Keq = np.exp(-(deltaGATP_o / (p.R * T) - ((p.F * Vm) / (p.R * T))))

    # calculate the enzyme coefficient:
    numo_E = ((cNai/p.KmNK_Na)**3) * ((cKo/p.KmNK_K)**2) * (cATP/p.KmNK_ATP)
    denomo_E = (1 + (cNai/p.KmNK_Na)**3)*(1+(cKo/p.KmNK_K)**2)*(1+(cATP/p.KmNK_ATP))

    fwd_co = numo_E/denomo_E

    f_Na = -3*block*p.alpha_NaK*fwd_co*(1 - (Q/Keq))  # flux as [mol/m2s]   scaled to concentrations Na in and K out

    f_K = -(2/3)*f_Na          # flux as [mol/m2s]

    return f_Na, f_K, -f_Na  # FIXME get rid of this return of extra -f_Na!!


