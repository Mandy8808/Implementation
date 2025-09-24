import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.optimize import newton
from scipy.optimize import root


def SystEq_EGBstars(r, yv, arg):
    """
    beta = arg
    N, g, phi, dN, dphi = yv
    """
    beta, c1 = arg
    n, g, p, dn, dp = yv
    
    f1 = dn
    f2 = (g**3*r*(-2*g**4*(-1 + g**2)*n*r**2 - 16*beta*c1**2*dp*g**2*(-1 + g**2)*p*r*(n + dn*r) - 8*beta*c1**4*dp**3*n*p*r*(-16*beta + g**2*(16*beta + r**2))\
        + c1**2*dp**2*(64*beta**2*c1**2*dn*(-1 + g**2)*p**2*r + n*(-64*beta**2*c1**2*(5 - 6*g**2 + g**4)*p**2 + g**2*r**2*(-16*beta + g**2*(16*beta\
        + r**2))))))/(256*beta**2*c1**4*dp**2*g**2*n*p**2*r**2 + 64*beta*c1**2*dp*p*(16*beta**2*c1**2*dn*(-3 + 2*g**2 + g**4)*p**2 - g**4*n*r**3)\
        + 4*g**2*(-128*beta**2*c1**2*dn*(-1 + g**2)*p**2*r + n*(-32*beta**2*c1**2*(-1 + g**2)**2*p**2 + g**4*r**4)))
    f3 = dp
    f4 = (g**2*r*(-2*g**4*n*r*((-1 + g**2)*n + dn*(1 + g**2)*r) - 16*beta*c1**2*dp*g**2*p*(2*(-1 + g**2)*n**2 + dn*(1 + g**2)*n*r + dn**2*(1 + g**2)*r**2)\
        - 16*beta*c1**4*dp**3*g**2*n*p*r*(-(n*r) + dn*(16*beta + r**2)) + c1**2*dp**2*(192*beta**2*c1**2*dn**2*(1 + g**2)*p**2*r - g**2*n**2*r*(16*beta\
        + g**2*(-16*beta + r**2)) + dn*n*(-64*beta**2*c1**2*(3 - 12*g**2 + g**4)*p**2 + g**2*r**2*(16*beta + g**2*(16*beta\
        + r**2))))))/(256*beta**2*c1**4*dp**2*g**2*n*p**2*r**2 + 64*beta*c1**2*dp*p*(16*beta**2*c1**2*dn*(-3 + 2*g**2 + g**4)*p**2 - g**4*n*r**3)\
        + 4*g**2*(-128*beta**2*c1**2*dn*(-1 + g**2)*p**2*r + n*(-32*beta**2*c1**2*(-1 + g**2)**2*p**2 + g**4*r**4)))
    f5 = (-16*beta*g**4*(-1 + g**2)*p*((-1 + g**2)*n + 4*dn*r) - 8*beta*c1**4*dp**4*g**2*n*p*r**2*(-16*beta + g**2*(16*beta + r**2)) - 2*dp*g**2*(g**4*(3\
        + g**2)*n*r**3 + 2*dn*(-32*beta**2*c1**2*(-3 + 2*g**2 + g**4)*p**2 + g**4*r**4)) + 8*beta*c1**2*dp**2*g**2*p*(-2*dn*r*(32*beta + g**4*r**2\
        - 2*g**2*(16*beta + 3*r**2)) + n*(16*beta + g**4*(16*beta - 3*r**2) + g**2*(-32*beta + 19*r**2))) + c1**2*dp**3*(-128*beta**2*c1**2*dn*p**2*(-24*beta\
        + 8*beta*g**4 + g**2*(16*beta + 3*r**2)) + g**2*n*r*(-64*beta**2*c1**2*(15 - 8*g**2 + g**4)*p**2 + g**2*r**2*(-16*beta + g**2*(16*beta\
        + r**2)))))/(256*beta**2*c1**4*dp**2*g**2*n*p**2*r**2 + 64*beta*c1**2*dp*p*(16*beta**2*c1**2*dn*(-3 + 2*g**2 + g**4)*p**2 - g**4*n*r**3)\
        + 4*g**2*(-128*beta**2*c1**2*dn*(-1 + g**2)*p**2*r + n*(-32*beta**2*c1**2*(-1 + g**2)**2*p**2 + g**4*r**4)))
    
    return [f1, f2, f3, f4, f5]


def SystEq_EGBHstars(r, yv, arg):
    """
    beta = arg
    N, g, phi, dN, dphi, p = yv
    """
    #print("SystEq_EGBHstars")
    beta, c1, edo = arg
    n, g, p, dn, dp, pres = yv

    if pres<=0:
        # print('=> ', pres)
        pres = 0.0
    
    den = edo(pres)
    
    f1 = dn
    f2 = (g**3*r*(-8*beta*c1**4*dp**3*n*p*r*(-16*beta + g**2*(16*beta + r**2)) - 16*beta*c1**2*dp*g**2*p*r*(dn*(-1 + g**2)*r + n*(-1 + g**2*(1 + c1**2*den*r**2)))\
        + c1**2*dp**2*(64*beta**2*c1**2*dn*(-1 + g**2)*p**2*r + n*(-64*beta**2*c1**2*(5 - 6*g**2 + g**4)*p**2 + g**2*r**2*(-16*beta + g**2*(16*beta + r**2))))\
        + 2*g**2*(-64*beta**2*c1**4*den*dn*(-1 + g**2)*p**2*r + n*(64*beta**2*c1**4*(-1 + g**2)**2*p**2*pres + g**2*r**2*(1 + g**2*(-1\
        + c1**2*den*r**2))))))/(256*beta**2*c1**4*dp**2*g**2*n*p**2*r**2 + 64*beta*c1**2*dp*p*(16*beta**2*c1**2*dn*(-3 + 2*g**2 + g**4)*p**2 - g**4*n*r**3)\
        + 4*g**2*(-128*beta**2*c1**2*dn*(-1 + g**2)*p**2*r + n*(-32*beta**2*c1**2*(-1 + g**2)**2*p**2 + g**4*r**4)))
    f3 = dp
    f4 = (g**2*r*(192*beta**2*c1**4*dn*dp**2*p**2*(-n + dn*r) + 2*g**6*n*(n*r*(-1 + c1**2*(den + 2*pres)*r**2) + dn*(64*beta**2*c1**4*p**2*pres - r**2 + c1**2*den*r**4))\
        + 16*beta*c1**2*g**2*(dp*p*(n - dn*r)*(2*n + dn*r) + 24*beta*c1**2*dn*p**2*(n*pres + den*dn*r) + c1**2*dp**3*n*p*r*(n*r - dn*(16*beta + r**2))\
        + dp**2*(-(n**2*r) + 12*beta*c1**2*dn**2*p**2*r + dn*n*(48*beta*c1**2*p**2 + r**2))) + g**4*(-16*beta*c1**2*dn**2*p*r*(8*beta*c1**2*den*p + dp*r) + n**2*(2*r\
        + c1**2*dp*(-(dp*r*(-16*beta + r**2)) - 32*p*(beta + beta*c1**2*pres*r**2))) + dn*n*(-512*beta**2*c1**4*p**2*pres - 2*r**2 + c1**2*dp*(-16*beta*p*r*(1\
        + 2*c1**2*den*r**2) + dp*(-64*beta**2*c1**2*p**2 + 16*beta*r**2 + r**4))))))/(256*beta**2*c1**4*dp**2*g**2*n*p**2*r**2\
        + 64*beta*c1**2*dp*p*(16*beta**2*c1**2*dn*(-3 + 2*g**2 + g**4)*p**2 - g**4*n*r**3) + 4*g**2*(-128*beta**2*c1**2*dn*(-1 + g**2)*p**2*r + n*(-32*beta**2*c1**2*(-1\
        + g**2)**2*p**2 + g**4*r**4)))
    f5 = (-8*beta*c1**4*dp**4*g**2*n*p*r**2*(-16*beta + g**2*(16*beta + r**2)) + 16*beta*g**4*p*(2*dn*r*(2 + g**2*(-2 + c1**2*den*r**2)) + (-1 + g**2)*n*(1 + g**2*(-1\
        + c1**2*(den + 2*pres)*r**2))) + 8*beta*c1**2*dp**2*g**2*p*(-2*dn*r*(32*beta + g**4*r**2 - 2*g**2*(16*beta + 3*r**2)) + n*(16*beta + g**2*(-32*beta + 19*r**2)\
        + g**4*(16*beta - 3*r**2 - 2*c1**2*den*r**4))) + c1**2*dp**3*(-128*beta**2*c1**2*dn*p**2*(-24*beta + 8*beta*g**4 + g**2*(16*beta + 3*r**2))\
        + g**2*n*r*(-64*beta**2*c1**2*(15 - 8*g**2 + g**4)*p**2 + g**2*r**2*(-16*beta + g**2*(16*beta + r**2)))) + 2*dp*g**2*(g**2*n*r*(64*beta**2*c1**4*(3 - 4*g**2\
        + g**4)*p**2*pres + g**2*r**2*(-3 + g**2*(-1 + c1**2*den*r**2))) - 2*dn*(g**4*r**4 + 32*beta**2*c1**2*p**2*(3 - 2*g**2 + g**4*(-1\
        + 2*c1**2*den*r**2)))))/(256*beta**2*c1**4*dp**2*g**2*n*p**2*r**2 + 64*beta*c1**2*dp*p*(16*beta**2*c1**2*dn*(-3 + 2*g**2 + g**4)*p**2 - g**4*n*r**3)\
        + 4*g**2*(-128*beta**2*c1**2*dn*(-1 + g**2)*p**2*r + n*(-32*beta**2*c1**2*(-1 + g**2)**2*p**2 + g**4*r**4)))
    f6 = -((dn*(den + pres))/n)
    
    return [f1, f2, f3, f4, f5, f6]


def main_NDsolverEGBHstars(beta, edo, V0=None, argB=None, c1=1, Nptos=1000, rmin=1e-2, rmax=1e2,
             met='RK45', Rtol=1e-6, Atol=1e-6, zeroP_val=1e-8, info=False):
    
  
    if argB:
        print("Solving from exterior")
        pR = argB[-1]
        R_star = argB[-2]
        argB = argB[:-2]
        rspan, sol = NDSol_EGBHstars_R_to_L(R_star, beta, argB, c1, edo, Nptos=Nptos, rmin=rmin, rmax=rmax,
             met=met, Rtol=Rtol, Atol=Atol, info=info, pR=pR)
        return rspan, sol
    else:
        print("Solving from interior")
        rspan, sol, R_star = NDSol_EGBHstars_L_to_R(V0, beta, c1, edo, Nptos=Nptos, rmin=rmin, rmax=rmax,
             met=met, Rtol=Rtol, Atol=Atol, zeroP_val=zeroP_val)
        return rspan, sol, R_star


def NDSol_EGBHstars_R_to_L(R_star, beta, argB, c1, edo, Nptos=1000, rmin=1e-2, rmax=1e2, pR=1e-8,
                           met='RK45', Rtol=1e-6, Atol=1e-6, info=False):
    """
    Solves the EGBH star system by:
    1. Solving the exterior solution from R_star outward.
    2. Solving the interior solution from R_star inward.
    3. Merging both solutions.
    """

    # 1. Solve exterior from R_star -> rmax
    rspan_ext, sol_ext = NDsolverEGBstars(beta, argB, c1=c1, Nptos=Nptos,
                                          rmin=R_star, rmax=rmax,
                                          met=met, Rtol=Rtol, Atol=Atol, info=info)
    
    if info:
        plt.plot(rspan_ext, sol_ext[2], ls='--', label='n')

    # 2. Extract state at R_star to use as initial conditions for interior
    V0 = [sol_ext[i][0] for i in range(len(sol_ext))]
    V0.append(pR)  # Add pressure = 0 at the boundary (surface)

    # 3. Solve interior from R_star -> rmin
    arg = [beta, c1, edo]
    rspan_int, sol_int = NDsolverEGBHstars(V0, arg, R_star, rmin, Nptos=Nptos,
                                           met=met, Rtol=Rtol, Atol=Atol)
    
    if info:
        plt.plot(rspan_int, sol_int[2], ls='--', label='n')
        plt.plot(rspan_int, sol_int[4], ls='--', label='n')
        plt.show()

    # print(rspan_ext[0], rspan_ext[-1], rspan_int[0], rspan_int[-1])
    # 4. Combine
    sol_ext = np.append(sol_ext, np.zeros((1, len(rspan_ext))), axis=0)  # Add pressure = 0 at the boundary (surface)
    rspan_full = np.concatenate((rspan_int[:0:-1], rspan_ext[::]))
    sol_full = np.concatenate((sol_int[:, :0:-1], sol_ext[:, ::]), axis=1)
    
    return rspan_full, sol_full

def NDSol_EGBHstars_L_to_R(V0, beta, c1, edo, Nptos=1000, rmin=1e-2, rmax=1e2,
             met='RK45', Rtol=1e-6, Atol=1e-6, zeroP_val=1e-8):
    """
    """
    arg = [beta, c1, edo]
    rspan, sol, R_star = NDsolverEGBHstars(V0, arg, rmin, rmax, Nptos=Nptos,
                                     met=met, Rtol=Rtol, Atol=Atol, zeroP_val=zeroP_val) 
    return rspan, sol, R_star


######
def NDsolverEGBHstars(V0, arg, rbound, rIter, Nptos=1000,
                      met='RK45', Rtol=1e-6, Atol=1e-6, zeroP_val=1e-8):
    
    rspan = np.linspace(rbound, rIter, Nptos)

    if rbound > rIter:
        print("We are integrating from R to the origin")
        sol = solve_ivp(SystEq_EGBHstars, [rbound, rIter], V0, t_eval=rspan,
                        args=[arg], method=met, rtol=Rtol, atol=Atol)
        return sol.t, sol.y
    
    # Define event: pressure crosses zero
    def Negpres(r, yV, arg):
        return yV[-1] - zeroP_val
    Negpres.terminal = True
    Negpres.direction = -1
 
    # First integration (interior solution)
    sol1 = solve_ivp(SystEq_EGBHstars, [rbound, rIter], V0, t_eval=rspan,
                     events=Negpres, args=[arg], method=met, rtol=Rtol, atol=Atol)

    R_star = sol1.t_events[0][0]
    V0_surface = [sol1.y_events[0][0][i] for i in range(sol1.y.shape[0] - 1)]  # Remove pressure
    # [sol1.y[i][-1] for i in range(sol1.y.shape[0] - 1)] 
    arg2 = [arg[0], arg[1]]

    # Continue with exterior solution
    # print(sol1.y[2][-1], sol1.y_events[0][0][2])
    print("Radio =====>  ", R_star)
    rspan2 = np.linspace(R_star, rIter, Nptos)
    sol2 = solve_ivp(SystEq_EGBstars, [R_star, rIter], V0_surface, t_eval=rspan2,
                     args=[arg2], method=met, rtol=Rtol, atol=Atol)
    
    # Combining
    sol2.y = np.append(sol2.y, np.zeros((1, len(sol2.t))), axis=0)  # Add pressure = 0 at the boundary (surface)
    r_combined = np.concatenate((sol1.t, sol2.t))
    y_combined = np.concatenate((sol1.y, sol2.y), axis=1)

    return r_combined, y_combined, R_star


def NDsolverEGBstars(beta, argB, c1=1, Nptos=1000, rmin=1e-2, rmax=1e2,
                     met='RK45', Rtol=1e-6, Atol=1e-6, info=False):
    """
    Solves the EGB star system using boundary conditions at rbound and integrating inwards and outwards.
    """

    # Unpack boundary condition parameters
    a0, a1, g1, rbound = argB

    # Ensure rmax is not smaller than rbound
    rmax = max(rmax, rbound)

    # Initial condition at rbound
    full_arg = [a0, a1, g1, beta, c1]
    nRmax, dnRmax, gMax, pMax, dpMax = BoundaConditionInf(rbound, full_arg)
    V0 = [nRmax, gMax, pMax, dnRmax, dpMax]

    # Argument for system equations
    eq_args = [beta, c1]

    if rmax > rbound:
        if info:
            print('Solving the system in two intervals')

        # Inward integration
        rspanIn = np.linspace(rbound, rmin, Nptos)
        solIn = solve_ivp(SystEq_EGBstars, [rbound, rmin], V0, t_eval=rspanIn,
                          args=[eq_args], method=met, rtol=Rtol, atol=Atol)

        # Outward integration
        rspanOut = np.linspace(rbound, rmax, Nptos)
        solOut = solve_ivp(SystEq_EGBstars, [rbound, rmax], V0, t_eval=rspanOut,
                           args=[eq_args], method=met, rtol=Rtol, atol=Atol)

        # Combine results
        rspan = np.concatenate((solIn.t[::-1], solOut.t))
        sol = np.concatenate((solIn.y[:, ::-1], solOut.y), axis=1)
    else:
        if info:
            print('Solving inward only')
        rspan = np.linspace(rbound, rmin, Nptos)
        sol_raw = solve_ivp(SystEq_EGBstars, [rbound, rmin], V0, t_eval=rspan,
                            args=[eq_args], method=met, rtol=Rtol, atol=Atol)
        rspan = sol_raw.t[::-1]
        sol = sol_raw.y[:, ::-1]

    return rspan, sol

############
def BoundaConditionInf(rmax, arg):
    """
    """
    a0, a1, g1, beta, c1 = arg
    
    nRmax = np.sqrt((9*a1**4*c1**4*g1 - 384*a0*a1*beta*c1**2*(a1**2*c1**2 - g1*(g1 - 5*rmax)) + 480*rmax**4*(g1 + rmax) - 4*a1**2*c1**2*g1*(384*beta + 9*g1**2\
        - 10*g1*rmax + 10*rmax**2))/rmax**5)/(4.*np.sqrt(30))
    dnRmax = (np.sqrt(0.8333333333333334)*(384*a0*a1*beta*c1**2*(a1**2*c1**2 - g1*(g1 - 4*rmax)) + g1*(-9*a1**4*c1**4 - 96*rmax**4 + 4*a1**2*c1**2*(384*beta\
        + 9*g1**2 - 8*g1*rmax + 6*rmax**2))))/(8.*rmax**6*np.sqrt((9*a1**4*c1**4*g1 - 384*a0*a1*beta*c1**2*(a1**2*c1**2 - g1*(g1 - 5*rmax)) + 480*rmax**4*(g1\
        + rmax) - 4*a1**2*c1**2*g1*(384*beta + 9*g1**2 - 10*g1*rmax + 10*rmax**2))/rmax**5))
    gMax = 4*np.sqrt(30)*np.sqrt(rmax**6/(192*a0*a1**3*beta*c1**4*(3*g1 - 10*rmax) - 3840*a0*a1*beta*c1**2*g1*rmax**2 + 480*rmax**5*(g1 + rmax)\
        + a1**4*c1**4*(-12*(160*beta + g1**2) + 5*g1*rmax) + 4*a1**2*c1**2*(12*g1**2*(42*beta + g1**2) - 15*g1*(64*beta + g1**2)*rmax + 20*g1**2*rmax**2\
        - 30*g1*rmax**3 + 60*rmax**4)))
    pMax = (a1**5*c1**4*(-32*g1 + 9*rmax) + 4*a1**3*c1**2*(80*beta*g1 + 37*g1**3 - 29*g1**2*rmax + 20*g1*rmax**2 - 10*rmax**3) + 8*a1*(-10*g1**5\
        + 6*beta*g1**2*(7*g1 - 6*rmax) + 12*g1**4*rmax - 15*g1**3*rmax**2 + 20*g1**2*rmax**3 - 30*g1*rmax**4 + 60*rmax**5) + 32*a0*(6*a1**4*beta*c1**4\
        + 15*rmax**6 + 3*a1**2*beta*c1**2*g1*(-7*g1 + 4*rmax) + beta*g1**2*(-10*g1**2 + 12*g1*rmax - 15*rmax**2)))/(480.*rmax**6)
    dpMax = (3*a1**5*c1**4*(64*g1 - 15*rmax) - 4*a1**3*c1**2*(480*beta*g1 + 222*g1**3 - 145*g1**2*rmax + 80*g1*rmax**2 - 30*rmax**3) - 192*a0*beta*(6*a1**4*c1**4\
        + a1**2*c1**2*g1*(-21*g1 + 10*rmax) - 10*g1**2*(g1**2 - g1*rmax + rmax**2)) + 96*a1*(3*beta*g1**2*(-7*g1 + 5*rmax) + 5*(g1 - rmax)*(g1**2 - g1*rmax\
        + rmax**2)*(g1**2 + g1*rmax + rmax**2)))/(480.*rmax**7)
    
    return nRmax, dnRmax, gMax, pMax, dpMax

def serOrig(rval, prof, arg, rminFit=1.0, rmaxFit=5.0, namefit="fphi", info=True):
    """
    
    """
    n0, beta, c1 = arg
    
    funcIt = {
        "fN" : lambda r, a0, am1: n0 + (n0*r**2*(92160*beta**2 - (61440*a0*beta**2*r)/am1 - (5760*beta*(am1**2 - 8*a0**2*beta)*r**2)/am1**2\
            + (48*a0*beta*(-231*am1**4 - 6784*a0**2*am1**2*beta + 49152*a0**4*beta**2)*r**3)/(3*am1**5 - 64*a0**2*am1**3*beta) + (5*(249*am1**6*c1**2\
            - 393216*a0**6*beta**3*c1**2 + 128*am1**4*beta*(-36 + 5*a0**2*c1**2) + 1024*a0**2*am1**2*beta**2*(96 + 41*a0**2*c1**2))*r**4)/(am1**4*(3*am1**2\
            - 64*a0**2*beta)*c1**2)))/(5.89824e6*beta**3),
        "fg" : lambda r, a0, am1: 1 - r**2/(32.*beta) - ((81*a0*am1**2 - 384*a0**3*beta)*r**3)/(6.*(192*am1**3*beta - 4096*a0**2*am1*beta**2)) + ((3*am1**4\
            + 44*a0**2*am1**2*beta - 512*a0**4*beta**2)*r**4)/(512.*am1**2*beta**2*(3*am1**2 - 64*a0**2*beta)) + (a0*(-5*am1**2 + 32*a0**2*beta)*(-27*am1**2\
            + 128*a0**2*beta)*r**5)/(4096.*am1**3*beta**2*(3*am1**2 - 64*a0**2*beta)) + ((1536*beta*(3*am1**3 - 64*a0**2*am1*beta)**2 + (-621*am1**8\
            + 20664*a0**2*am1**6*beta + 795648*a0**4*am1**4*beta**2 - 10289152*a0**6*am1**2*beta**3\
            + 25165824*a0**8*beta**4)*c1**2)*r**6)/(393216.*am1**4*beta**3*(3*am1**2 - 64*a0**2*beta)**2*c1**2),
        "fphi" :lambda r, a0, am1: a0 + am1/r + (3*am1*r)/(128.*beta) + (7*a0*am1**2*r**2)/(768*am1**2*beta - 16384*a0**2*beta**2) + (am1*(69*am1**2\
        - 128*a0**2*beta)*r**3)/(98304.*beta**2*(-3*am1**2 + 64*a0**2*beta))
    }
    
    ind = (rval >= rminFit) * (rval <= rmaxFit)
    rIt = rval[ind]
    profIt = prof[ind]
    popt, pcov = curve_fit(funcIt[namefit], rIt, profIt)
    
    if info:
        print(f"Fitting {namefit} with parameters: {popt}")
        print(f"Error: {np.sqrt(np.diag(pcov))}")
    
    fN0 = funcIt["fN"](rval, *popt)
    fg0 = funcIt["fg"](rval, *popt)
    fphi = funcIt["fphi"](rval, *popt)
    
    return fN0, fg0, fphi

def RicciScalar(r, yv):
    """
    """
    n, g, _, dn, _ = yv
    ddn = np.gradient(dn, r)
    dg = np.gradient(g, r)
    
    RicciSac = (2*(g**3*n + dg*r*(2*n + dn*r) - g*(n + r*(2*dn + ddn*r))))/(g**3*n*r**2)
    return RicciSac


def Kretschmann(r, yv):
    """
    """
    n, g, p, dn, dp = yv
    ddn = np.gradient(dn, r)
    dg = np.gradient(g, r)
    
    KretschmannVal = (4*g**2*(-1 + g**2)**2*n**2 + 8*(dn**2*g**2 + dg**2*n**2)*r**2 + 4*(dg*dn\
                      - ddn*g)**2*r**4)/(g**6*n**2*r**4)
    return KretschmannVal

def masaADMProf(rval, gprof):
    """
    """
    # Calculate the mass profile
    masa = rval * (1 - 1/gprof**2) / 2
    return masa


### ISCO

def ISCO(rval, metricprofs, values, phiprofs=None, check=True, zeroAsum=1e-4, tol=1e-14, newtonMet=False):
    """
    Calculate the ISCO radius based on the metric function N(r).
    The ISCO radius is defined as the radius where the effective potential has a local minimum.
    """
    # Unpack angular momentum value and coupling constant
    l, gs = values

    # Unpack the metric profile
    gprof, Nprof, dNprof = metricprofs
    fgprof = interp1d(rval, gprof, bounds_error=False, fill_value="extrapolate")

    ddNprof = np.gradient(dNprof, rval)
    fNprof = interp1d(rval, Nprof, bounds_error=False, fill_value="extrapolate")
    fdNprof = interp1d(rval, dNprof, bounds_error=False, fill_value="extrapolate")
    fddNprof = interp1d(rval, ddNprof, bounds_error=False, fill_value="extrapolate")

    # If phiprof is provided, we can use it to calculate the effective potential
    if phiprofs is not None:
        phiprof, dphiprof = phiprofs
        ddphiprof = np.gradient(dphiprof, rval)
        # Interpolate the profiles to ensure they are defined at all points
        fphi = interp1d(rval, phiprof, bounds_error=False, fill_value="extrapolate")
        dfphi = interp1d(rval, dphiprof, bounds_error=False, fill_value="extrapolate")
        ddfphi = interp1d(rval, ddphiprof, bounds_error=False, fill_value="extrapolate")
    else:
        [phiprof, dphiprof, ddphiprof] = [0*rval for _ in range(3)]  # Default to zero if not provided
        [fphi, dfphi, ddfphi] = [lambda r: 0*r for _ in range(3)]  # Default to zero if not provided

    # Find root of the effective potential
    fCond_Veff = lambda r: fdNprof(r)/fNprof(r) + (-l**2 + dfphi(r)*gs*(1 + gs*fphi(r))*r**3)/(gs*fphi(r)*(2 + gs*fphi(r))*r**3 + r*(l**2 + r**2))
    fddVeff = lambda r: (2*(fddNprof(r)*(l**2 + r**2 + gs*fphi(r)*(2 + gs*fphi(r))*r**2)**2 + fNprof(r)*(3*l**2 + gs*(ddfphi(r)*gs**3*fphi(r)**3*r**4\
                        + gs*fphi(r)**2*(3*l**2 + gs*(3*ddfphi(r) - 2*dfphi(r)**2*gs)*r**4) + r*(6*dfphi(r)*l**2 + dfphi(r)**2*gs*r*(l**2 - 2*r**2)\
                        + ddfphi(r)*r*(l**2 + r**2)) + fphi(r)*(6*l**2 + gs*r*(6*dfphi(r)*l**2 - 4*dfphi(r)**2*gs*r**3 + ddfphi(r)*r*(l**2\
                        + 3*r**2)))))))/(fgprof(r)**2*fNprof(r)*(1 + gs*fphi(r))**2*r**2*(l**2 + r**2 + gs*fphi(r)*(2 + gs*fphi(r))*r**2))
    
    # Numerically find the minimum of the effective potential
    ESq =  (Nprof**2*(l**2 + rval**2 + 2*gs*phiprof*rval**2 + gs**2*phiprof**2*rval**2))/rval**2
    Veff = (1 + l**2/((1 + gs*phiprof)**2*rval**2) + (gprof**2 - 1/(Nprof**2*(1 + gs*phiprof)**2))*ESq)/gprof**2
    try:
        # Initial guess for the radius
        in_index = abs(fCond_Veff(rval)) < zeroAsum  # Filter to find points where the condition is close to zero
        x0val = rval[in_index]  # Get the values of rval where the condition is close to zero
        #print(x0val)
        x0val = list(set(np.round(x0val, 0)) - {0})  # Remove duplicates and zero
        #print(x0val)
        if len(x0val) == 0:
            print("Warning: No initial guess found where the condition is close to zero.")
            in_index = np.argmin(np.abs(Veff))  # Find the index of the minimum effective potential
            x0val = [rval[in_index]]
        
        raiz, ddVeff = [], []
        for x0 in x0val:
            if newtonMet:
                raiztemp = newton(fCond_Veff, x0=x0, tol=tol, maxiter=3000)  # Use the Newton-Raphson method to find the root
            else:
                raiztemp = root(fCond_Veff, x0=x0, tol=tol, method='hybr').x[0]

            if raiztemp > rval[-1] or raiztemp < rval[0]:
                print(f"Warning: Root {raiztemp} is outside the range of rval. Skipping this root.")
                continue
            ddVefftemp = fddVeff(raiztemp)  # Check if the second derivative is positive (local minimum)
            raiz.append(raiztemp)
            ddVeff.append(ddVefftemp)
        #raiz = list(set(np.round(raiz, 1)))  # Remove duplicates
        #ddVeff = list(set(np.round(ddVeff, 1)))  # Remove duplicates

        if check:
            #plt.figure(figsize=(8,5))
            #plt.plot(rval, Veff, label='Veff', color='blue')
            #plt.grid(True)
            print(f"Raíz encontrada: x = {raiz}", "con Veff ->", x0val)
    except RuntimeError as e:
        print("El método de Newton no pudo converger a una raíz:")
        print(f"-> {e}")
        raiz, ddVeff = [None], [None]
    
    return raiz, ddVeff
    