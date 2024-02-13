# 28 NOV 2023

# LOADING MODULES
import matplotlib.pyplot as plt 
import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp, quad
from scipy.linalg import eig


# TOOLS
def progressbar(current_value, total_value, bar_lengh, progress_char): 
    """
    Barra de progreso
    """
    percentage = int((current_value/total_value)*100)                                                # Percent Completed Calculation 
    progress = int((bar_lengh * current_value ) / total_value)                                       # Progress Done Calculation 
    loadbar = "Progress: [{:{len}}]{}%".format(progress*progress_char,percentage, len = bar_lengh)    # Progress Bar String
    print(loadbar, end='\r')

def find_nearest(array, value):
    """
    Encontrando el valor más cercano
    """
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    #print(idx)
    return (array[idx], idx)

def DatSepara(data):
    """
    Para separar datos
    """
    ndatos = len(data)

    dats0, datEing, dataEingVect = [], [], []
    for i in range(ndatos):
        s0, Eing, EingVec = data[i][0], data[i][1], data[i][2]
        
        dats0.append(s0)
        datEing.append(Eing)
        dataEingVect.append(EingVec)

    return dats0, datEing, dataEingVect

# EIGENVALUES & EIGENVECTORES
def roundElem(dat, dec=10):
    if len(dat)==0:
        return None
    else:
        dat = list(dat)
        datRound = np.round(dat, decimals=dec)
        Roundreal = np.real(datRound)
        Roundimag = np.imag(datRound)
        setRoundrealP = set(np.abs(Roundreal))
        setRoundimagP = set(np.abs(Roundimag))
        return [np.sort(list(setRoundrealP), axis=None) , np.sort(list(setRoundimagP), axis=None)]

def Uniendo(datos0, datos):

    s0ref, eingref, einvectref = datos0
    s0mod, eingmod, einvectmod = datos

    dataSigma0, Autovalores, Autovect = [], [], []
    ndat = len(s0ref)
    for i in range(ndat):
        temp = s0ref[i]
        ind = np.array(s0mod)<temp
        if sum(ind)==0:
            Autovalores.append(eingref[i])
            Autovect.append(einvectref[i])
            dataSigma0.append(temp)
        else:
            ndat2 = len(s0mod)
            temp1, temp2, temp3 = [], [], []
            for j in range(ndat2):
                if ind[j]:
                    Autovalores.append(eingmod[j])
                    Autovect.append(einvectmod[j])
                    dataSigma0.append(s0mod[j])
                else:
                    temp1.append(eingmod[j])
                    temp2.append(einvectmod[j])
                    temp3.append(s0mod[j])
            Autovalores.append(eingref[i])
            Autovect.append(einvectref[i])
            dataSigma0.append(temp) 

            eingmod = temp1.copy()
            einvectmod = temp2.copy()
            s0mod = temp3.copy()
    return dataSigma0, Autovalores, Autovect

def sep(Auto_Valores, Auto_Funciones, datosProfiles, inf=False):
    """
    Separando los autovalores y autovectores 
    """
    Ncasos = range(len(Auto_Valores)) # numero de soluciones que se estudiarán
    dataR, dataI = [], []
    for i in Ncasos:
        fsNtmp = datosProfiles[i][0]
        s0 = fsNtmp(0)
        if inf:
            print('Autovalores Reales para rho_0=%7.6f'%s0)
        
        jj = np.real(Auto_Valores[i])!=0  # reales
        #gg = np.array([not(k) for k in jj])  # imag
        
        s0 = float(fsNtmp(0))
        dataR.append([s0, Auto_Valores[i][jj], Auto_Funciones[i][:, jj]])
        #dataI.append([s0, Auto_Valores[i][gg], Auto_Funciones[i][:, gg]])

        progressbar(i, len(Auto_Valores)-1, 30, '■')
    return dataR#, dataI

def test(datA, datB, s0, Rtol=1e-04, Atol=1e-04):
    """
    Si los dos son verdaderos la solución es la trivial
    Re(A) = 0
    Im(A) = 0
    """
    cte = s0/np.abs(datB[0]).max()
    if s0<=1e-04:
        r1 = np.isclose(datA[0][0]*cte, 0.+0.j)  # A = 0 rtol=Rtol*1e-03, atol=Atol*1e-03
        r2 = np.isclose(np.imag(datB[0][10])*cte, 0.) # imag B = 0 , rtol=Rtol*1e-03, atol=Atol*1e-03
    else:
        r1 = np.isclose(datA[0][0]*cte, 0.+0.j, rtol=Rtol, atol=Atol)  # A = 0
        r2 = np.isclose(np.imag(datB[0][10])*cte, 0., rtol=Rtol, atol=Atol)  # imag B = 0
    
    rf = r1*r2
    #print(s0, ' ', datA[0][0]*cte, ' ', r1)
    return rf

def choosing(EingVals, EingVects, data, Rtol=1e-04, Atol=1e-04, NoTriv=False):
    """
    FILTRO PARA AUTOVALORES NO-TRIVIAL
    """
    engval, engvec = [], []
    Ntot = len(EingVals)
    for i in range(Ntot):
        s0, rMax, Nptos = data[i]
        Nptos = int(Nptos)
        utilez = [i, Nptos, rMax]
        Nind = len(EingVals[i])
        engvalTemp, engvecTemp = [], []
        for j in range(Nind):
            EingTemp = EingVals[i][j]
            #print(s0)
            rval, datA, datB = VectoresAB(j, EingVects, EingVals, utilez, info=False)
            testR = test(datA, datB, s0, Rtol=Rtol, Atol=Atol)
            #print(testR)

            if NoTriv==False and testR==False:
                engvalTemp.append(EingTemp)
                engvecTemp.append([rval, datA, datB])
            elif NoTriv==True and testR==True:
                engvalTemp.append(EingTemp)
                engvecTemp.append([rval, datA, datB])

        engval.append(engvalTemp)
        engvec.append(engvecTemp)

        progressbar(i, Ntot-1, 30, '■')
    return engval, engvec


# INTEGRALES DEL FUNCIONAL
def Tf(datos):
    r, sigma, _ = datos
    r = r.astype('float64')
    sigma = sigma.astype('float64')

    rmin, rfin = r[0], r[-1]
    dsigma = np.gradient(sigma, r)
    dsigmaF = interp1d(r, dsigma, kind='quadratic') 
    
    intf = lambda r: r**2*dsigmaF(r)**2
    Tval = 4*np.pi*quad(intf, rmin, rfin)[0]
    return Tval

def Ff(datos):
    r, sigma, _ = datos
    r = r.astype('float64')
    sigma = sigma.astype('float64')

    rmin, rfin = r[0], r[-1]
    sigmaF = interp1d(r, sigma, kind='quadratic')
    
    intf = lambda r: r**2*sigmaF(r)**4
    Fval = 4*np.pi*quad(intf, rmin, rfin)[0]
    Fval = Fval/4.
    return Fval

def EnFuncion(datos, Lamb):
    Tfval = Tf(datos)
    Ffval = Ff(datos)
    
    Enf = -Tfval-Lamb*2*Ffval
    return Enf, Tfval, Ffval


# FONDO
def system(r, V, arg):
    """
    Sistema de ecuaciones de primer orden
    f = campo, u = potencial
    Variables: f, df, u, du = V
    """
    f, df, u, du = V
    Lamb, alf = arg

    #print(f, u)

    if np.abs(f)>20: #80
        return [0, 0, 0, 0]
    
    if r > 0:
        ddf = -f*u - 2*df/r + Lamb*f**3
        ddu = -alf*f**2 - 2*du/r
        return [df, ddf, du, ddu]
    else:
        ddf = (-u*f+Lamb*f**3)/3  # o más?
        ddu = 0
        return [df, ddf, du, ddu]

def NpuntosCho(s0, rMax):
    if s0>=3.5:
        Nptos = int(rMax/2)
    elif s0>=2.5:
        Nptos = int(rMax/4)
    elif s0>=1.5:
        Nptos = int(rMax/5)
    elif s0>=0.5:
        Nptos = int(rMax/6)
    elif s0>=0.05:
        Nptos = int(rMax/7)
    else:
        Nptos = int(rMax/8)
    
    return Nptos

def fondo(soluciones_Fondo, rtake = -160):
    """
    Obteniendo la configuración del fondo
    """

    # Fondo
    s0 = soluciones_Fondo[0]
    r0M = soluciones_Fondo[1]
    Ext = (s0*r0M)+7000
    Np = int(Ext/2)

    # Resolviendo
    en, Mas, rD, sD, dsD, uD, duD, cer0, LamV = profilesFromSolut(soluciones_Fondo) 
    
    # Extendiendo
    rDnew, sDnew, dsDnew, uDnew, duDnew, datosEquiv = extend(rD[:rtake], sD[:rtake], dsD[:rtake], uD[:rtake], duD[:rtake],
                                                                Ext, Np, inf=False)

    # interpolación de los datos
    fsN = interp1d(rDnew, sDnew, kind='quadratic') # quadratic
    fdsN = interp1d(rDnew, dsDnew, kind='linear')
    fuN = interp1d(rDnew, uDnew, kind='quadratic')
    fduN = interp1d(rDnew, duDnew, kind='quadratic')
    return  fsN, fdsN, fuN, fduN, rDnew[-1]

def convergInd(soluciones_Fondo, rangoN=None, rangoRmax=None, L=0, rtake = -160):
    """
    Estudio de la convergencia dado los datos para la solución
    """
    # Fondo
    fsN, fdsN, fuN, fduN, rfin = fondo(soluciones_Fondo, rtake = rtake)

    dist = [0.7, 0.8, 0.9, 1]
    ptosVal = [100, 250, 500, 750, 1000]
    
    _, alpha, Lamb, nodos = soluciones_Fondo[1], soluciones_Fondo[2], soluciones_Fondo[3], soluciones_Fondo[4]

    Auto_Real = []
    s0 = fsN(0)
    datFunc = [fsN, fuN]
    if rangoN==None:
      for rMax in rangoRmax:
        Nptos = NpuntosCho(s0, rMax)+ptosVal[3]
        print('Variando rMax', rMax, Nptos)
        util = [Nptos, rMax, Lamb, alpha, L]    
        
        lambd, _, _, _ = espectroL(datFunc, util, inf=False)
        ind = np.real(lambd)!=0
        lambdR = lambd[ind]
        Auto_Real.append([rMax, lambdR, Nptos])
    elif rangoRmax==None:
      for Nptos in rangoN:
        rMax = dist[2]*rfin
        print('Variando Nptos', rMax, Nptos)
        util = [Nptos, rMax, Lamb, alpha, L]   

        lambd, _, _, _ = espectroL(datFunc, util, inf=False)
        ind = np.real(lambd)!=0
        lambdR = lambd[ind]
        Auto_Real.append([rMax, lambdR, Nptos]) 

    return Auto_Real
    
def Freq_solveG(f_max, f_min, Lamb, alf, rmax_, rmin_, nodos, u0=1.0, df0=0, du0=0,
                met='RK45', Rtol=1e-09, Atol=1e-10):
    """
    SHOOTING PARA ENCONTRAR N nodos
    Orden de las variables U = w, dw, phi, dphi
    """
    print('Finding a profile with ', nodos, 'nodes')
    # IMPORTANT: it is not possible to find two event at same time
    # Events
    arg = [Lamb, alf]
    def Sig(r, U, arg): return U[0]
    def dSig(r, U, arg): return U[1]
    Sig.direction = 0
    dSig.direction = 0
    while True:
        f0_ = (f_max+f_min)/2
        U0 = [f0_, df0, u0, du0]
        sol_ = solve_ivp(system, [rmin_, rmax_], U0, events=(Sig, dSig),
                         args=(arg,), method=met,  rtol=Rtol, atol=Atol)
                          # 'DOP853''LSODA'
        #print(f0_)
        if sol_.t_events[1].size == nodos+1 and sol_.t_events[0].size == nodos:
            print('Found', f0_)
            return f0_, rmax_, sol_.t_events[0]
        elif sol_.t_events[1].size > nodos+1:  # una vez por nodo
            if sol_.t_events[0].size > nodos:  # dos veces por nodo
                f_min = f0_
                rTemp_ = sol_.t_events[0][-1]
            else:  # si pasa por cero más veces que 2*nodos se aumenta la w, sino se disminuye
                f_max = f0_
                rTemp_ = sol_.t_events[1][-1]
        elif sol_.t_events[1].size <= nodos+1:
            if sol_.t_events[0].size > nodos:  # dos veces por nodo
                f_min = f0_
                rTemp_ = sol_.t_events[0][-1]
            else:
                f_max = f0_
                rTemp_ = sol_.t_events[1][-1]

        # checking the lim freq.
        if abs((f_max-f_min)/2) <= 1e-15:
            print('Maxima precisión alcanzada', f0_, 'radio', rTemp_)
            return f0_, rTemp_, sol_.t_events[0]

def Freq_solveG2(f0, u_max, u_min, Lamb, alf, rmax_, rmin_, nodos, df0=0, du0=0,
                met='RK45', Rtol=1e-09, Atol=1e-10):
    """
    Orden de las variables U = w, dw, phi, dphi
    """
    print('Finding a profile with ', nodos, 'nodes')
    # IMPORTANT: it is not possible to find two event at same time
    # Events
    arg = [Lamb, alf]
    def Sig(r, U, arg): return U[0]
    def dSig(r, U, arg): return U[1]
    Sig.direction = 0
    dSig.direction = 0
    while True:
        u0_ = (u_max+u_min)/2
        U0 = [f0, df0, u0_, du0]
        sol_ = solve_ivp(system, [rmin_, rmax_], U0, events=(Sig, dSig),
                         args=(arg,), method=met,  rtol=Rtol, atol=Atol)
                          # 'DOP853''LSODA'
        #print(u0_, abs((u_max-u_min)/2))
        if sol_.t_events[1].size == nodos+1 and sol_.t_events[0].size == nodos:
            print('Found', u0_)
            return u0_, rmax_, sol_.t_events[0]
        elif sol_.t_events[1].size > nodos+1:  # una vez por nodo
            if sol_.t_events[0].size > nodos:  # dos veces por nodo
                u_max = u0_
                rTemp_ = sol_.t_events[0][-1]
            else:  # si pasa por cero más veces que 2*nodos se aumenta la w, sino se disminuye
                u_min = u0_
                rTemp_ = sol_.t_events[1][-1]
        elif sol_.t_events[1].size <= nodos+1:
            if sol_.t_events[0].size > nodos:  # dos veces por nodo
                u_max = u0_
                rTemp_ = sol_.t_events[0][-1]
            else:
                u_min = u0_
                rTemp_ = sol_.t_events[1][-1]

        # checking the lim freq.
        if abs((u_max-u_min)/2) <= 1e-14: #1e-14
            print('Maxima precisión alcanzada', u0_, 'radio', rTemp_)
            return u0_, rTemp_, sol_.t_events[0]

def energ(r, sig, V0):
    """
    Energia
    """
    sigF = interp1d(r, sig, kind='quadratic') 
    Af = lambda r: r*sigF(r)**2
    Bf = lambda r: r**2*sigF(r)**2

    rmin = r[0]
    rfin = r[-1]

    En = V0 - quad(Af, rmin, rfin)[0]  # energía: (2c^2 m)/Lambda
    Mas = quad(Bf, rmin, rfin)[0]  # masa: c*hb/(G*m*Lambda^(1/2))
    return En, Mas


# EXTENDIENDO PERFILES
def extend(rD, sD, dsD, uD, duD, Ext, Np=1000, inf=False, ptos=400):
    """
    Extendiendo solución del fondo
    """
    # Parámetros
    def parametrosS(r, S):
        yr1, yr2 = S[-2], S[-1]
        r1, r2 = r[-2], r[-1]

        k = np.real(np.log(np.abs(yr1*r1/(yr2*(r2)))))
        s = np.exp(-k*r1)/r1
        C = yr1/s
        return C, k

    #def parametrosS2(r, S, En, M, ptos):
    #    def expDec(x, c1):
    #        k = np.sqrt(-En)
    #        sig = c1*np.exp(-k*x)/x**(1-M/(2*k))
    #        return sig

    #    popt, pcov = curve_fit(expDec, r[-ptos:], S[-ptos:])
    #    return popt

    # funciones asíntóticas
    def sigm(r, C, k):
        y = C*np.exp(-k*r)/r
        dy = -(C*np.exp(-k*r)*(1+k*r))/r**2
        return y, dy

    #def sigm2(r, C, En, M):
    #    k = np.sqrt(-En)
    #    y = C*np.exp(-k*r)/r**(1-M/(2*k))
    #    dy = C*np.exp(-k*r)*r**(-2+M/(2*k))*(M-2*k*(1+k*r))/(2*k)
    #    return y, dy

    def Up(r, A, B):
        y = A+B/r
        dy = -B/r**2
        return y, dy

    rad = np.linspace(rD[-1], rD[-1]+Ext, Np)

    # calculando parámetros
    En, Mas = energ(rD, sD, uD[0])
    Ap, k = parametrosS(rD, sD)
    #Ap = parametrosS2(rD, sD, En, Mas, ptos=ptos)
    
    # uniendo datos
    sExt, dsExt = sigm(rad, Ap, k)
    #sExt, dsExt = sigm2(rad, Ap, En, Mas)
    uExt, duExt = Up(rad, En, Mas)

    rDnew = np.concatenate((rD[:-1], rad), axis=None)
    sDnew = np.concatenate((sD[:-1], sExt), axis=None)
    dsDnew = np.concatenate((dsD[:-1], dsExt), axis=None)
    uDnew = np.concatenate((uD[:-1], uExt), axis=None)
    duDnew = np.concatenate((duD[:-1], duExt), axis=None)

    fsN = interp1d(rDnew, sDnew, kind='quadratic') # quadratic
    fprof = lambda x: x**2*fsN(x)**2
    masa = quad(fprof, rDnew[0], rDnew[-1])[0]
    
    # checking
    if inf:
        print('checking ')
        print('Energia: ', En, ' ', uExt[-1]) #, ' ', k**2)
        print('Masa: ', Mas,  ' ', masa)

    return rDnew, sDnew, dsDnew, uDnew, duDnew, [masa, En, sD[0]]

def profiles(nodos, Lam, U0, Nptos=2000, rmin=0, rmax=1000, fmax=3.1,
             fmin=0.0, Rtol=1e-13, Atol=1e-15, alf=1, met='DOP853'):
    """
    Calculando el perfil
    """
    
    f0, rTemp, posNodos = Freq_solveG(fmax, fmin, Lam, alf, rmax, rmin, nodos, met=met, u0=U0,
                                  Rtol=Rtol, Atol=Atol)

    # boundary conditions
    V0 = [f0, 0., U0, 0.]  # sigma, dsigma, u, du
    rspan = np.linspace(rmin, rTemp, Nptos)
    arg = [Lam, alf]

    sol2 = solve_ivp(system, [rmin, rTemp], V0, t_eval=rspan,
                     args=(arg,), method=met, rtol=Rtol, atol=Atol)

    Ec = sol2.y[2][-1]  # energía u = E - Uf
    #masa = -(sol2.y[2][-1]-Ec)*sol2.t[-1]  # M = -Uf*r
    

    # calculando energía y masa por la integral
    En, Mas = energ(sol2.t, sol2.y[0], U0) 

    #print(r'masa=', masa, r' masaInt= ', Mas)
    print(r'masa=', Mas)
    print('')
    print(r'energía= ', Ec, r'energíaInt= ', En)
    print('')
    
    return En, Mas, sol2.t, sol2.y[0], sol2.y[1], sol2.y[2], sol2.y[3],\
           posNodos, Lam


def profiles2(nodos, Lam, f0, umin, umax, Nptos=2000, rmin=0, rmax=1000, Rtol=1e-13, Atol=1e-15, alf=1, met='DOP853'):
    """
    """
    
    U0, rTemp, posNodos = Freq_solveG2(f0, umax, umin, Lam, alf, rmax, rmin, nodos,
                                       met=met, Rtol=Rtol, Atol=Atol)

    # boundary conditions
    V0 = [f0, 0., U0, 0.]  # sigma, dsigma, u, du
    rspan = np.linspace(rmin, rTemp, Nptos)
    arg = [Lam, alf]

    sol2 = solve_ivp(system, [rmin, rTemp], V0, t_eval=rspan,
                     args=(arg,), method=met, rtol=Rtol, atol=Atol)

    Ec = sol2.y[2][-1]  # energía u = E - Uf
    #masa = -(sol2.y[2][-1]-Ec)*sol2.t[-1]  # M = -Uf*r
    
    # calculando energía y masa por la integral
    En, Mas = energ(sol2.t, sol2.y[0], U0) 

    #print(r'masa=', masa, r' masaInt= ', Mas)
    print(r'masa=', Mas)
    print('')
    print(r'energía= ', Ec, r'energíaInt= ', En)
    print('')
    
    return En, Mas, sol2.t, sol2.y[0], sol2.y[1], sol2.y[2], sol2.y[3],\
           posNodos, Lam

# Usando una solución
def profilesFromSolut(datos, rmin=0, Nptos=2000, inf=False):
    """
    """
    
    f0, rTemp, alf, Lam, nodos, posNodos, met, Rtol, Atol, U0 = datos

    # boundary conditions
    V0 = [f0, 0., U0, 0.]  # sigma, dsigma, u, du
    rspan = np.linspace(rmin, rTemp, Nptos)
    arg = [Lam, alf]

    sol2 = solve_ivp(system, [rmin, rTemp], V0, t_eval=rspan,
                     args=(arg,), method=met, rtol=Rtol, atol=Atol)

    Ec = sol2.y[2][-1]  # energía u = E - Uf
    #masa = -(sol2.y[2][-1]-Ec)*sol2.t[-1]  # M = -Uf*r
    

    # calculando energía y masa por la integral
    En, Mas = energ(sol2.t, sol2.y[0], U0) 

    if inf:
        print(r'masa=', Mas)
        print('')
        print(r'energía= ', Ec, r'energíaInt= ', En)
        print('')
    
    return En, Mas, sol2.t, sol2.y[0], sol2.y[1], sol2.y[2], sol2.y[3],\
           posNodos, Lam


# ESPECTRAL
# Chebushev polynomial differentiation matrix
def cheb(op):
    '''Chebushev polynomial differentiation matrix.
       Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
       N - size of diff matrix - op+1 where op is polynomial order.
    '''

    N = op+1
    # creating the Chebyshev points
    j = np.arange(N)
    x = np.cos(j*np.pi/(N-1))

    #  off-diagonal entries -> Dij
    Dtemp = np.ones(N)
    Dtemp[0], Dtemp[N-1] = 2.0, 2.0
    Dtemp *= (-1.0)**j
    Dtemp = Dtemp.reshape(N, 1)
    Dtemp = np.dot(Dtemp, (1.0/Dtemp).T)

    X = np.tile(x.reshape(N, 1), (1, N))
    dX = X - X.T  # distance difference (dX is a matrix)

    Dtemp = Dtemp/(dX+np.eye(N))  # eye Return a N+1 array with ones on the diagonal and zeros elsewhere.

    # diagonal entries
    Dii = np.diag(Dtemp.sum(axis=1))  # sum by the row, and construct a diagonal array

    # Dn
    D = Dtemp - Dii

    return D, x    

def espectro(datFunc, util, inf=False):
    """
    In: 
    fsN, fuN = [datFunc]
    Nptos, rMax, Lamb, alpha = [util]

    where: 
    fsN -> extended scalar field profiled
    fuN -> extended U_eff potential
    Nptos -> number of Chebishev points
    rMax  -> out border of the \ell-boson star
    alpha = 1
    Lamb -> 0, 1, -1

    Out:

    lEnigF1 -> eingevalues 
    lEnigT1 -> eingevalues
    VF1 -> eingefilds 
    x_chev  -> Chebishev points
    """
    # print(util)
    Nptos, rMax, Lamb, alpha = util
    fsN, fuN = datFunc

    ###########################################################
    ### Computing the Diagonal Matrix: DNSq, Sigma0, U0
    ### (N-1)x(N-1)
    #########################################################
    # calculando matriz de derivada
    D_chev, x_chev = cheb(Nptos)
    
    # rescalando la distancia de [0, L] -> [-1, 1]. Recordar x=2(r/L)-1
    r_dis = np.array([(-x_chev[i]+1)*rMax/2. for i in range(Nptos+1)])
    
    plt.plot(r_dis, fsN(r_dis))
    plt.xlim(0, 40)
    plt.savefig('foo%3.2f.png'%fsN(0), bbox_inches='tight')
    # plt.xscale('log')
    # plt.show()
    plt.close()
    
    if inf:
        print('Comprobando reescalamiento ', r_dis[0] == 0., r_dis[-1] == rMax)
    
    # Utilez
    r_dis2 = np.copy(r_dis[1:Nptos])
    D2_chev = np.dot(D_chev, D_chev)/((rMax/2)**2)  # Calculando D^2. Rescalando D_[0, L] = D[-1, 1]/(L/2)
    
    ## Matriz DNSq
    D2i_chev = np.copy(D2_chev[1:Nptos, 1:Nptos])  # ignoramos primera (indice 0) y última fila, así como primera y última columna
    
    ## Matriz Inv DNSq
    temp = np.copy(D2i_chev)
    TrianJInv = np.linalg.inv(temp)  # TrianJInv = (DSq)^-1
    
    ## Matriz Sigma0
    Sigma0 = np.diag(fsN(r_dis2))  # sigma_{1}^{0}
    
    ## Matriz U
    U0 = np.diag(fuN(r_dis2))  # potencial u0 = E-\triangle^{-1}(|sigma_{1}^{0}|^{2})

    # comprobando
    if inf:
        print('Comprobaciones')
        print('Comprobando dimensiones del bloque Sigma0 ', Sigma0.shape==((Nptos-1), (Nptos-1)))
        print('Comprobando dimensiones de la matriz U ', U0.shape==((Nptos-1), (Nptos-1)))
        check = np.allclose(np.dot(temp, TrianJInv), np.eye((Nptos-1)))  # check
        print('Comprobando la inversa de D2 ', check)

    ##################################
    ## Creando la matriz general
    ## 2(N-1) x 2(N-1)
    ################################
    num =  2  # número de filas
    rows, cols = (Nptos-1), (Nptos-1)
    OM_chev = np.zeros((num*rows, num*cols))

    # Llenando la matriz
    HopDisc = D2i_chev + U0 - Lamb*np.dot(Sigma0, Sigma0)
    row, col = 0, 1
    OM_chev[row*rows:(row+1)*rows, col*cols:(col+1)*cols] = HopDisc
   
    row, col = 1, 0
    temp1 = np.dot(TrianJInv, Sigma0)
    temp2 = np.dot(Sigma0, temp1)
    HopDisc2 = D2i_chev  + U0 - 2*alpha*temp2 - 3*Lamb*np.dot(Sigma0, Sigma0)
    OM_chev[row*rows:(row+1)*rows, col*cols:(col+1)*cols] = HopDisc2

    # Obtenemos los autovalores y los autovectores
    # derechos usando scipy.linalg.eig
    lEnig1, V1 = eig(OM_chev)

    # Comprobando que los autovalores y autovectores son la solución del
    # sistema Ax=Lx
    test = []
    for i in range(Nptos-1):
        ntest = np.allclose(
                 OM_chev@V1[:, i]-(lEnig1[i]*V1[:, i]),
                 np.zeros((num*rows), dtype=complex)
                            )
        test.append(ntest)
    test = np.array(test).sum()
    if inf:
        print('Comprobando que se cumple Ax=Lx ->', test, Nptos-1)

    # Obteniendo la Lambda verdadera
    lEnigT1 = 1j*np.copy(lEnig1)  # lEnig1 = -i Lambda -> Lambda = i lEnig1
    # Organizando de menor a mayor los autovalores
    lEnigF1 = np.copy(lEnigT1)
    ii = np.argsort(lEnigF1)
    lEnigF1 = lEnigF1[ii]
    VF1 = V1[:, ii]

    return lEnigF1, lEnigT1, VF1, x_chev  #, V1


## Con L
def espectroL(datFunc, util, inf=False):
    """
    In: 
    fsN, fuN = [datFunc]
    Nptos, rMax, Lamb, alpha, L = [util]

    where: 
    fsN -> extended scalar field profiled
    fuN -> extended U_eff potential
    Nptos -> number of Chebishev points
    rMax  -> out border of the \ell-boson star
    alpha = 1
    Lamb -> 0, 1, -1

    Out:

    lEnigF1 -> eingevalues 
    lEnigT1 -> eingevalues
    VF1 -> eingefilds 
    x_chev  -> Chebishev points
    """
    # print(util)
    Nptos, rMax, Lamb, alpha, L = util
    fsN, fuN = datFunc

    ###########################################################
    ### Computing the Diagonal Matrix: DNSq, Sigma0, U0
    ### (N-1)x(N-1)
    #########################################################
    # calculando matriz de derivada
    D_chev, x_chev = cheb(Nptos)
    
    # rescalando la distancia de [0, L] -> [-1, 1]. Recordar x=2(r/L)-1
    r_dis = np.array([(-x_chev[i]+1)*rMax/2. for i in range(Nptos+1)])
    
    plt.plot(r_dis, fsN(r_dis))
    plt.xlim(0, 40)
    plt.savefig('foo%3.2f.png'%fsN(0), bbox_inches='tight')
    # plt.xscale('log')
    # plt.show()
    plt.close()
    
    if inf:
        print('Comprobando reescalamiento ', r_dis[0] == 0., r_dis[-1] == rMax)
    
    # Utilez
    r_dis2 = np.copy(r_dis[1:Nptos])
    D2_chev = np.dot(D_chev, D_chev)/((rMax/2)**2)  # Calculando D^2. Rescalando D_[0, L] = D[-1, 1]/(L/2)

    ## Matriz DNSq
    D2i_chev = np.copy(D2_chev[1:Nptos, 1:Nptos])  # ignoramos primera (indice 0) y última fila, así como primera y última columna

    ## Matriz Ueff
    Rmatriz = np.diag(1/r_dis2**2)  # Rm=1/r^2
    Lmatriz = L*(L+1)*Rmatriz  # Lm=L(L+1)*Rm
    U0 = np.diag(fuN(r_dis2))  # potencial u0 = E-\triangle^{-1}(|sigma_{1}^{0}|^{2})
    Ueff = U0-Lmatriz
    
    ## Matriz Inv
    temp = np.copy(D2i_chev)-Lmatriz
    TrianJInv = np.linalg.inv(temp)  # TrianJInv = (DSq-L(L+1)/r^2)^-1
    
    ## Matriz Sigma0
    Sigma0 = np.diag(fsN(r_dis2))  # sigma_{1}^{0}

    # comprobando
    if inf:
        print('Comprobaciones')
        print('Comprobando dimensiones del bloque Sigma0 ', Sigma0.shape==((Nptos-1), (Nptos-1)))
        print('Comprobando dimensiones de la matriz U ', Ueff.shape==((Nptos-1), (Nptos-1)))
        check = np.allclose(np.dot(temp, TrianJInv), np.eye((Nptos-1)))  # check
        print('Comprobando la inversa de D2 ', check)

    ##################################
    ## Creando la matriz general
    ## 2(N-1) x 2(N-1)
    ################################
    num =  2  # número de filas
    rows, cols = (Nptos-1), (Nptos-1)
    OM_chev = np.zeros((num*rows, num*cols))

    # Llenando la matriz
    HopDisc = D2i_chev + Ueff - Lamb*np.dot(Sigma0, Sigma0)
    row, col = 0, 1
    OM_chev[row*rows:(row+1)*rows, col*cols:(col+1)*cols] = HopDisc
   
    row, col = 1, 0
    temp1 = np.dot(TrianJInv, Sigma0)
    temp2 = np.dot(Sigma0, temp1)
    HopDisc2 = D2i_chev  + Ueff - 2*alpha*temp2 - 3*Lamb*np.dot(Sigma0, Sigma0)
    OM_chev[row*rows:(row+1)*rows, col*cols:(col+1)*cols] = HopDisc2

    # Obtenemos los autovalores y los autovectores
    # derechos usando scipy.linalg.eig
    lEnig1, V1 = eig(OM_chev)

    # Comprobando que los autovalores y autovectores son la solución del
    # sistema Ax=Lx
    test = []
    for i in range(Nptos-1):
        ntest = np.allclose(
                 OM_chev@V1[:, i]-(lEnig1[i]*V1[:, i]),
                 np.zeros((num*rows), dtype=complex)
                            )
        test.append(ntest)
    test = np.array(test).sum()
    if inf:
        print('Comprobando que se cumple Ax=Lx ->', test, Nptos-1)

    # Obteniendo la Lambda verdadera
    lEnigT1 = 1j*np.copy(lEnig1)  # lEnig1 = -i Lambda -> Lambda = i lEnig1
    # Organizando de menor a mayor los autovalores
    lEnigF1 = np.copy(lEnigT1)
    ii = np.argsort(lEnigF1)
    lEnigF1 = lEnigF1[ii]
    VF1 = V1[:, ii]

    return lEnigF1, lEnigT1, VF1, x_chev  #, V1


####################
# Organizando #####
###################

def Organ(datos, Rtol=1e-02, Atol=1e-03):
    """
    """
    R_row, i = Ref_row(datos)  # fila de referencia
    datos_T = datos.copy()

    Nrow = len(datos_T.index)
    for ind in range(Nrow):
        if ind == i:  # saltando cuando coincida con la fila referencial
            continue
        datos_T = Organ_row(ind, R_row, datos_T, Rtol, Atol)

    return datos_T


# definiendo la fila que se usará como referente
def Ref_row(datos):
    """
    """

    Nrow = len(datos.index)
    for i in range(Nrow):
        R_row = datos.iloc[[i]]
        if True not in np.array(R_row.isnull()):
            print('The referential row is %3d' % i)
            return R_row, i

def Organ_row(ind, R_row, datos_T, Rtol, Atol):
    """
    """
    for i in R_row.columns[1:]:  # quitando el label de la primera columna
        # tomando el valor de la columna ref
        valSupI = np.imag(R_row[i])[0]
        valSupR = np.real(R_row[i])[0]

        # print(valSupR)
        # print(valSupI)
        # print(' ')

        # tomando la fila a organizar
        tempFrame = datos_T.iloc[[ind]]
        tempArrayI = np.imag(tempFrame)[0]
        tempArrayR = np.real(tempFrame)[0]

        # print(tempArrayR)
        # print(tempArrayI)
        # print(' ')

        # creando filtro de comparación
        compI = np.array([np.isclose(i, valSupI, rtol=Rtol, atol=Atol) for i
                          in tempArrayI])
        compR = np.array([np.isclose(i, valSupR, rtol=Rtol, atol=Atol) for i
                          in tempArrayR])
        filt = compI*compR

        # print(compI)
        # print(compR)
        # print(' ')

        if True in filt:
            # actualizo los datos
            j, = np.where(filt == True)  # encuentro el índice del q es cercano
            val1 = (np.array(tempFrame)[0])[filt]  # el valor que intercambiaré
            if (len(val1) > 1):  # revisando q no existan dos iguales
                val1 = val1[0]  # quedandome con el primero
            val2 = datos_T.loc[ind, i]  # el valor que moveré
            # intercambio los valores
            datos_T.loc[ind, i] = val1
            datos_T.loc[ind, j] = val2

    return datos_T

def menor(dat): 
    dif = []
    ii = 0
    for i in dat:
        dif.append([ii, abs(0-i)])
        ii += 1
    
    dif = np.array(dif)
    
    temp0 = np.copy(dif)
    temp = []
    while True:
        menor = min(temp0[:, 1])
        ind = list(np.where(temp0[:, 1] == menor)[0])
        for i in ind:
            temp.append(list(temp0[i]))
        
        temp0 = np.delete(temp0, ind, axis=0) # eliminandolos
        
        if len(temp0)==0:
            break
    
    temp = np.array(temp)
    ind = temp[:, 0]
    return ind

def VectoresAB(autoV, Auto_Funciones, Auto_Valores, utilez, info=True):
    
    k, Nptos, rMax = utilez
    vect = Auto_Funciones[k][:, autoV]
    Auto_a = vect[:(Nptos-1)]
    Auto_b = vect[(Nptos-1):]
    #Auto_a = vect[:Nptos]
    #Auto_b = vect[Nptos:]

    if info:
        print(len(vect), len(Auto_a), len(Auto_b), Nptos-1)
        print('Autovalor estudiado ', Auto_Valores[k][autoV])

    _, x_chev = cheb(Nptos)
    r_dis = np.array([(-x_chev[i]+1)*rMax/2. for i in range(Nptos+1)])

    datA, datB = [], []
    c1 = 1
    for i in range(c1):
        tempA = Auto_a[i*(Nptos-1):(i+1)*(Nptos-1)]
        tempB = Auto_b[i*(Nptos-1):(i+1)*(Nptos-1)]

        datA.append(tempA/r_dis[1:Nptos])
        datB.append(tempB/r_dis[1:Nptos])

    return r_dis, datA, datB