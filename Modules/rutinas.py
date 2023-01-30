# Next are present a number of routines used in
# arXiv:

#####
# modules
#####
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, quad
from scipy.linalg import eig

#####
# Routines
#####


def cheb(op):
    '''
    Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
   
    In:
    op -> polynomial order
    N - size of diff matrix - op+1 where op is polynomial order.
   
    Out: 
    D -> Chebushev polynomial differentiation matrix.
    x -> Chebushev points
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

    D = Dtemp - Dii

    return D, x


def system(r, V, arg):
    """
    Spherically-symmetric Nonrelativistic ell-boson system Eq. 41 in
    Ref.:
    Notice that was used the rescale sigma(r) = r^l f(r) 

    In:
    r -> radial coordinate
    V -> a vector (with the values at r) the form [f, df, u, du], with "u" the shifted potential and 'd' indicate the
         first derivative of f, and u.
    arg -> ell value

    Out:
    A vector [df, ddf, du, ddu] with the values after an iteration step.
    """

    f, df, u, du = V
    ell = arg
    
    if r > 0:
        ddf = -f*u - 2*(ell+1)*df/r
        ddu = -r**(2*ell)*f**2 - 2*du/r
        return [df, ddf, du, ddu]
    else:
        ddf = u*f/(2*ell + 3)
        ddu = 0
        return [df, ddf, du, ddu]


# Shooting methodology using a bisection approax
def Freq_solveG(f_max, f_min, ell, rmax_, rmin_, nodos, u0=1.0, df0=0, du0=0,
                met='RK45', Rtol=1e-09, Atol=1e-10):
    """
    Orden de las variables U = w, dw, phi, dphi
    """
    print('Finding a profile with ', nodos, 'nodes')
    
    # IMPORTANT: it is not possible to find two event at same time
    # Events
    arg = [ell]
    def Sig(r, U, arg): return U[0]
    def dSig(r, U, arg): return U[1]
    Sig.direction = 0
    dSig.direction = 0
    
    while True:
        f0_ = (f_max+f_min)/2
        U0 = [f0_, df0, u0, du0]

        sol_ = solve_ivp(system, [rmin_, rmax_], U0, events=(Sig, dSig),
                         args=(arg), method=met,  rtol=Rtol, atol=Atol)
                          # 'DOP853''LSODA'

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
            print('Maximum precision achieved', f0_, 'radius', rTemp_)
            return f0_, rTemp_, sol_.t_events[0]


# ENERGIA
def energ(r, ell, sig, V0):
    sigF = interp1d(r, sig, kind='quadratic')
    Af = lambda r: r**(2*ell+1)*sigF(r)**2
    Bf = lambda r: r**(2*(ell+1))*sigF(r)**2

    rmin = r[0]
    rfin = r[-1]

    Av = V0 - quad(Af, rmin, rfin)[0]
    Bv = quad(Bf, rmin, rfin)[0]
    lam = (2*ell+1)/(Bv)
    en = (2*(2*ell+1)**2*Av)/Bv**2  # G^2m^5/hb^2

    return Av, Bv, lam, en


# CALCULANDO PERFILES
def profiles(nodos, ell, Nptos=2000, rmin=0, rmax=1000, fmax=3.1,
             fmin=0.0, u0=1, Rtol=1e-08, Atol=1e-09):
    """
    """
    f0, rTemp, posNodos = Freq_solveG(fmax, fmin, ell, rmax, rmin, nodos,
                                      Rtol=Rtol, Atol=Atol)

    # boundary conditions
    V0 = [f0, 0., u0, 0.]  # sigma, dsigma, u, du
    rspan = np.linspace(rmin, rTemp, Nptos)
    arg = [ell]

    sol2 = solve_ivp(system, [rmin, rTemp], V0, t_eval=rspan,
                     args=(arg), method='RK45', rtol=Rtol, atol=Atol)

    masa = sol2.y[3][-1]*sol2.t[-1]**2
    uR = masa/sol2.t[-1]
    c = sol2.y[2][-1]

    # calculando energía y normalizando (no se usa U(inf) pq la
    Av, Bv, lam, en = energ(sol2.t, ell, sol2.y[0], u0)

    print('masa ', masa, 'Uf ', uR)
    print('')
    print('E(U(inf)) ', c*lam**2)

    print(r'energía = ', en, r'$\lambda = $', lam)
    print('')
    #
    sDF = sol2.t**ell*sol2.y[0]
    sigF = interp1d(sol2.t/lam, sDF*lam**2, kind='quadratic')
    Nf = lambda r: r**2*sigF(r)**2/(2*ell+1)
    Nv = quad(Nf, rmin/lam, rTemp/lam)[0]

    print('Checking normalización, ', Nv, 1)

    return en, lam, Nv, sol2.t, sol2.y[0], sol2.y[1], sol2.y[2], sol2.y[3],\
           posNodos, ell


# EXTENDIENDO PERFILES
def extend(ell, rD, sD, dsD, uD, duD, Ext, Np=1000):

    # Parámetros
    def parametrosS(r, S):
        yr1, yr2 = S[-2], S[-1]
        r1, r2 = r[-2], r[-1]

        k = np.real(np.log(np.abs(yr1*r1/(yr2*(r2)))))
        s = np.exp(-k*r1)/r1
        C = yr1/s
        return C, k

    # funciones asíntóticas
    def sigm(r, C, k):
        y = C*np.exp(-k*r)/r
        dy = -(C*np.exp(-k*r)*(1+k*r))/r**2
        return y, dy

    def Up(r, A, B):
        y = A+B/r
        dy = -B/r**2
        return y, dy

    rad = np.linspace(rD[-1], rD[-1]+Ext, Np)

    # calculando parámetros
    Ap, k = parametrosS(rD, sD)
    AA, BB, lam, enT = energ(rD, ell, sD, uD[0])

    # uniendo datos
    sExt, dsExt = sigm(rad, Ap, k)
    uExt, duExt = Up(rad, AA, BB)
    print('checking ', AA, BB, k, enT)

    rDnew = np.concatenate((rD[:-1], rad), axis=None)
    sDnew = np.concatenate((sD[:-1], sExt), axis=None)
    dsDnew = np.concatenate((dsD[:-1], dsExt), axis=None)
    uDnew = np.concatenate((uD[:-1], uExt), axis=None)
    duDnew = np.concatenate((duD[:-1], duExt), axis=None)

    return rDnew, sDnew, dsDnew, uDnew, duDnew


def extend2(ell, rD, sD, dsD, uD, duD, Ext, Np=1000):

    # Parámetros
    def parametrosS2(r, S, En, ell, B):
        yr1 = S[-1]
        r1 = r[-1]
        sE = np.sqrt(abs(En))
        s = np.exp(-sE*r1)/(r1**(1+ell-B/(2*sE)))
        C = yr1/s
        return C

    # funciones asíntóticas
    def sigm(r, C1, k, ell, B):
        y = C1*np.exp(-k*r)/(r**(1+ell-B/(2*k)))
        # dy = -(C1*np.exp(-k*r)*(1+k*r))/r**2
        dy = C1*np.exp(-k*r)*r**(B/(2*k)-(2+ell))*(B-2*k**2*r-2*k*(1+ell))/(2*k)
        return y, dy

    def Up(r, A, B):
        y = A+B/r
        dy = -B/r**2
        return y, dy

    rad = np.linspace(rD[-1], rD[-1]+Ext, Np)

    # calculando parámetros
    # Ap = parametrosS(rD, sD)

    AA, BB, lam, enT = energ(rD, ell, sD, uD[0])
    Ap = parametrosS2(rD, sD, enT, ell, BB)

    # uniendo datos
    k = np.sqrt(abs(AA)) #np.sqrt(abs(enT)/2)  # E/2 pq E[G^2m^5/hb^2] y Ebarra tiene un 2
    sExt, dsExt = sigm(rad, Ap, k, ell, BB)
    uExt, duExt = Up(rad, AA, BB)
    print('checking ', AA, BB, k, enT)

    rDnew = np.concatenate((rD[:-1], rad), axis=None)
    sDnew = np.concatenate((sD[:-1], sExt), axis=None)
    dsDnew = np.concatenate((dsD[:-1], dsExt), axis=None)
    uDnew = np.concatenate((uD[:-1], uExt), axis=None)
    duDnew = np.concatenate((duD[:-1], duExt), axis=None)

    return rDnew, sDnew, dsDnew, uDnew, duDnew


# ESPECTRO
def espectro(L, N, fsN, fuN, ell):
    # calculando matriz de derivada
    D_chev, x_chev = cheb(N)

    # rescalando la distancia de [0, L] -> [-1, 1]
    # recordar x=2(r/L)-1
    
    x_cod = np.array([(-x_chev[i]+1)*L/2. for i in range(N+1)])
    
    #j = np.arange(N)
    #x_chev = np.cos(j*np.pi/(N-1)+np.pi)
    #x_cod = np.array([(x_chev[i]+1)*L/2. for i in range(N+1)])

    # check
    print('Comprobando reescalamiento ', x_cod[0] == 0., x_cod[-1] == L)

    # reescalando la matriz de derivada de D_[0, L] = D[-1, 1]/(L/2)
    D2_chev = np.dot(D_chev, D_chev)/((L/2)**2)
    # ignoramos primera (indice 0) y última fila, así como primera
    # y última columna
    D2i_chev = np.copy(D2_chev[1:N, 1:N])

    # creando matrices de fondo R0, U0
    x_cod2 = np.copy(x_cod[1:N])
    R0_chev = np.diag(fsN(x_cod2))
    U0_chev = np.diag(fuN(x_cod2)-(ell*(ell+1))/(x_cod2**2))

    # creando matriz derecha  SM #
    # SM_chev = np.zeros((2*(N-1), 2*(N-1)))  # , dtype=complex
    # ident = np.eye(N-1)
    # SM_chev[:N-1, : N-1] = -1*ident  # -1j*
    # SM_chev[N-1:, N-1:] = -1*ident  # -1j*
    SM_chev = -1*np.eye(2*(N-1), 2*(N-1))

    # creando el operador (matriz operador)
    OM_chev = np.zeros((2*(N-1), 2*(N-1)))  # f, c , dtype=complex

    # comprobando
    invD2i_chev = np.linalg.inv(D2i_chev)
    check = np.allclose(np.dot(D2i_chev, invD2i_chev), np.eye(N-1))  # check
    print('Comprobando la inversa de D2 ', check)

    temp1 = np.dot(invD2i_chev, R0_chev)
    temp2_chev = np.dot(R0_chev, temp1)

    OM_chev[:N-1, N-1:] = D2i_chev+U0_chev
    OM_chev[N-1:, :N-1] = D2i_chev+U0_chev-2*temp2_chev

    # Obtenemos los autovalores y los autovectores
    # derechos usando scipy.linalg.eig
    lEnig1, V1 = eig(OM_chev, SM_chev)

    # Comprobando que los autovalores y autovectores son la solución del
    # sistema Ax=Lx
    test = []
    for i in range(N-1):
        ntest = np.allclose(
                 OM_chev@V1[:, i]-(lEnig1[i]*np.dot(SM_chev, V1[:, i])),
                 np.zeros((2*(N-1)), dtype=complex)
                            )
        test.append(ntest)
    test = np.array(test)
    print('Comprobando que se cumple Ax=Lx ->', test)

    # Obteniendo la Lambda verdadera
    lEnigT1 = -1j*np.copy(lEnig1)  # Lambda = i lambda -> lambda=-i Lambda
    # Organizando de menor a mayor los autovalores
    lEnigF1 = np.copy(lEnigT1)
    ii = np.argsort(lEnigF1)
    lEnigF1 = lEnigF1[ii]
    VF1 = V1[:, ii]

    return lEnigF1, lEnigT1, VF1, x_chev  #, V1


# Nuevo reescalamiento
def espectroRN(N, fsN, fuN, Sc, ell):
    # calculando matriz de derivada
    D_chev, x_chev = cheb(N)

    # rescalando la distancia de  [1, -1] -> [inf, 0]
    # es importante extender las soluciones hasta xval[-1]
    x_cod = np.array([2*Sc*(1/(1-x_chev[i])-1/2) for i in range(N+1)])

    # check
    print('Comprobando reescalamiento ', x_cod[0], 'infinito', x_cod[-1] == 0)

    # reemplazando el primer valor de xval que es infinito por la última distancia +1
    x_cod[0] = x_cod[1]+1

    # reescalando la matriz de derivada de D_[0, L] = D[-1, 1]/(L/2)
    d2 = np.dot(D_chev, D_chev)
    # factor de re-reescalamiento
    fac_dxDr = np.diag(2*Sc/(Sc+x_cod)**2)
    dfac_dxDr = np.diag(-2/(x_cod+Sc))

    # fac_dxDr = 2*Sc/(Sc+x_cod)**2

    temp0 = np.dot(fac_dxDr, dfac_dxDr)
    temp1 = np.dot(temp0, D_chev)
    temp2 = np.dot(fac_dxDr, fac_dxDr)
    temp3 = np.dot(temp2, d2)
    D2_chev = temp1+temp3

    # ignoramos primera (indice 0) y última fila, así como primera
    # y última columna
    D2i_chev = np.copy(D2_chev[1:N, 1:N])

    # creando matrices de fondo R0, U0
    x_cod2 = np.copy(x_cod[1:N])  # removing the values r=infty, r=0
    R0_chev = np.diag(fsN(x_cod2))
    U0_chev = np.diag(fuN(x_cod2)-(ell*(ell+1))/(x_cod2**2))

    # creando matriz derecha  SM #
    # SM_chev = np.zeros((2*(N-1), 2*(N-1)))  # , dtype=complex
    # ident = np.eye(N-1)
    # SM_chev[:N-1, : N-1] = -1*ident  # -1j*
    # SM_chev[N-1:, N-1:] = -1*ident  # -1j*
    SM_chev = -1*np.eye(2*(N-1), 2*(N-1))

    # creando el operador (matriz operador)
    OM_chev = np.zeros((2*(N-1), 2*(N-1)))  # f, c , dtype=complex

    # comprobando
    invD2i_chev = np.linalg.inv(D2i_chev)
    check = np.allclose(np.dot(D2i_chev, invD2i_chev), np.eye(N-1))  # check
    print('Comprobando la inversa de D2 ', check)

    temp1 = np.dot(invD2i_chev, R0_chev)
    temp2_chev = np.dot(R0_chev, temp1)

    OM_chev[:N-1, N-1:] = D2i_chev+U0_chev
    OM_chev[N-1:, :N-1] = D2i_chev+U0_chev-2*temp2_chev

    # Obtenemos los autovalores y los autovectores
    # derechos usando scipy.linalg.eig
    lEnig1, V1 = eig(OM_chev, SM_chev)

    # Comprobando que los autovalores y autovectores son la solución del
    # sistema Ax=Lx
    test = []
    for i in range(N-1):
        ntest = np.allclose(
                 OM_chev@V1[:, i]-(lEnig1[i]*np.dot(SM_chev, V1[:, i])),
                 np.zeros((2*(N-1)), dtype=complex)
                            )
        test.append(ntest)
    test = np.array(test)
    print('Comprobando que se cumple Ax=Lx ->', test)

    # Obteniendo la Lambda verdadera
    lEnigT1 = 1j*np.copy(lEnig1)
    # Organizando de menor a mayor los autovalores
    ii = np.argsort(lEnigT1)
    lEnigF1 = np.copy(lEnigT1)[ii]
    VF1 = V1[:, ii]

    return lEnigF1, lEnigT1, VF1, x_chev


# old only for ell=0
def espectro2(L, N, fsN, fuN):
    # calculando matriz de derivada
    D, x = cheb(N)

    # rescalando la distancia de [0, L] -> [-1, 1]
    xval = np.array([(-x[i]+1)*L/2. for i in range(N+1)])

    # reescalando la matriz de derivada de D_[0, L] = D[-1, 1]/L
    D2 = np.dot(D, D)/(L/2)**2
    # ignoramos primera (indice 0) y última fila, así como primera
    # y última columna
    D2i = D2[1:N, 1:N]

    # creando matrices de fondo R0, U0
    R0 = np.diag(fsN(xval[1:N]))
    U0 = np.diag(fuN(xval[1:N]))

    # creando matriz derecha  SM #
    SM = np.zeros((3*(N-1), 3*(N-1)))  # , dtype=complex
    ident = np.eye(N-1)
    SM[N-1: 2*(N-1), : N-1] = -1*ident  # -1j*
    SM[2*(N-1):, N-1: 2*(N-1)] = 1*ident  # -1j*

    # creando el operador (matriz operador)
    OM = np.zeros((3*(N-1), 3*(N-1)))  # f, c , dtype=complex
    OM[: N-1, : N-1] = -2*R0
    OM[: N-1, 2*(N-1):] = D2i
    OM[N-1: 2*(N-1), N-1: 2*(N-1)] = D2i+U0
    OM[2*(N-1):, :N-1] = -D2i-U0
    OM[2*(N-1):, 2*(N-1):] = R0

    # Obtenemos los autovalores y los autovectores derechos
    # usando scipy.linalg.eig
    lEnig, V = eig(OM, SM)

    # Comprobando que los autovalores y autovectores son la solución
    # del sistema Ax=Lx
    test = []
    for i in range(N-1):
        ntest = np.allclose(
            OM@V[:, i]-(lEnig[i]*np.dot(SM, V[:, i])),
            np.zeros((3*(N-1)), dtype=complex)
                        )
        test.append(ntest)
    test = np.array(test)
    print('Comprobando que se cumple Ax=Lx ->', test)

    # Obteniendo la Lambda verdadera
    lEnigT = 1j*np.copy(lEnig)
    # Organizando de menor a mayor los autovalores
    ii = np.argsort(lEnigT)
    lEnigF = np.copy(lEnigT)[ii]
    VF = V[:, ii]

    # quitando los NaN
    iii = ~np.isnan(lEnigF)
    lEnigF = lEnigF[iii]
    VF = VF[:, iii]

    return lEnigF, lEnigT, VF, x, OM, SM
