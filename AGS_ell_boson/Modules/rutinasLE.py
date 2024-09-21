#####
# Module for the linear stability
###
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, quad
from scipy.linalg import eig


##########
# Tools
#########

###
# Chebushev polynomial differentiation matrix
###

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


#####
# Numerical Resolution of the background solution 
####

# system equations
def system(r, V, arg):
    """
    Sistema de ecuaciones de primer orden
    f = campo, u = potencial
    Variables: f, df, u, du = V
    """
    f, df, u, du = V
    ell = arg
    if r > 0:
        ddf = -f*u - 2*(ell+1)*df/r
        ddu = -r**(2*ell)*f**2 - 2*du/r
        return [df, ddf, du, ddu]
    else:
        ddf = -u*f/(2*ell + 3)  # o +
        ddu = -r**(2*ell)*f**2/3 # 0
        return [df, ddf, du, ddu]


# SHOOTING PARA ENCONTRAR N nodos
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
            print('Maxima precisión alcanzada', f0_, 'radio', rTemp_)
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


#############
# ESPECTRO
############

def espectro(datFunc, util):

    L, Nptos, rMax = util
    fsN, fuN = datFunc

    # calculando matriz de derivada
    D_chev, x_chev = cheb(Nptos)
    # rescalando la distancia de [0, L] -> [-1, 1]. Recordar x=2(r/L)-1
    r_dis = np.array([(-x_chev[i]+1)*rMax/2. for i in range(Nptos+1)])
    
    # check
    print('Comprobando reescalamiento ', r_dis[0] == 0., r_dis[-1] == rMax)

    ##### CREANDO OPERADORES MATRICIALES ######
    # Utilez
    r_dis2 = np.copy(r_dis[1:Nptos])
    D2_chev = np.dot(D_chev, D_chev)/((rMax/2)**2)  # Calculando D^2. Rescalando D_[0, L] = D[-1, 1]/(L/2)
    D2i_chev = np.copy(D2_chev[1:Nptos, 1:Nptos])  # ignoramos primera (indice 0) y última fila, así como primera y última columna
    
    # Matriz Sigma0
    Sigma0 = np.diag(fsN(r_dis2))  # sigma_{1}^{0}
    # Matriz U0
    U0matriz = np.diag(fuN(r_dis2))  # potencial u0 = E-\triangle^{-1}(|sigma_{1}^{0}|^{2})
    # Matriz R
    Rmatriz = np.diag(1/r_dis2**2)  # Rm=1/r^2
    # Matriz L
    Lmatriz = L*(L+1)*Rmatriz  # Lm=L(L+1)*Rm
    # Matriz H
    Hmatriz = D2i_chev+U0matriz-Lmatriz
    # Matriz Ueff
    Ueff = -U0matriz+2*Rmatriz

    # Creando el operador TrianInv = [D^2-L(L+1)/r^2]^{-1}
    temp = D2i_chev-Lmatriz
    TrianInv = np.linalg.inv(temp)
    # comprobando
    check = np.allclose(np.dot(temp, TrianInv), np.eye(Nptos-1))  # check
    print('Comprobando la inversa de D2 ', check)

    # Creando la matriz general (matriz operador)
    num =  6  # número de ecuaciones, Dim(OM_chev) = num*(Nptos-1) x num*(Nptos-1)
    OM_chev = np.zeros((num*(Nptos-1), num*(Nptos-1)))

    # Llenando la matriz
    row, col = 0, 2
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = D2i_chev-Ueff-Lmatriz

    row, col = 0, 3
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Lmatriz

    row, col = 1, 2
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Rmatriz

    row, col = 1, 3
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    row, col = 2, 0
    temp1 = np.dot(TrianInv, Sigma0)
    temp2 = np.dot(Sigma0, temp1)
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = D2i_chev-Ueff-2*temp2-Lmatriz

    row, col = 2, 1
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Lmatriz

    row, col = 3, 0
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Rmatriz

    row, col = 3, 1
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    row, col = 4, 5
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    row, col = 5, 4
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    # Obtenemos los autovalores y los autovectores
    # derechos usando scipy.linalg.eig
    lEnig1, V1 = eig(OM_chev)

    # Comprobando que los autovalores y autovectores son la solución del
    # sistema Ax=Lx
    test = []
    for i in range(Nptos-1):
        ntest = np.allclose(
                 OM_chev@V1[:, i]-(lEnig1[i]*V1[:, i]),
                 np.zeros((num*(Nptos-1)), dtype=complex)
                            )
        test.append(ntest)
    test = np.array(test)
    print('Comprobando que se cumple Ax=Lx ->', test)

    # col = 2
    # print(OM_chev[:1, col*(Nptos-1): (col+1)*(Nptos-1)])

    # Obteniendo la Lambda verdadera
    lEnigT1 = -1j*np.copy(lEnig1)  # Lambda = i lambda -> lambda=-i Lambda
    # Organizando de menor a mayor los autovalores
    lEnigF1 = np.copy(lEnigT1)
    ii = np.argsort(lEnigF1)
    lEnigF1 = lEnigF1[ii]
    VF1 = V1[:, ii]

    return lEnigF1, lEnigT1, VF1, x_chev  #, V1


## sin usar las ultimas dos ecuaciones
def espectro2(datFunc, util):

    L, Nptos, rMax = util
    fsN, fuN = datFunc

    # calculando matriz de derivada
    D_chev, x_chev = cheb(Nptos)
    # rescalando la distancia de [0, L] -> [-1, 1]. Recordar x=2(r/L)-1
    r_dis = np.array([(-x_chev[i]+1)*rMax/2. for i in range(Nptos+1)])
    
    # check
    print('Comprobando reescalamiento ', r_dis[0] == 0., r_dis[-1] == rMax)

    ##### CREANDO OPERADORES MATRICIALES ######
    # Utilez
    r_dis2 = np.copy(r_dis[1:Nptos])
    D2_chev = np.dot(D_chev, D_chev)/((rMax/2)**2)  # Calculando D^2. Rescalando D_[0, L] = D[-1, 1]/(L/2)
    D2i_chev = np.copy(D2_chev[1:Nptos, 1:Nptos])  # ignoramos primera (indice 0) y última fila, así como primera y última columna
    
    # Matriz Sigma0
    Sigma0 = np.diag(fsN(r_dis2))  # sigma_{1}^{0}
    # Matriz U0
    U0matriz = np.diag(fuN(r_dis2))  # potencial u0 = E-\triangle^{-1}(|sigma_{1}^{0}|^{2})
    # Matriz R
    Rmatriz = np.diag(1/r_dis2**2)  # Rm=1/r^2
    # Matriz L
    Lmatriz = L*(L+1)*Rmatriz  # Lm=L(L+1)*Rm
    # Matriz H
    Hmatriz = D2i_chev+U0matriz-Lmatriz
    # Matriz Ueff
    Ueff = -U0matriz+2*Rmatriz

    # Creando el operador TrianInv = [D^2-L(L+1)/r^2]^{-1}
    temp = D2i_chev-Lmatriz
    TrianInv = np.linalg.inv(temp)
    # comprobando
    check = np.allclose(np.dot(temp, TrianInv), np.eye(Nptos-1))  # check
    print('Comprobando la inversa de D2 ', check)

    # Creando la matriz general (matriz operador)
    num =  4  # número de ecuaciones, Dim(OM_chev) = num*(Nptos-1) x num*(Nptos-1)
    OM_chev = np.zeros((num*(Nptos-1), num*(Nptos-1)))

    # Llenando la matriz
    row, col = 0, 2
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = D2i_chev-Ueff-Lmatriz

    row, col = 0, 3
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Lmatriz

    row, col = 1, 2
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Rmatriz

    row, col = 1, 3
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    row, col = 2, 0
    temp1 = np.dot(TrianInv, Sigma0)
    temp2 = np.dot(Sigma0, temp1)
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = D2i_chev-Ueff-2*temp2-Lmatriz

    row, col = 2, 1
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Lmatriz

    row, col = 3, 0
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Rmatriz

    row, col = 3, 1
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    # row, col = 4, 5
    # OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    # row, col = 5, 4
    # OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    # Obtenemos los autovalores y los autovectores
    # derechos usando scipy.linalg.eig
    lEnig1, V1 = eig(OM_chev)

    # Comprobando que los autovalores y autovectores son la solución del
    # sistema Ax=Lx
    test = []
    for i in range(Nptos-1):
        ntest = np.allclose(
                 OM_chev@V1[:, i]-(lEnig1[i]*V1[:, i]),
                 np.zeros((num*(Nptos-1)), dtype=complex)
                            )
        test.append(ntest)
    test = np.array(test)
    print('Comprobando que se cumple Ax=Lx ->', test)

    # col = 2
    # print(OM_chev[:1, col*(Nptos-1): (col+1)*(Nptos-1)])

    # Obteniendo la Lambda verdadera
    lEnigT1 = -1j*np.copy(lEnig1)  # Lambda = i lambda -> lambda=-i Lambda
    # Organizando de menor a mayor los autovalores
    lEnigF1 = np.copy(lEnigT1)
    ii = np.argsort(lEnigF1)
    lEnigF1 = lEnigF1[ii]
    VF1 = V1[:, ii]

    return lEnigF1, lEnigT1, VF1, x_chev  #, V1

##  las ultimas dos ecuaciones
def espectro3(datFunc, util):

    L, Nptos, rMax = util
    fsN, fuN = datFunc

    # calculando matriz de derivada
    D_chev, x_chev = cheb(Nptos)
    # rescalando la distancia de [0, L] -> [-1, 1]. Recordar x=2(r/L)-1
    r_dis = np.array([(-x_chev[i]+1)*rMax/2. for i in range(Nptos+1)])
    
    # check
    print('Comprobando reescalamiento ', r_dis[0] == 0., r_dis[-1] == rMax)

    ##### CREANDO OPERADORES MATRICIALES ######
    # Utilez
    r_dis2 = np.copy(r_dis[1:Nptos])
    D2_chev = np.dot(D_chev, D_chev)/((rMax/2)**2)  # Calculando D^2. Rescalando D_[0, L] = D[-1, 1]/(L/2)
    D2i_chev = np.copy(D2_chev[1:Nptos, 1:Nptos])  # ignoramos primera (indice 0) y última fila, así como primera y última columna
    
    # Matriz Sigma0
    Sigma0 = np.diag(fsN(r_dis2))  # sigma_{1}^{0}
    # Matriz U0
    U0matriz = np.diag(fuN(r_dis2))  # potencial u0 = E-\triangle^{-1}(|sigma_{1}^{0}|^{2})
    # Matriz R
    Rmatriz = np.diag(1/r_dis2**2)  # Rm=1/r^2
    # Matriz L
    Lmatriz = L*(L+1)*Rmatriz  # Lm=L(L+1)*Rm
    # Matriz H
    Hmatriz = D2i_chev+U0matriz-Lmatriz
    # Matriz Ueff
    Ueff = -U0matriz+2*Rmatriz

    # Creando el operador TrianInv = [D^2-L(L+1)/r^2]^{-1}
    temp = D2i_chev-Lmatriz
    TrianInv = np.linalg.inv(temp)
    # comprobando
    check = np.allclose(np.dot(temp, TrianInv), np.eye(Nptos-1))  # check
    print('Comprobando la inversa de D2 ', check)

    # Creando la matriz general (matriz operador)
    num =  2  # número de ecuaciones, Dim(OM_chev) = num*(Nptos-1) x num*(Nptos-1)
    OM_chev = np.zeros((num*(Nptos-1), num*(Nptos-1)))

    # Llenando la matriz
    #row, col = 0, 2
    #OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = D2i_chev-Ueff-Lmatriz

    #row, col = 0, 3
    #OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Lmatriz

    #row, col = 1, 2
    #OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Rmatriz

    #row, col = 1, 3
    #OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    #row, col = 2, 0
    #temp1 = np.dot(TrianInv, Sigma0)
    #temp2 = np.dot(Sigma0, temp1)
    #OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = D2i_chev-Ueff-2*temp2-Lmatriz

    #row, col = 2, 1
    #OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Lmatriz

    #row, col = 3, 0
    #OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = 2*Rmatriz

    #row, col = 3, 1
    #OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    row, col = 0, 1
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    row, col = 1, 0
    OM_chev[row*(Nptos-1):(row+1)*(Nptos-1), col*(Nptos-1): (col+1)*(Nptos-1)] = Hmatriz

    # Obtenemos los autovalores y los autovectores
    # derechos usando scipy.linalg.eig
    lEnig1, V1 = eig(OM_chev)

    # Comprobando que los autovalores y autovectores son la solución del
    # sistema Ax=Lx
    test = []
    for i in range(Nptos-1):
        ntest = np.allclose(
                 OM_chev@V1[:, i]-(lEnig1[i]*V1[:, i]),
                 np.zeros((num*(Nptos-1)), dtype=complex)
                            )
        test.append(ntest)
    test = np.array(test)
    print('Comprobando que se cumple Ax=Lx ->', test)

    # col = 2
    # print(OM_chev[:1, col*(Nptos-1): (col+1)*(Nptos-1)])

    # Obteniendo la Lambda verdadera
    lEnigT1 = -1j*np.copy(lEnig1)  # Lambda = i lambda -> lambda=-i Lambda
    # Organizando de menor a mayor los autovalores
    lEnigF1 = np.copy(lEnigT1)
    ii = np.argsort(lEnigF1)
    lEnigF1 = lEnigF1[ii]
    VF1 = V1[:, ii]

    return lEnigF1, lEnigT1, VF1, x_chev  #, V1

######
# Organizando
#######

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