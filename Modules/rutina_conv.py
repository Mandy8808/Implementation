import numpy as np
import pandas as pd


# Organizando
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
