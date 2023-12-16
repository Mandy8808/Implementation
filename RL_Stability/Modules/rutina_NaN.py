cont2 = 1
yy = datos_Autov.copy()

while True:
    # creando un sub dataframe con las columnas con NaN
    filt = (yy.isnull() == True).any()  # viendo quien es NaN
    sub_df = yy.loc[: , filt]

    # encontrando indices de los NaN
    temp1 = np.sum(sub_df.isnull(), axis=1)
    indTemp = temp1[temp1 != 0].index

    # organizando un primer caso
    stop_Acum = 0
    for ind in indTemp:
        yy, stop = Organizando(ind, sub_df, yy)
        stop_Acum += stop

    if stop_Acum == len(indTemp):
        print('Quedó organizado')
        datos_Autov_Org = yy.copy()
        break

    if (cont2%300==0):
        print('Hay un problema')
        break

    cont2 += 1



datos_Autov_Org



def Organizando(ind, sub_datos, datos_T):
    """
    ind -> me da el índice de la fila del NaN
    datos_T -> toda la tabla de datos
    sub_datos -> dataframe con las columnas que tienen NaN
    """
    temp = sub_datos.iloc[[ind-1, ind]]
    Num_NaN = len(temp.columns)
    count, stop = 0, 0
    for i in temp.columns:
        valSupI =  np.imag(temp[i])[0] # tomando el valor arriba del NaN
        valSupR =  np.real(temp[i])[0] # tomando el valor arriba del NaN
        tempFrame = datos_T.iloc[[ind]]  # tomando la fila con NaN para encontrar el cercano
        tempArrayI = np.imag(tempFrame)[0]
        tempArrayR = np.real(tempFrame)[0]

        compI = np.array([np.isclose(i, valSupI, rtol=1e-04, atol=1e-06) for i in tempArrayI])
        compR = np.array([np.isclose(i, valSupR, rtol=1e-04, atol=1e-06) for i in tempArrayR])
        filt = compI*compR

        if True in filt:
            # actualizo los datos
            j, = np.where(filt==True)  # encuentro el índice del q es cercano
            val1 = (np.array(tempFrame)[0])[filt]
            datos_T.loc[ind, i] = val1  # NaN -> val1
            datos_T.loc[ind, j] = np.nan
            #print(valSupR)
        else:
            count += 1
            print('Ya quedaron organizados %2d de %2d'%(count, Num_NaN))
            if count == Num_NaN:
                print('Ya está organizado la fila %2d'%ind)
                stop = 1

    return datos_T, stop
