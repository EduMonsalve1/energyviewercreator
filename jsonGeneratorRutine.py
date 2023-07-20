from pathlib import Path
import pandas as pd
import numpy as np
import zipfile
import random
import pyproj
import shutil
import gzip
import json
import os

# Ubicación de las carpetas con los archivos
path_corr1,path_corr2,path_corr3=False,False,False
print("\nBienvenido a la función de procesamiento de Base de Datos y Resultados PLP")
Choice = input("Desea correr un caso ya creado o un caso nuevo con formato '.zip' ( 1 / 2 )?\n")
print()
if Choice == '1':
    while not path_corr1:
        path_data=input("Ingrese la ruta de la carpeta en donde se ubican los archivos CSV: \n")
        print()
        aux=input(f"\nSe ha ingresado la siguiente ruta: '{path_data}',\nde ser correcta escriba 'y', en caso contrario escriba 'n' y vuelva a agregar el path\n")
        print()
        if aux == "y":
            path_corr1=True
if Choice == '2':
    while not path_corr1:
        archivos_zip = [archivo for archivo in os.listdir('.') if archivo.endswith('.zip')]
        Cou = 1
        print("Se obtiene la lista de todos los archivos PLP '.zip' disponibles en el directorio actual")
        for zips in archivos_zip:
            print(f"({Cou})",zips)
            Cou += 1
        print()
        num_zip = input("Seleccione el caso PLP que desea procesar: \n")
        print()
        Zip = archivos_zip[int(num_zip)-1]
        ruta_actual = os.getcwd()
        Zip_path = os.path.join(ruta_actual,Zip)
        aux = input(f"\nSe ha ingresado la siguiente ruta: '{Zip_path}',\nde ser correcta escriba 'y', en caso contrario escriba 'n' y vuelva a agregar el path\n")
        print()
        if aux == "y":
            path_corr1=True
    while not path_corr2:
        Folder_IPLP = input("Ingrese la ruta de la carpeta de destino, donde se creará el nuevo caso PLP:\n")
        aux = input(f"\nSe ha ingresado la siguiente ruta: '{Folder_IPLP}',\nde ser correcta escriba 'y', en caso contrario escriba 'n' y vuelva a agregar el path\n")
        print()
        if aux == "y":
            path_corr2=True
    Folder, Zip_file = os.path.split(Zip_path)
    Zip_name = os.path.splitext(Zip_file)[0]
    # Ruta de destino para guardar los archivos descomprimidos
    path_data = os.path.join(Folder_IPLP,Zip_name)
    Destino_casos = os.path.basename(Folder_IPLP)
    print(f"  Se procede a descomprimir el archivo '{Zip_file}'")
    print(f"  La carpeta de destino de los archivos CSV es '{Destino_casos}'")
    print("  Se inicia la descompresión del archivo '.zip'")
    # Crear la carpeta de destino si no existe
    if not os.path.exists(path_data): os.makedirs(path_data)
    # Extraer todos los archivos del ZIP en la carpeta de destino
    with zipfile.ZipFile(Zip_path, 'r') as zip_ref: zip_ref.extractall(path_data)
    PLPs = ['plplin.csv.gz','plpbar.csv.gz','plpcen.csv.gz','plpemb.csv.gz']
    for plp in PLPs:
        #ruta_archivo_gz = os.path.join(ruta_sin_extension,plp)
        ruta_archivo_gz = os.path.join(path_data,plp)
        descomprimido = ruta_archivo_gz[:-3]  # Elimina la extensión .gz
        # Descomprimir el archivo .gz
        with gzip.open(ruta_archivo_gz, 'rb') as archivo_gz:
            with open(descomprimido, 'wb') as archivo_descomprimido:
                shutil.copyfileobj(archivo_gz, archivo_descomprimido)
        os.remove(ruta_archivo_gz)
    print(f"  Descompresión del archivo completada exitosamente\n")
    os.remove(Zip_path)
while not path_corr3:
    Folder_Json = input("Ingrese la ruta de la carpeta de destino, donde se enviará la carpeta con archivos Json:\n")
    aux = input(f"Se ha ingresado la siguiente ruta: '{Folder_Json}',\nde ser correcta escriba 'y', en caso contrario escriba 'n' y vuelva a agregar el path\n")
    print()
    if aux == "y":
        path_corr3=True
namedata = f"Json_{os.path.basename(path_data[5:])}"
print(f"A continuación, el nombre de la carpeta en donde se encontrarán\nlos archivos Json se llamará '{namedata}'")
print()
print("---------------------------------- Iniciando Carga de archivos-----------------------------------\n")

print("Este proceso puede demorar unos minutos dependiendo del tamaño de los archivos\n")

print("--------Cargando Archivo plpbar-------------\n")
plpbar=pd.read_csv(os.path.join(path_data,'plpbar.csv'))
plpbar.columns=["Hidro","time","TipoEtapa","id","BarName","CMgBar","DemBarP","DemBarE","PerBarP","PerBarE","BarRetP","BarRetE"]
plpbar['BarName']=plpbar['BarName'].str.replace(" ","")
plpbar["Hidro"] = plpbar["Hidro"].str.replace(" ", "")

indexbus=plpbar[['id','BarName']].drop_duplicates(keep="first").reset_index(drop=True)

print("--------Archivo plpbar cargado-------------\n")


print("--------Cargando Archivo de ubicaciones Ubibar-------------\n")
ubibar=pd.read_csv(os.path.join(path_data,'ubibar.csv'),sep=';',encoding='utf-8-sig')
ubibar=ubibar.drop('ID',axis=1)
ubibar['LATITUD']=ubibar['LATITUD'].apply(lambda x:x.replace(',','.')).apply(float)
ubibar['LONGITUD']=ubibar['LONGITUD'].apply(lambda x:x.replace(',','.')).apply(float)
ubibar.columns=["BarName","latitud","longitud"]
ubibar['BarName']=ubibar['BarName'].str.replace(" ","")
print("--------Archivo de ubicaciones Ubibar Cargado-------------\n")


print("--------Cargando Archivo plpcen-------------\n")
plpcen=pd.read_csv(os.path.join(path_data,'plpcen.csv'))
plpcen.columns=["Hidro","time","TipoEtapa","id","CenName","tipo","bus_id","BarName","CenQgen","CenPgen","CenEgen","CenInyP","CenInyE","CenRen","CenCVar","CenCostOp","CenPMax"]
plpcen['CenName']=plpcen["CenName"].str.replace(" ","")
plpcen=plpcen.drop(["CenEgen","CenInyP","CenInyE","CenRen","CenCostOp","CenPMax"],axis=1)
plpcen["Hidro"] = plpcen["Hidro"].str.replace(" ", "")
plpcen['tipo']="otros"

indexcen=plpcen[['id','CenName','tipo','bus_id']].drop_duplicates(keep="first").reset_index(drop=True)

print("--------Archivo plpcen cargado-------------\n")


print("--------Cargando Archivo centralesinfo-------------\n")
centralsinfo=pd.read_csv(os.path.join(path_data,'centralesinfo.csv'),sep=';',encoding='utf-8-sig')
centralsinfo.columns=['id','CenName','type','CVar','effinciency','bus_id','serie_hidro_gen','serie_hidro_ver','min_power','max_power',"VembIn","VembFin","VembMin","VembMax","cotaMínima"]

cols = ['min_power', 'max_power', 'effinciency', 'CVar', 'VembIn', 'VembFin', 'VembMin', 'VembMax', 'cotaMínima']
centralsinfo['CenName'] = centralsinfo["CenName"].str.replace(" ", "")
for col in cols:
    centralsinfo[col] = centralsinfo[col].replace(",", ".", regex=True)

hydric_adicional = pd.read_csv(os.path.join(path_data,'hydric_adicional.csv'),sep=";")

tiposcentrales=pd.read_csv(os.path.join(path_data,'centralestype.csv'),sep=';',encoding='utf-8-sig').rename(columns={'cen_name':'CenName'})
typecentrals=indexcen.merge(tiposcentrales,on='CenName')

for x in range(len(indexcen['id'])):
    tipo=typecentrals[typecentrals['CenName']==indexcen['CenName'][x]]['cen_type'].values
    
    if len(tipo)>0:
        plpcen.loc[plpcen['id'] == indexcen['id'][x], 'tipo'] = tipo[0]
	
print("--------Archivo centralesinfo cargado-------------\n")


print("--------Cargando Archivo plplin-------------\n")
plplin=pd.read_csv(os.path.join(path_data,'plplin.csv'))
# Cambiando los nombres de las columnas
plplin.columns=["Hidro","time","TipoEtapa","id","LinName","bus_a","bus_b","LinFluP","LinFluE","capacity","LinUso","LinPerP","LinPerE","LinPer2P","LinPer2E","LinITP","LinITE"]
plplin['LinName']=plplin['LinName'].str.replace(" ","")
plplin["Hidro"] = plplin["Hidro"].str.replace(" ", "")

indexlin=plplin[['id','LinName',"bus_a","bus_b"]].drop_duplicates(keep="first").reset_index(drop=True)

print("--------Archivo plplin Cargado-------------\n")


print("--------Cargando Archivo linesinfo-------------\n")
linesinfo=pd.read_csv(os.path.join(path_data,'linesinfo.csv'),sep=';',encoding='utf-8-sig')
linesinfo.columns=["id","LinName","bus_a","bus_b","max_flow_a_b","max_flow_b_a","voltage","r","x","segments","active"]
linesinfo['LinName']=linesinfo['LinName'].str.replace(" ","")
linesinfo['max_flow_a_b']=(linesinfo["max_flow_a_b"].apply(str)).apply(lambda x:x.replace(',','.')).apply(float)
linesinfo['max_flow_b_a']=(linesinfo['max_flow_b_a'].apply(str)).apply(lambda x:x.replace(',','.')).apply(float)
linesinfo['r']=(linesinfo['r'].apply(str)).apply(lambda x:x.replace(',','.')).apply(float)
linesinfo['x']=(linesinfo['x'].apply(str)).apply(lambda x:x.replace(',','.')).apply(float)

linesfinal=indexlin.drop(['id','bus_a','bus_b'],axis=1).merge(linesinfo,on='LinName')
linesfinal['id']=(linesfinal['id']).apply(int)

print("--------Archivo linesinfo Cargado-------------\n")

print("--------Cargando Archivo Reservoirs-------------\n")

reservoirs = pd.read_csv(os.path.join(path_data,'plpemb.csv'))
reservoirs.rename(columns={'Bloque': 'time', 'EmbNum': 'id', 'EmbNom': 'EmbName'}, inplace=True)
reservoirs['EmbName']=reservoirs['EmbName'].str.replace(" ","")
reservoirs['Hidro']=reservoirs['Hidro'].str.replace(" ","")

indexres = reservoirs[['id','EmbName']].drop_duplicates(keep="first").reset_index(drop=True)

junctionsinfo=centralsinfo[centralsinfo['type'].isin(["E",'S','R'])].reset_index(drop=True)
junctionsinfo['serie_hidro_gen']=(junctionsinfo['serie_hidro_gen'].apply(str)).apply(lambda x:x.replace(',','.')).apply(float)
junctionsinfo['serie_hidro_ver']=(junctionsinfo['serie_hidro_ver'].apply(str)).apply(lambda x:x.replace(',','.')).apply(float)

reservoirsinfo=centralsinfo[centralsinfo['type'].isin(["E"])].reset_index(drop=True)
reservoirsinfo.rename(columns={'CenName':'EmbName'}, inplace=True)

for i, emb_name in enumerate(reservoirsinfo['EmbName']):
    if emb_name in indexres['EmbName'].values:
        idx = indexres.index[indexres['EmbName'] == emb_name][0]
        reservoirsinfo.at[i, 'id'] = indexres.at[idx, 'id']

print("--------Archivo Reservoirs Cargado-------------\n")

print("--------Cargando Archivo indhor-------------\n")
indhor = pd.read_csv(os.path.join(path_data,'indhor.csv'),encoding='latin-1')
print("--------Archivo indhor Cargado-------------\n")

# Creando directorios

electricTopology=namedata+'/Topology/Electric'
hydricTopology=namedata+'/Topology/Hydric'

os.makedirs(electricTopology,exist_ok=True)
os.makedirs(hydricTopology,exist_ok=True)


hidrolist=plpbar['Hidro'].unique()
busscenariolist=[]
centralscenariolist=[]
linescenariolist=[]
reservoirscenariolist=[]
for hidronum in range(len(hidrolist)):
	# Creamos los directorios
	busscenario= namedata+f'/Scenarios/{hidronum+1}/Bus'
	centralscenario=namedata+f'/Scenarios/{hidronum+1}/Centrals'
	linescenario=namedata+f'/Scenarios/{hidronum+1}/Lines'
	reservoirscenario=namedata+f'/Scenarios/{hidronum+1}/Reservoirs'

	os.makedirs(busscenario,exist_ok=True)
	busscenariolist.append(busscenario)

	os.makedirs(centralscenario,exist_ok=True)
	centralscenariolist.append(centralscenario)

	os.makedirs(linescenario,exist_ok=True)
	linescenariolist.append(linescenario)

	os.makedirs(reservoirscenario,exist_ok=True)
	reservoirscenariolist.append(reservoirscenario)

marginal_cost_path=namedata+f'/Scenarios/Marginal_cost_percentil'
line_flow_percentil_path=namedata+f'/Scenarios/Flow_Line_percentil'
generation_sistem_path=namedata+f'/Scenarios/Generation_system'
os.makedirs(marginal_cost_path,exist_ok=True)
os.makedirs(line_flow_percentil_path,exist_ok=True)
os.makedirs(generation_sistem_path,exist_ok=True)
hydrofile = [x for x in range(1,len(hidrolist)+1)]

with open( namedata+'/Scenarios/hydrologies.json', 'w') as f:
  json.dump(hydrofile, f)


# Variables indicadoras de cantidades

# Número de horas de bloques temporales del proyecto
time=plplin['time'].max()

# Número de barras
nbus=len(indexbus['id'])
lbus=list(indexbus['id'])

# Número de generadores
ngen=len(indexcen['id'])

# Número de lineas
nlin=len(indexlin['id'])

# Número de Reservoirs
nres = len(reservoirs['EmbName'].unique())

# Mixed de Lineas Previcionales

filtered_lines = linesfinal[linesfinal.duplicated(subset=['bus_a', 'bus_b'], keep=False)]

groups = filtered_lines.groupby(['bus_a', 'bus_b'])['id']

parallel_lines = []
for _, group in groups:
    parallel_lines.append(group.values.tolist())
lnesfinal_aux = linesfinal.copy()
for sublist in parallel_lines:
    first_element = sublist[0]
    suma_maxab=0
    suma_maxba=0
    linesfinal.loc[linesfinal['id'] == first_element, 'LinName'] += "- mixed"
    for element in sublist:
        if linesfinal.loc[linesfinal['id'] == element, 'active'].values[0] == 1: 
            suma_maxab+=linesfinal.loc[linesfinal['id'] == element, 'max_flow_a_b'].values[0]
            suma_maxba+=linesfinal.loc[linesfinal['id'] == element, 'max_flow_b_a'].values[0]
            linesfinal.loc[linesfinal['id'] == element, 'active'] = 0
    linesfinal.loc[linesfinal['id'] == first_element, 'active'] = 1
    linesfinal.loc[linesfinal['id'] == first_element, 'max_flow_a_b']=suma_maxab
    linesfinal.loc[linesfinal['id'] == first_element, 'max_flow_b_a']=suma_maxba

# Crear un array con todos los primeros elementos de las listas en 'parallel_lines'
first_elements = np.array([x[0] for x in parallel_lines])
plplin_copy = plplin.copy()
# Añadir "- mixed" al final de los valores en 'LinName' donde 'id' hace match con el primer valor de cada lista en 'parallel_lines'
plplin_copy['LinName'] = np.where(plplin_copy['id'].isin(first_elements), plplin_copy['LinName'] + '- mixed', plplin_copy['LinName'])

plplin_copy = pd.merge(plplin_copy, lnesfinal_aux[['id', 'active']], on='id', how='left')

# Crear un nuevo dataframe donde cada fila es una lista en 'parallel_lines'
parallel_df = plplin_copy[plplin_copy['id'].isin([item for sublist in parallel_lines for item in sublist])].copy()

# Creamos una columna "parallel_id" que tenga el primer id de la línea paralela correspondiente para cada fila.
parallel_dict = {id_par: par[0] for par in parallel_lines for id_par in par}
parallel_df['parallel_id'] = parallel_df['id'].map(parallel_dict)

# Ahora podemos agrupar por 'Hidro', 'time' y 'parallel_id', y sumar 'LinFluP' y 'capacity' dentro de cada grupo.
grouped_df = parallel_df[parallel_df['active'] == 1].groupby(['Hidro', 'time', 'parallel_id'])[['LinFluP', 'capacity']].sum().reset_index()

# Primero, fusionamos 'parallel_df' con 'grouped_df'.
result = pd.merge(parallel_df, grouped_df, on=['Hidro', 'time', 'parallel_id'], how='left', suffixes=('', '_sum'))

# Ahora, reemplazamos los valores de 'LinFluP' y 'capacity' con los de 'LinFluP_sum' y 'capacity_sum' solo para las filas donde 'id' es igual a 'parallel_id'.
result.loc[result['id'] == result['parallel_id'], 'LinFluP'] = result['LinFluP_sum']
result.loc[result['id'] == result['parallel_id'], 'capacity'] = result['capacity_sum']

# Finalmente, eliminamos las columnas 'LinFluP_sum' y 'capacity_sum' ya que no las necesitamos más.
result = result.drop(columns=['LinFluP_sum', 'capacity_sum'])

# Si también quieres eliminar la columna 'parallel_id', puedes hacerlo así:
result = result.drop(columns=['parallel_id'])

# Asignar los valores calculados en 'result' a las filas correspondientes en 'plplin_copy'
plplin_copy.set_index(['Hidro', 'time', 'id'], inplace=True)
result.set_index(['Hidro', 'time', 'id'], inplace=True)

plplin_copy.update(result[['LinFluP', 'capacity']])

# Restablecer el índice
plplin_copy.reset_index(inplace=True)

plplin = plplin_copy

# Función generadora de latitudes y longitudes

def aleatory_direction():
    latitud=-random.uniform(10, 85)
    longitud=-random.uniform(10, 85)
    return latitud,longitud

def LatLon_To_XY(Lat,Lon):
  B = pyproj.Transformer.from_crs(4326,20049) #WGS84->EPSG:20049 (Chile 2021/UTM zone 19S)
  UTMx, UTMy = B.transform(Lat,Lon)
  return UTMx, UTMy

def XY_To_LatLon(x,y):
  B = pyproj.Transformer.from_crs(20049,4326)
  Lat, Lon = B.transform(x,y)
  return Lat, Lon

def valorXY(LatP, LonP, scale):
  A = LatLon_To_XY(LatP, LonP)
  X,Y = A[0]*scale, A[1]*scale
  return Y,X


print("A continuación se iniciará el proceso de creación de la carpeta Scenarios, este proceso en total puede llegar a tomar más de 1 hora 30 minutos\n")

print("Creando archivos de Bloques a Fechas")
indhor2=indhor.drop('Hora',axis=1).groupby(['Año','Mes'])
indhorlist=[]
for x in indhor2:
    indhorlist.append([str(x[1]['Bloque'].min()),str(x[1]['Bloque'].max()),str(x[0])])
with open( namedata+'/Scenarios/indhor.json', 'w') as f:
  json.dump(indhorlist, f)


print("Creando archivos generación por Sistema por Hidrología")

typegenlist=typecentrals.cen_type.unique()
for i,hydro in enumerate(hidrolist):
    print(hydro+" lista")
    dic_type_gen={}
    auxdf = plpcen[plpcen['Hidro']==hydro]
    auxdf=auxdf.groupby(['tipo','time'])['CenPgen'].sum().reset_index().groupby('tipo')
    for group in auxdf:
        tipo = group[0]
        df_tipo = group[1]
        dic_type_gen[tipo] = [row for row in df_tipo[['time', 'CenPgen']].to_dict(orient='records')]

    
    with open(generation_sistem_path+f'/generation_system_{i+1}.json', 'w') as f:
        json.dump(dic_type_gen, f)
	
print("Creando archivos para Grafico Percentiles Costo Marginal")
def percentilCM():
    datos_bar = plpbar[['Hidro', 'time','id', 'BarName', 'CMgBar']]
    lista_bar = datos_bar.BarName.unique()

    i=1
    for barra in lista_bar:
        print(f'Procesando datos de {barra} [{i}/{len(lista_bar)}]')
        data_barraTx = datos_bar.loc[(datos_bar.BarName == barra)]
        idbar=data_barraTx['id'].unique()[0]
        data_barraTx = data_barraTx[~(data_barraTx['Hidro'] == 'MEDIA')]
        Promedio = data_barraTx[['time','CMgBar']]
        xy =Promedio.groupby(['time']).mean()
        
        data_barraTx = data_barraTx.groupby(['time']).agg(perc0=('CMgBar', lambda x: x.quantile(0.0)),
                                                                perc20=(
                                                                    'CMgBar', lambda x: x.quantile(0.2)),
                                                                perc80=(
                                                                    'CMgBar', lambda x: x.quantile(0.8)),
                                                                perc100=('CMgBar', lambda x: x.quantile(1)))

        data_barraTx['promedio'] = xy
        data_barraTx = data_barraTx.assign(name=barra)
        data_barraTx = data_barraTx.assign(id=idbar)
        data_barraTx.reset_index(inplace=True)
        data_barraTx=data_barraTx[['id','time','name','perc0','perc20','perc80','perc100','promedio']]
        data_barraTx.to_json(marginal_cost_path+f"/bus_{idbar}.json",orient='records')
        i=i+1

percentilCM()

print("Creando archivos para Grafico Percentiles flujos de lineas de transmisión")
def percentilFL():
    datos_lineas=plplin[['id','Hidro', 'time', 'LinName', 'LinFluP', 'capacity']]
    lista_lineas = datos_lineas.LinName.unique()
    n_lineas = len(lista_lineas)
    i=1
    for linea in lista_lineas:
        print(f'Procesando datos de {linea} [{i}/{n_lineas}]')
        data_lineaTx = datos_lineas.loc[(datos_lineas.LinName == linea)]
        idlin=data_lineaTx['id'].unique()[0]
        data_lineaTx = data_lineaTx[~(data_lineaTx['Hidro'] == 'MEDIA')]
        fluMax = data_lineaTx[['time','capacity']]
        xy =-fluMax.groupby(['time']).max()
        data_lineaTx = data_lineaTx.groupby(['time']).agg(perc0=('LinFluP', lambda x: x.quantile(0.0)),
                                                                perc20=(
                                                                    'LinFluP', lambda x: x.quantile(0.2)),
                                                                perc80=(
                                                                    'LinFluP', lambda x: x.quantile(0.8)),
                                                                perc100=('LinFluP', lambda x: x.quantile(1)))

        data_lineaTx['Min'] = xy
        data_lineaTx['Max'] = -xy
        i = i+1
        data_lineaTx.reset_index(inplace=True)
        data_lineaTx = data_lineaTx.assign(id=idlin)
        data_lineaTx = data_lineaTx.assign(LinName = linea)
        data_lineaTx.to_json(line_flow_percentil_path+f"/line_{idlin}.json",orient='records')

percentilFL()


print("Creando Archivos Bus en Scenario \n")
# Bus contiene:
'''
		(*) id <int>: identificador de la barra 
		(*) time <int>: instante de registro
		(*) name <str>: nombre de la barra
		marginal_cost <float>: costo marginal, genera el gráfico de costo
					[USD/MWh]
		DemBarE <float>: construye el gráfico de demanda de Energía [MWh]
		DemBarP <float>: construye el gráfico de demanda de Potencia [MW]
		Value <float>: mismo valor que marginal_cost [MWh]
'''

def busscenariofunction(dfbusauxlist, pathbus):
    for x in range(nbus): 
        idbus = indexbus['id'][x]
        aux = pd.DataFrame({
            'id': idbus,
            'time': dfbusauxlist[x]['time'],
            'name': indexbus['BarName'][x],
            'marginal_cost': dfbusauxlist[x]['CMgBar'],
            'value': dfbusauxlist[x]['CMgBar'],
            'DemBarE': dfbusauxlist[x]['DemBarE'],
            'DemBarP': dfbusauxlist[x]['DemBarP'],
            'BarRetP': dfbusauxlist[x]['BarRetP']
        })
        aux.to_json(pathbus + f"/bus_{idbus}.json", orient='records')


for hidronum,hidroname in enumerate(hidrolist):
	
	dfbussauxx=plpbar.query(f"(Hidro=='{hidroname}')").reset_index()
	dfbuslist=[]
	for x in lbus:
		idaux=x
		dfbuslist.append(dfbussauxx[dfbussauxx.id==idaux].reset_index(drop=True))
	print(f"{((hidronum+1)/len(hidrolist))*100}% Completado")
	busscenariofunction(dfbuslist,busscenariolist[hidronum])

print("Archivos Bus en Scenario creados\n")

print("Creando Archivos Central en Scenario \n")
# Centrals contiene:
'''
		(*) id <int>: identificador del generador
		(*) time <int>: instante de registro
		(*) bus_id <int>: identificador de la barra a la que se conecta
		(*) name <str>: nombre del generador
		CenPgen <float>: energía generada en el instante time [MW]
		value <float>: mismo valor que CenPgen [MW]
		(?) CenCVar <unknown>: parámetro no identificado
		(?) CenQgen <unknown>: parámetro no identificado
        
'''
def centralscenariofunction(dfcenauxlist, cenpath):
    for x in range(ngen):
        if indexcen['bus_id'][x] == 0 or np.isnan(indexcen['bus_id'][x]):
            continue
        aux_df = pd.DataFrame({
            'id': indexcen['id'][x],
            'time': range(1, time + 1),
            'bus_id': int(indexcen['bus_id'][x]),
            'name': indexcen['CenName'][x],
            'CenPgen': dfcenauxlist[x]['CenPgen'] if len(dfcenauxlist[x]) > 0 else [0]*time,
            'value': dfcenauxlist[x]['CenPgen'] if len(dfcenauxlist[x]) > 0 else [0]*time,
            'CenCVar': dfcenauxlist[x]['CenCVar'] if len(dfcenauxlist[x]) > 0 else [0]*time,
            'CenQgen': dfcenauxlist[x]['CenQgen'] if len(dfcenauxlist[x]) > 0 else [0]*time,
        })
        aux_df.to_json(cenpath + f"/central_{indexcen['id'][x]}.json", orient='records')

for hidronum, hidroname in enumerate(hidrolist):
    dfcensauxx = plpcen.query(f"(Hidro=='{hidroname}')").reset_index()
    dfcenlist = [dfcensauxx[dfcensauxx.id == indexcen['id'][x]].reset_index(drop=True) for x in range(ngen)]
    print(f"{((hidronum + 1) / len(hidrolist)) * 100}% Completado")
    centralscenariofunction(dfcenlist, centralscenariolist[hidronum])

print("Archivos Central en Scenario creados \n")

print("Creando Archivos Lineas en Scenario \n")
'''

        (*) id <int>: identificador de la linea 
		(*) time <int>: instante de registro
		(*) bus_a <int>: identificador de la barra de origen
		(*) bus_b <int>: identificador de la barra de destino
		flow <float>: flujo en el instante time [MW]
		value <float>: mismo valor que flow [MW]
        
'''
# if not Path('linesscenariolist.pickle').is_file():
def linescenariofunction(dflinelist, linpath):
    for x in range(nlin):
        if linesfinal['active'][x] != 1:
            continue
        idaux = linesfinal['id'][x]
        bus_a_id = linesfinal['bus_a'][x]
        bus_b_id = linesfinal['bus_b'][x]
        name = linesfinal['LinName'][x]
        aux_df = pd.DataFrame({
            'id': idaux,
            'time': range(1, time + 1),
            'name': name,
            'bus_a': bus_a_id,
            'bus_b': bus_b_id,
            'flow': dflinelist[x]['LinFluP'],
            'value': dflinelist[x]['LinFluP'],
            'capacity': dflinelist[x]['capacity'],
        })
        aux_df.to_json(linpath + f"/line_{idaux}.json", orient='records')

for hidronum, hidroname in enumerate(hidrolist):
    dflinesaux = plplin.query(f"(Hidro=='{hidroname}')").reset_index()
    dflinelist = [dflinesaux[dflinesaux.id == linesfinal['id'][x]].reset_index(drop=True) for x in range(nlin)]
    print(f"{((hidronum + 1) / len(hidrolist)) * 100}% Completado")
    linescenariofunction(dflinelist, linescenariolist[hidronum])
print("Archivos Line en Scenario creados \n")

print("Creando Archivos Reservoirs en Scenario \n")

# Resevoirs contiene:
'''
		(*) time <int>: instante de registro
		(*) id <int>: identificador del embalse
		(*) junction_id <int>: identificador del canal al que se conecta
		(*) name <str>: nombre del embalse
		level <float>: nivel en el instante time
		value <float>: mismo valor que level
'''

def resscenariofunction(dfreslist, respath):
    for x in range(nres):
        idaux = indexres['id'][x]
        name = indexres['EmbName'][x]
        junction_id = junctionsinfo[junctionsinfo['CenName'] == name]['id'].values[0]
        aux_df = pd.DataFrame({
            'time': range(1, time + 1),
            'id': idaux,
            'junction_id': junction_id,
            'name': name,
            'level': (dfreslist[x]['EmbFac'] * dfreslist[x]['EmbVfin']) / 1000000,
            'value': (dfreslist[x]['EmbFac'] * dfreslist[x]['EmbVfin']) / 1000000,
        })
        aux_df.to_json(respath + f"/reservoir_{idaux}.json", orient='records')

for hidronum, hidroname in enumerate(hidrolist):
    dfresaux = reservoirs.query(f"(Hidro=='{hidroname}')").reset_index()
    dfreslist = [dfresaux[dfresaux.id == indexres['id'][x]].reset_index(drop=True) for x in range(nres)]
    print(f"{((hidronum + 1) / len(hidrolist)) * 100}% Completado")
    resscenariofunction(dfreslist, reservoirscenariolist[hidronum])
print("Archivos Reservoir en Scenario creados \n")



print("Ahora se procede a crear los Archivos correspondientes a la Topología y datos estáticos de los elementos")

print("Creando datos topologicos Bus")

ubibar[['latUTM','lonUTM']]=ubibar.apply(lambda row: valorXY(row['latitud'],row['longitud'],scale=0.00001),axis=1,result_type='expand')
dirdfbus=ubibar
# bus electric contiene:

'''   
		(*) id <int>: identificador de la barra
		(*) name <str>: nombre de la barra
		longitude <float>
		latitude <float>
		active <int>: indica si la barra está activa
'''
auxiliar=[]
buselectricfilas_aux=[]
for x in range(nbus): # Para cada barra (bus)
	if dirdfbus['BarName'].isin([indexbus['BarName'][x]]).tolist().count(True)>0:
		latitud=float(dirdfbus[dirdfbus['BarName']==indexbus['BarName'][x]]['latitud'].values[0])
		longitud=float(dirdfbus[dirdfbus['BarName']==indexbus['BarName'][x]]['longitud'].values[0])
	else:
		auxiliar.append(indexbus['BarName'][x])
		latitud,longitud=aleatory_direction()

	aux=[]
	aux.append(indexbus['id'][x])
	aux.append(indexbus['BarName'][x])
	aux.append(longitud)
	aux.append(latitud)
	aux.append(1)
	buselectricfilas_aux.append(aux)

buselectric=pd.DataFrame(buselectricfilas_aux,columns=['id','name','longitude','latitude','active'])

buselectric.to_json(electricTopology+"/bus.json",orient='records')

print("Creando datos topologicos Central")

# centrals electric contiene:

'''   
        (*) id <int>: identificador del generador
		(*) bus_id <int>: id de la barra conectada al generador
		(*) name <str>: nombre del generador
		active <int>: indica si el generador está activo
		capacity <float>: capacidad del generador [MW]
		min_power <float>: generación mínima [MW]
		max_power <float>: generación máxima [MW]
		type <str>: tipo de generador
		longitude <float>
		latitude <float>
		(?) effinciency <float>: Rendimiento [MWh/m3s]
		(?) flow <float>: parámetro no identificado
		(?) rmin <float>: parámetro no identificado
		(?) rmax <float>: parámetro no identificado
		(?) cvar <float>: Costo Variable
		(?) cvnc <unknown>: parámetro no identificado
		(?) cvc <unknown>: parámetro no identificado
		(?) entry_date <unknown>: parámetro no identificado
'''

centralselectricfilas_aux=[]
for x in range(ngen): # Para cada generador (central)
	if indexcen['bus_id'][x]==0 or np.isnan(indexcen['bus_id'][x]): # No existe la barra 0, por lo que no se consideran dichos generadores
		pass
	else:
		latitud,longitud=None,None
		aux=[]
		aux.append(indexcen['id'][x])
		aux.append(int(indexcen['bus_id'][x]))
		aux.append(indexcen['CenName'][x])
		aux.append(1)
		# capacidad
		aux.append(0)
		aux.append(centralsinfo[centralsinfo['CenName']==indexcen['CenName'][x]]['min_power'])
		aux.append(centralsinfo[centralsinfo['CenName']==indexcen['CenName'][x]]['max_power'])
		tipo=typecentrals[typecentrals['CenName']==indexcen['CenName'][x]]['cen_type'].values
		if len(tipo)>0:
			aux.append(tipo[0])
		else:
			aux.append(None)
		aux.append(longitud)
		aux.append(latitud)
		aux.append(centralsinfo[centralsinfo['CenName']==indexcen['CenName'][x]]['effinciency'])
		for x in range(7):
			aux.append(0)
		centralselectricfilas_aux.append(aux)

centralelectric=pd.DataFrame(centralselectricfilas_aux,columns=['id','bus_id','name','active','capacity','min_power','max_power','type','longitude','latitude','efficiency','flow','rmin','rmax','cvar',
'cvnc','cvc','entry_date'])

centralelectric.to_json(electricTopology+"/centrals.json",orient='records')


print("Creando datos topologicos Lineas")
# Lines electric tiene:
'''  
        (*) id <int>: identificador de la línea
		(*) bus_a <int>: id de la barra origen
		(*) bus_b <int>: id de la barra destino
		active <int>: indica si la línea está activa
		capacity <float>: capacidad máxima de la línea [MW]  ->
		max_flow_a_b <float>: flujo máximo en dirección
					dispuesta [MW]
		max_flow_b_a <float>: flujo máximo en dirección
					contraria [MW]
		voltage <float>: voltaje de la línea [kV]
		r <float>: resistencia de la línea [Ω]
		x <float>: reactancia de la línea [Ω]
		(? )segments <int>: parámetro no identificado
		(?) entry_date <unknown>: parámetro no identificado
		(?) exit_date <unknown>: parámetro no identificado
'''

lineselectricfilas_aux=[]
for x in range(nlin): # Para cada linea
	if linesfinal['active'][x]==1:
		aux=[]
		bus_a_id = linesfinal['bus_a'][x]
		bus_b_id = linesfinal['bus_b'][x]
		name = linesfinal['LinName'][x]
		aux.append(linesfinal['id'][x])
		aux.append(name)
		aux.append(bus_a_id)
		aux.append(bus_b_id)
		aux.append(1)
		# capacidad
		aux.append(0)
		aux.append(linesfinal['max_flow_a_b'][x])
		aux.append(linesfinal['max_flow_b_a'][x])
		aux.append(linesfinal['voltage'][x])
		aux.append(linesfinal['r'][x])
		aux.append(linesfinal['x'][x])
		aux.append(linesfinal['segments'][x])
		aux.append(None)
		aux.append(None)
		lineselectricfilas_aux.append(aux)

lineelectric=pd.DataFrame(lineselectricfilas_aux,columns=['id','name','bus_a','bus_b','active','capacity','max_flow_a_b','max_flow_b_a','voltage','r','x','segments','entry_date','exit_date'])

lineelectric.to_json(electricTopology+"/lines.json",orient='records')


print("Creando datos topologicos Reservoirs")

'''   
        (*) id <int>: identificador del embalse
		(*) junction_id <int>: id del embalse relacionada (mismo valor id)
		(*) name <str>: nombre del embalse
		(*) type <str>: tipo de embalse
		min_vol <float>: volumen mínimo del embalse
		max_vol <float>: volumen máximo del embalse
		start_vol <float>: volumen inicial del embalse
		end_vol <float>: volumen final del embalse
		active <bool>: indica si el embalse está activo
		(?) hyd_independant <bool>: parámetro no identificado
		(?) future_cost <unknown>: parámetro no identificado
		(?) cmin <unknown>: cota m.s.n.m mínima
'''

reshydricfilas_aux=[]
for x in range(nres): # Para cada linea
	aux=[]
	idaux=indexres['id'][x]
	name=indexres['EmbName'][x]
	junction_id = junctionsinfo[junctionsinfo['CenName']==name]['id'].values[0]
	
	aux.append(idaux)
	aux.append(junction_id)
	aux.append(name)
	aux.append(reservoirsinfo[reservoirsinfo['id']==idaux]['type'].values[0])
	aux.append(reservoirsinfo[reservoirsinfo['id']==idaux]['VembMin'].values[0])
	aux.append(reservoirsinfo[reservoirsinfo['id']==idaux]['VembMax'].values[0])
	aux.append(reservoirsinfo[reservoirsinfo['id']==idaux]['VembIn'].values[0])
	aux.append(reservoirsinfo[reservoirsinfo['id']==idaux]['VembFin'].values[0])
	aux.append(1)
	aux.append(0)
	aux.append(None)
	aux.append(reservoirsinfo[reservoirsinfo['id']==idaux]['cotaMínima'].values[0])
	reshydricfilas_aux.append(aux)

reshydric=pd.DataFrame(reshydricfilas_aux,columns=['id','junction_id','name','type','min_vol','max_vol','start_vol','end_vol','active','hyd_independant','future_cost','cmin'])

reshydric.to_json(hydricTopology+"/reservoirs.json",orient='records')

print("Creando datos topologicos Junctions")

'''
	(*) id <int>: identificador de la unión
	(*) name <str>: nombre de la unión
	longitude <float>
	latitude <float>
	active <bool>: indica si la barra está activa
	drainage <bool>: parámetro no identificado
'''

junctionhydricfilas_aux=[]
for x in range(len(junctionsinfo)): # Para cada junction
	latitud,longitud=aleatory_direction()
	aux=[]
	aux.append(junctionsinfo['id'][x])
	aux.append(junctionsinfo['CenName'][x])
	aux.append(longitud)
	aux.append(latitud)
	aux.append(1)
	aux.append(0)
	aux.append(junctionsinfo['bus_id'][x])
	
	junctionhydricfilas_aux.append(aux)

junctionhydric=pd.DataFrame(junctionhydricfilas_aux,columns=['id','name','logitude','latitude','active','drainage','bus_id'])

junctionhydric.to_json(hydricTopology+"/junctions.json",orient='records')

print("Creando datos topologicos Waterways")

'''
        (*) id <int>: identificador del canal
		(*) name <str>: nombre del canal
		(*) type <str>: tipo de waterway
		(*) junc_a_id <int>: id de la unión de origen
		(*) junc_b_id <int>: id de la unión de destino
		active <bool>: indica si el canal está activo
		(?) fmin <unknown>: parámetro no identificado
		(?) fmax <unknown>: parámetro no identificado
		(?) cvar <unknown>: parámetro no identificado 
        (?) delay <unknown>: parámetro no identificado
'''

junctionhydricfilas_aux=[]
countid=1
for x in range(len(junctionsinfo)):
    gen_id=junctionsinfo.serie_hidro_gen[x]
    ver_id=junctionsinfo.serie_hidro_ver[x]
    name_a = junctionsinfo.CenName[x]
    df_adicional = hydric_adicional[hydric_adicional['embalse'] == name_a]
    if not pd.isnull(gen_id):
        aux=[]
        aux.append(countid)
        countid+=1
        name_b = junctionsinfo[junctionsinfo['id']==gen_id].CenName.values[0]
        name = name_a+'_Gen_'+name_b
        aux.append(name)
        aux.append("generation")
        aux.append(junctionsinfo.id[x])
        aux.append(gen_id)
        aux.append(1)
        aux.append(None)
        aux.append(None)
        aux.append(None)
        aux.append(None)
        junctionhydricfilas_aux.append(aux)
    if not pd.isnull(ver_id):
        aux=[]
        aux.append(countid)
        countid+=1
        name_b = junctionsinfo[junctionsinfo['id']==ver_id].CenName.values[0]
        name = name_a+'_Vert_'+name_b
        aux.append(name)
        aux.append("spillover")
        aux.append(junctionsinfo.id[x])
        aux.append(ver_id)
        aux.append(1)
        aux.append(None)
        aux.append(None)
        aux.append(None)
        aux.append(None)
        junctionhydricfilas_aux.append(aux)
    if len(df_adicional)>0:
        for i in range(len(df_adicional)):
            tipo =df_adicional['type'].iloc[i]
            name =""
            central = df_adicional['central'].iloc[i].lower()
            id_central = centralsinfo[centralsinfo['CenName'].str.lower() == central]['id'].values[0]
            aux=[]
            aux.append(countid)
            countid+=1
            name_b = junctionsinfo[junctionsinfo['id']==id_central].CenName.values[0]
            if tipo == "filtration":
                name = name_a+'_Fil_'+name_b
            elif tipo == "extraction":
                name = name_a+'_Ext_'+name_b
            aux.append(name)
            aux.append(tipo)
            aux.append(junctionsinfo.id[x])
            aux.append(id_central)
            aux.append(1)
            aux.append(None)
            aux.append(None)
            aux.append(None)
            aux.append(None)
            junctionhydricfilas_aux.append(aux)
waterwayshydric=pd.DataFrame(junctionhydricfilas_aux,columns=["id","name","type","junc_a_id","junc_b_id","active","fmin","fmax","cvar","delay"])
waterwayshydric.to_json(hydricTopology+"/waterways.json",orient='records')

print("Archivos listos para visualizar ubicados en la ruta en donde se encuentra este archivo py")
print()
print(f"Se inicia la compresión de la carpeta de visualización '{namedata}' en formato '.zip'")
print("Esto puede tardar entre 3 a 5 minutos debido al tamaño de los archivos")
# Se comprime el archivo de salidas .json
current_directory = os.getcwd()
json_folder_path = os.path.join(current_directory, namedata)
zip_path = os.path.join(current_directory, namedata[5:])
shutil.make_archive(zip_path, "zip", json_folder_path)

print(f"Se le aplica un cambio de nombre para facilitar la carga del caso ({namedata[5:]}.zip)")

# Ruta de destino para mover la carpeta
Destino_Json = Folder_Json
# Mover la carpeta
shutil.move(json_folder_path, Destino_Json)

print(f"Se mueve la carpeta de visualización '{namedata}' hacia '{os.path.basename(Destino_Json)}'")
print("Se ha finalizado con éxito la función de procesamiento de Base de Datos y Resultados PLP")