import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import contextily as ctx
import json
import base64
from io import BytesIO
from pyproj import Transformer
from shapely.geometry import LineString, box
from TRACLUS_web import traclus as tr
import io
import os

def load_and_simplify_data(filename, rows, tolerance=0.001, umbral_distancia=0.01, chunksize=10000):
    try:
        """ df_list = []
        chunk_iter = pd.read_csv(filename, sep=",", chunksize=chunksize, nrows=rows, low_memory=False)
        for chunk in chunk_iter:
            df_list.append(chunk)
        df = pd.concat(df_list) """
        # Cargar datos
        df = pd.read_csv(filename, nrows=rows, sep=",", low_memory=False)

        # Filtrar y crear LineString para cada polilínea
        def create_line(x):
            points = json.loads(x)
            if len(points) > 1:
                return LineString(points)
            return None

        df['geometry'] = df['POLYLINE'].apply(create_line)
        df = df[df['geometry'].notnull()]  # Eliminar filas con geometrías nulas
        gdf = gpd.GeoDataFrame(df, geometry='geometry')

        # Simplificar las geometrías
        gdf['geometry'] = gdf['geometry'].simplify(tolerance)

        # Convertir las polilíneas de JSON a listas de coordenadas
        df['POLYLINE'] = df['POLYLINE'].apply(lambda x: json.loads(x) if pd.notnull(x) else None)
        df = df[df['POLYLINE'].apply(lambda x: x is not None and len(x) > 0)]  # Filtrar polilíneas nulas o vacías

        # Preparar trayectorias para TRACLUS
        trayectorias = [np.array(polyline) for polyline in df['POLYLINE']]

        return gdf, trayectorias, df
    
    except Exception as e:
        print(f"Error cargando y simplificando datos: {e}")
        raise e

""" #! Carga de datos pureba provisional
def load_and_simplify_data_df(input_data, rows, tolerance=0.001, umbral_distancia=0.01):
    # Verifica si se ha pasado un filename o contenido de archivo
    if isinstance(input_data, str):
        # Si es un filename, se carga directamente desde el sistema
        df = pd.read_csv(input_data, nrows=rows, sep=",", low_memory=False)
    else:
        # Si es contenido de archivo, decodifica y carga desde base64
        content_type, content_string = input_data.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), nrows=rows, sep=",", low_memory=False)

    # Filtrar y crear LineString para cada polilínea
    def create_line(x):
        points = json.loads(x)
        if len(points) > 1:
            return LineString(points)
        return None
    
    df['geometry'] = df['POLYLINE'].apply(create_line)
    
    # Eliminar filas con geometrías nulas
    df = df[df['geometry'].notnull()]
    
    # Convertir a Geopandas DataFrame
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    
    # Simplificar las geometrías
    gdf['geometry'] = gdf['geometry'].simplify(tolerance)

    # Convertir las polilíneas de JSON a listas de coordenadas
    df['POLYLINE'] = df['POLYLINE'].apply(lambda x: json.loads(x) if pd.notnull(x) else None)

    # Filtrar filas con polilíneas vacías o nulas
    df = df[df['POLYLINE'].apply(lambda x: x is not None and len(x) > 0)]

    # Preparar las trayectorias para TRACLUS
    trayectorias = [np.array(polyline) for polyline in df['POLYLINE']]
    
    return gdf, trayectorias, df['POLYLINE'] """

def filter_data_in_area(gdf, minx, miny, maxx, maxy):
    # Crear un polígono de área de interés
    area_of_interest = box(minx, miny, maxx, maxy)
    
    # Filtrar los datos para incluir solo aquellos completamente dentro del área de interés
    gdf_filtered = gdf[gdf.geometry.within(area_of_interest)]
    
    return gdf_filtered

def list_coordinates(gdf):   
    # Crear listas vacías para las coordenadas x e y
    x_coords_flat = []
    y_coords_flat = []

    # Iterar sobre cada geometría en el GeoDataFrame
    for geom in gdf['geometry']:
        # Verificar que la geometría sea una LineString
        if isinstance(geom, LineString):
            # Iterar sobre cada punto en la LineString
            for point in geom.coords:
                x_coords_flat.append(point[0])  # Añadir la coordenada x a la lista
                y_coords_flat.append(point[1])  # Añadir la coordenada y a la lista
    
    return x_coords_flat, y_coords_flat

def map_ilustration(gdf, minx, miny, maxx, maxy):
    gdf = gdf.set_crs("EPSG:4326")

    # Luego, usar estas coordenadas en la función de filtrado
    gdf = filter_data_in_area(gdf, minx, miny, maxx, maxy)

    # Transforcion en EPSG:3857 para alinear con el mapa base de Contextily
    gdf = gdf.to_crs(epsg=3857)
    
    # Crear una figura con Matplotlib
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    gdf.plot(ax=ax, linewidth=0.5, color='green')

    # Añadir un mapa base con Contextily
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # Añadir título y etiquetas
    plt.title('Mapa de Trayectorias con Mapa de Fondo')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')

    # Crear un objeto BytesIO para guardar la imagen
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Mover el 'cursor' al principio del archivo en memoria
    
    # Es importante cerrar la figura para liberar memoria
    plt.close(fig)

    # Codificar la imagen generada en base64
    # encoded_string = base64.b64encode(img_data.read()).decode('utf-8')

    return img_data

def map_heat(gdf, minx, miny, maxx, maxy, bin_count=300, posicion_x=0, posicion_y=0, zoom=2):  
    # Obtener las coordenadas x e y de las geometrías 
    x_coords_flat, y_coords_flat = list_coordinates(gdf)

    # Calcular el histograma bidimensional de las coordenadas x e y
    heatmap, _, _ = np.histogram2d(x_coords_flat, y_coords_flat, bins=bin_count, density=True, range=[[minx, maxx], [miny, maxy]])

    # Inicializar el transformador de coordenadas
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)

    # Transformar las coordenadas
    xmin, ymin = transformer.transform(minx, miny)
    xmax, ymax = transformer.transform(maxx, maxy)
    
    """ # Calcular el centro y el rango de los ejes x e y
    x_center, y_center = ((xmin + xmax) / 2) + posicion_x, ((ymin + ymax) / 2) + posicion_y
    x_range, y_range = (xmax - xmin) / zoom, (ymax - ymin) / zoom """

    # Crear la figura y los ejes para matplotlib
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # Crear una normalización logarítmica
    norm = colors.LogNorm(vmin=heatmap.min()+1, vmax=heatmap.max())

    # Mostrar el mapa de calor y capturar el objeto mappable retornado por imshow
    mappable = ax.imshow(heatmap.T, origin='lower', norm=norm ,extent=[xmin, xmax, ymin, ymax], aspect='auto', alpha=0.7, zorder=2)

    # Añadir el mapa base
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')

    """ # Ajustar los límites de los ejes para coincidir con los bordes del histograma
    ax.set_xlim(x_center - x_range, x_center + x_range)
    ax.set_ylim(y_center - y_range, y_center + y_range) """

    # Añadir barra de color, títulos y etiquetas usando el objeto mappable
    plt.colorbar(mappable, label='Densidad')
    plt.title('Mapa de Calor de Trayectorias con Mapa de Fondo')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')

    # Crear un objeto BytesIO para guardar la imagen
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Mover el 'cursor' al principio del archivo en memoria
    
    # Es importante cerrar la figura para liberar memoria
    plt.close(fig)

    # Codificar la imagen generada en base64
    #encoded_string = base64.b64encode(img_data.read()).decode('utf-8')

    return img_data

# Ahora mismo funciona para obtener las coordenadas maximas y minimas de un GeoDataFrame
def solicitar_coordenadas(gdf):
    """ print("Por favor, introduce las coordenadas para el área de interés.")
    minx = float(input("Introduce la longitud mínima (minx): "))
    miny = float(input("Introduce la latitud mínima (miny): "))
    maxx = float(input("Introduce la longitud máxima (maxx): "))
    maxy = float(input("Introduce la latitud máxima (maxy): ")) """
    
    """ minx=-8.689
    miny=41.107
    maxx=-8.560
    maxy=41.185 """

    x_coords_flat, y_coords_flat = list_coordinates(gdf)

    maxx, maxy, minx, miny = max(x_coords_flat), max(y_coords_flat), min(x_coords_flat), min(y_coords_flat)

    return minx, miny, maxx, maxy 




#Funciones para la pagina "Comparacion de algoritmos" (Pagina 2)
#* Representación de resultados

def create_gdf(data):
    valid_geometries = [LineString(x) for x in data if len(x) > 1]
    gdf = gpd.GeoDataFrame(geometry=valid_geometries, crs='EPSG:4326')

    return gdf

def get_cluster_trajectories(df, trajectories, directional=True, use_segments=True, clustering_algorithm=None, 
                            OPTICS_metric=None, OPTICS_algorithm=None, OPTICS_eps=None, OPTICS_sample=None, 
                            DBSCAN_metric=None, DBSCAN_algorithm=None, DBSCAN_eps=None, DBSCAN_sample=None, 
                            HDBSCAN_metric=None, HDBSCAN_algorithm=None, HDBSCAN_sample=None, 
                            Aggl_metric=None, Aggl_linkage=None, Aggl_n_clusters=None, 
                            Spect_affinity=None, Spect_assign_labels=None, Spect_n_clusters=None):
    
    result = tr(trajectories=trajectories, directional=directional, use_segments=use_segments, clustering_algorithm=clustering_algorithm, 
                OPTICS_min_samples=OPTICS_sample, OPTICS_max_eps=OPTICS_eps, OPTICS_metric=OPTICS_metric, OPTICS_algorithm=OPTICS_algorithm, 
                DBSCAN_min_samples=DBSCAN_sample, DBSCAN_eps=DBSCAN_eps, DBSCAN_metric=DBSCAN_metric, DBSCAN_algorithm=DBSCAN_algorithm, 
                HDBSCAN_min_samples=HDBSCAN_sample, HDBSCAN_metric=HDBSCAN_metric, HDBSCAN_algorithm=HDBSCAN_algorithm, 
                Spect_n_clusters=Spect_n_clusters, Spect_affinity=Spect_affinity, Spect_assign_labels=Spect_assign_labels,
                Aggl_n_clusters=Aggl_n_clusters, Aggl_linkage=Aggl_linkage, Aggl_metric=Aggl_metric)
    
    _, segments, _, _, cluster_assignments, representative_trajectories = result
    # Representacion de las trayectorias pero sin el primer elemento, este parece ser solo un conjunto basura
    representative_clusters = representative_trajectories[1:representative_trajectories.__len__()]

    TRACLUS_map = plot_map_traclus(representative_clusters)
    TRACLUS_map_df = plot_map_traclus_df(representative_clusters, df['POLYLINE'])
    tabla_relacional = relational_table(df, segments, cluster_assignments)

    return TRACLUS_map, TRACLUS_map_df, tabla_relacional

def plot_map_traclus(representative_clusters, cmap='tab20'):
    # Crear un GeoDataFrame
    gdf = create_gdf(representative_clusters)

    # Visualizar en un mapa
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    gdf.plot(ax=ax, cmap=cmap, linewidth=2)
    

    # Añadir mapa base
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs.to_string())

    # Añadir título y etiquetas
    plt.title('Representación de Trayectorias TRACLUS')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')

    # Crear un objeto BytesIO para guardar la imagen
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Mover el 'cursor' al principio del archivo en memoria
    #plt.show()
    # Es importante cerrar la figura para liberar memoria
    plt.close(fig)

    # Codificar la imagen generada en base64
    #TRACLUS_map = base64.b64encode(img_data.read()).decode('utf-8')

    return img_data

def plot_map_traclus_df(representative_clusters, df, cmap='tab20'):
    # Crear un GeoDataFrame
    gdf2 = create_gdf(df)
    gdf = create_gdf(representative_clusters)

    # Visualizar en un mapa
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    gdf2.plot(ax=ax, color='blue', alpha=0.2, linewidth=0.5)
    gdf.plot(ax=ax, cmap=cmap, linewidth=2)
    

    # Añadir mapa base
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs.to_string())

    # Añadir título y etiquetas
    plt.title('Representación de Trayectorias TRACLUS con trayectorias originales')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')

    # Crear un objeto BytesIO para guardar la imagen
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Mover el 'cursor' al principio del archivo en memoria
    #plt.show()
    # Es importante cerrar la figura para liberar memoria
    plt.close(fig)

    # Codificar la imagen generada en base64
    #TRACLUS_map_df = base64.b64encode(img_data.read()).decode('utf-8')

    return img_data


# Funciones para la pagina "Estadisticas" (Pagina 3)
#* Tablas de datos

def relational_table(df, segments, cluster_assignments):
    # Temporary list to store data before creating GeoDataFrame
    gdf_stc_data = []
    index = 0

    for segment, cluster_id in zip(segments, cluster_assignments):
        if isinstance(segment, np.ndarray):
            line = LineString(segment)
            tray_id_found = False
            
            for i, polyline in enumerate(df['POLYLINE'][index:len(df['POLYLINE'])]):
                # Check if the polyline has 2 or more points
                if i > 0:
                    index += 1

                if len(polyline) >= 2:
                    tray_line = LineString(polyline)
                    
                    if tray_line.intersects(line):
                        tray_id = index
                        tray_id_found = True
                        break  # Exit the loop once a matching tray_line is found
                else:
                    continue  # Skip this polyline as it cannot form a valid LineString

            if not tray_id_found:
                tray_id = -1  # Indicator for 'not found'

            # Append valid data to the list
            gdf_stc_data.append({'geometry': line, 'cluster_id': cluster_id, 'tray_id': tray_id})

    # Create the GeoDataFrame from the collected data
    gdf_stc = gpd.GeoDataFrame(gdf_stc_data, columns=['geometry', 'cluster_id', 'tray_id'])

    return gdf_stc
