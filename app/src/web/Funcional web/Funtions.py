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

def load_and_simplify_data(filename, rows, tolerance=0.001, umbral_distancia=0.01):
    # Cargar datos
    df = pd.read_csv(filename, nrows=rows, sep=",", low_memory=False)
    
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
    

    #! ARREGLO PROVISIONAL 

    # Simplificar las geometrías
    gdf['geometry'] = gdf['geometry'].simplify(tolerance)

    # Convertir las polilíneas de JSON a listas de coordenadas
    df['POLYLINE'] = df['POLYLINE'].apply(lambda x: json.loads(x) if pd.notnull(x) else None)

    # Filtrar filas con polilíneas vacías o nulas
    df = df[df['POLYLINE'].apply(lambda x: x is not None and len(x) > 0)]

    # Preparar las trayectorias para TRACLUS
    trayectorias = [np.array(polyline) for polyline in df['POLYLINE']]
    
    return gdf, trayectorias, df['POLYLINE']

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
    plt.title('Mapa de Trayectorias de Taxis con Mapa de Fondo')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')

    # Crear un objeto BytesIO para guardar la imagen
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Mover el 'cursor' al principio del archivo en memoria
    
    # Es importante cerrar la figura para liberar memoria
    plt.close(fig)

    # Codificar la imagen generada en base64
    encoded_string = base64.b64encode(img_data.read()).decode('utf-8')

    return encoded_string

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
    plt.title('Mapa de Calor de Trayectorias de Taxis con Mapa de Fondo')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')

    # Crear un objeto BytesIO para guardar la imagen
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)  # Mover el 'cursor' al principio del archivo en memoria
    
    # Es importante cerrar la figura para liberar memoria
    plt.close(fig)

    # Codificar la imagen generada en base64
    encoded_string = base64.b64encode(img_data.read()).decode('utf-8')

    return encoded_string

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

def get_cluster_trajectories(df, trajectories, max_eps=None, min_samples=5, min_cluster_size=None, cluster_selection_epsilon=None, n_clusters=2, affinity='rbf', n_neighbors=5, linkage='ward', threshold=None, directional=True, use_segments=True, clustering_algorithm=None):
    _, _, _, _, _, representative_trajectories = tr(trajectories,  max_eps, min_samples, min_cluster_size, cluster_selection_epsilon, n_clusters, affinity, n_neighbors, linkage, threshold, directional, use_segments, clustering_algorithm)

    # Representacion de las trayectorias pero sin el primer elemento, este parece ser solo un conjunto basura
    representative_clusters = representative_trajectories[1:representative_trajectories.__len__()]
    n_clusters = len(representative_clusters)

    TRACLUS_map = plot_map_traclus(representative_clusters)
    TRACLUS_map_df = plot_map_traclus_df(representative_clusters, df)

    return TRACLUS_map, TRACLUS_map_df, n_clusters

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
    TRACLUS_map = base64.b64encode(img_data.read()).decode('utf-8')

    return TRACLUS_map

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
    TRACLUS_map_df = base64.b64encode(img_data.read()).decode('utf-8')

    return TRACLUS_map_df


# Funciones para la pagina "Estadisticas" (Pagina 3)
#* Tablas de datos

def create_dataframe():

    # Esta función podría ser más compleja y cargar o generar los datos
    return pd.DataFrame({
        'Columna 1': [1, 2, 3, 4],
        'Columna 2': ['A', 'B', 'C', 'D'],
        'Columna 3': [5, 6, 7, 8],
        'Columna 4': ['E', 'F', 'G', 'H']
    })
