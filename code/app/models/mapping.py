import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import contextily as ctx
from io import BytesIO
from pyproj import Transformer
from shapely.geometry import LineString, box

#* Mapas

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

    # Crear la figura y los ejes para matplotlib
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # Crear una normalización logarítmica
    norm = colors.LogNorm(vmin=heatmap.min()+1, vmax=heatmap.max())

    # Mostrar el mapa de calor y capturar el objeto mappable retornado por imshow
    mappable = ax.imshow(heatmap.T, origin='lower', norm=norm ,extent=[xmin, xmax, ymin, ymax], aspect='auto', alpha=0.7, zorder=2)

    # Añadir el mapa base
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')

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

    return img_data

# Ahora mismo funciona para obtener las coordenadas maximas y minimas de un GeoDataFrame
def get_coordinates(gdf):

    x_coords_flat, y_coords_flat = list_coordinates(gdf)

    maxx, maxy, minx, miny = max(x_coords_flat), max(y_coords_flat), min(x_coords_flat), min(y_coords_flat)

    return minx, miny, maxx, maxy

#* TRACLUS comparation mapas

def create_gdf(data):
    valid_geometries = [LineString(x) for x in data if len(x) > 1]
    gdf = gpd.GeoDataFrame(geometry=valid_geometries, crs='EPSG:4326')

    return gdf

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

    # Es importante cerrar la figura para liberar memoria
    plt.close(fig)

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

    # Es importante cerrar la figura para liberar memoria
    plt.close(fig)

    return img_data
