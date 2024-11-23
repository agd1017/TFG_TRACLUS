import pandas as pd
import geopandas as gpd
import numpy as np
import json
from shapely.geometry import LineString
import base64
import io

def load_and_simplify_data(filename, rows, tolerance=0.001):
    try:
        _, content_string = filename.split(',')
        decoded = base64.b64decode(content_string)

        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), nrows=rows) 
        else:
            df = pd.read_excel(io.BytesIO(decoded), nrows=rows)

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

# Funciones para la pagina "Estadisticas" (Pagina 3)
#* Tablas de datos

def relational_table(df, segments, cluster_assignments, representative_trajectories):
    # Lista temporal para almacenar datos antes de crear el GeoDataFrame
    gdf_stc_data = []

    for segment, cluster_id in zip(segments, cluster_assignments):
        if isinstance(segment, np.ndarray):
            line = LineString(segment)
            tray_id = -1  # Indicador de 'no encontrado'

            # Buscamos la trayectoria representativa correspondiente al clúster
            for rep_id, rep_trajectory in enumerate(representative_trajectories):
                rep_line = LineString(rep_trajectory)

                if rep_line.intersects(line):
                    tray_id = rep_id  # `tray_id` es el índice de la trayectoria representativa
                    tray_id_found = True
                    break

            # Añadimos los datos a la lista, incluyendo tray_id
            gdf_stc_data.append({
                'line_index': df.index[0],  # Agregar el índice de la línea del df
                'geometry': line,
                'cluster_id': cluster_id,
                'tray_id': tray_id
            })

    # Crear el GeoDataFrame a partir de los datos recopilados
    gdf_stc = gpd.GeoDataFrame(gdf_stc_data, columns=['line_index', 'geometry', 'cluster_id', 'tray_id'])

    return gdf_stc

def get_cluster_graph(cluster_assignments):
    return  [asig for asig in cluster_assignments if asig != -1]