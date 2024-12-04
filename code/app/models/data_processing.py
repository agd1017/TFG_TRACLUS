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
    
def is_segment_in_trajectory(segment, trajectory):
    segment_line = LineString(segment)
    trajectory_line = LineString(trajectory)

    return segment_line.intersects(trajectory_line)


def relational_table(df, segments, cluster_assignments, representative_trajectories):
    gdf_stc_data = []
    i = 0

    for segment, cluster_id in zip(segments, cluster_assignments):
        line = LineString(segment)
        tray_id = -1  # Indicador de 'no encontrado'
        line_index = None

        for index, trajectory in zip(df.index, df['POLYLINE']):
            if is_segment_in_trajectory(segment, trajectory) and i <= index:
                line_index = index
                i = index
                break

        for rep_id, rep_trajectory in enumerate(representative_trajectories):
            rep_line = LineString(rep_trajectory)
            if rep_line.intersects(line):
                tray_id = rep_id
                break

        gdf_stc_data.append({
            'line_index': line_index,
            'segment': line,
            'cluster_id': cluster_id,
            'represent_tray_id': tray_id
        })

    gdf_stc = gpd.GeoDataFrame(gdf_stc_data, columns=['line_index', 'segment', 'cluster_id', 'represent_tray_id'])
    return gdf_stc

#* Graficos de barras

def get_cluster_graph(cluster_assignments):
    return  [asig for asig in cluster_assignments if asig != -1]