import base64
import os
from utils.config import UPLOAD_FOLDER

def list_experiment_folders():
    """Devuelve una lista con las carpetas dentro del directorio de resultados."""
    folders = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, f))]
    return folders

def list_files_in_folder(folder_name):
    """Devuelve una lista de archivos dentro de la carpeta seleccionada."""
    folder_path = os.path.join(UPLOAD_FOLDER, folder_name)
    files = os.listdir(folder_path)
    return files

def bytes_to_base64(image_bytes):
    if isinstance(image_bytes, bytes):
        return base64.b64encode(image_bytes).decode('utf-8')  # Convierte bytes a base64
    elif hasattr(image_bytes, 'getvalue'):
        image_bytes.seek(0)
        return base64.b64encode(image_bytes.getvalue()).decode('utf-8')  # Para BytesIO
    else:
        raise ValueError("Expected bytes or BytesIO object")

def save_html_or_binary(file_path, content):
        # Si el contenido es un flujo de bytes (_io.BytesIO), lo guardamos como binario
        if isinstance(content, bytes):
            with open(file_path, 'wb') as f:
                f.write(content)
        elif hasattr(content, 'getvalue'):  # Si es un objeto BytesIO
            with open(file_path, 'wb') as f:
                f.write(content.getvalue())
        else:
            # Si es una cadena de texto, lo guardamos como texto
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

