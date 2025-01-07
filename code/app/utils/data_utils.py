import base64
import os
from .config import UPLOAD_FOLDER

def list_experiment_folders():
    """
    Returns a list of folders inside the results directory.
    This function lists all directories within the 'UPLOAD_FOLDER'.
    
    Returns:
        list: A list of folder names inside the 'UPLOAD_FOLDER' directory.
    """
    folders = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, f))]
    return folders

def list_files_in_folder(folder_name):
    """
    Returns a list of files inside the selected folder.
    Given a folder name, this function lists all the files present in that folder.
    
    Args:
        folder_name (str): The name of the folder whose files are to be listed.
        
    Returns:
        list: A list of filenames inside the specified folder.
    """
    folder_path = os.path.join(UPLOAD_FOLDER, folder_name)
    files = os.listdir(folder_path)
    return files

def bytes_to_base64(image_bytes):
    """
    Converts image bytes to a base64 encoded string.
    This function can handle both raw bytes and BytesIO objects.
    
    Args:
        image_bytes (bytes or BytesIO): The image data to be converted to base64.
        
    Returns:
        str: A base64 encoded string representation of the image data.
    
    Raises:
        ValueError: If the input is neither bytes nor a BytesIO object.
    """
    if isinstance(image_bytes, bytes):
        return base64.b64encode(image_bytes).decode('utf-8')  # Converts bytes to base64
    elif hasattr(image_bytes, 'getvalue'):
        image_bytes.seek(0)
        return base64.b64encode(image_bytes.getvalue()).decode('utf-8')  # For BytesIO objects
    else:
        raise ValueError("Expected bytes or BytesIO object")
