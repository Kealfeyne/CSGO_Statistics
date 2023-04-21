from pathlib import Path
import gdown


def download_datasets():
    url = "https://drive.google.com/drive/folders/1pPpJ3hhAOzjwhpinh4s9563mbPlmIvWM?usp=share_link"
    folder_name = "data/"

    if not Path(folder_name).exists():
        gdown.download_folder(url, output=folder_name)

    return


download_datasets()
