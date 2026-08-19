import os
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_coco_val2017():
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = "data/val2017.zip"
    extract_path = "data"

    os.makedirs("data", exist_ok=True)

    
    if not os.path.exists(zip_path):
        print("COCO Val2017 Dataset Downloading (~1 GB)...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="COCO Val2017") as t:
            urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
        print("Download Completed")
    else:
        print("Zip file is available. The installation completed")

    
    target_dir = os.path.join(extract_path, "val2017")
    if not os.path.exists(target_dir):
        print("Extracting the zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction completed")
    else:
        print("The dataset has already been extracted.")

if __name__ == "__main__":
    download_coco_val2017()