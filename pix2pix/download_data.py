import os
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_coco_dataset(dataset_type="val2017"):
    """
    dataset_type: 'val2017' (~1 GB, 5k görsel) veya 'train2017' (~18 GB, 118k görsel)
    """
    url = f"http://images.cocodataset.org/zips/{dataset_type}.zip"
    zip_path = f"data/{dataset_type}.zip"
    extract_path = "data"

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(zip_path):
        print(f"COCO {dataset_type} indiriliyor...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dataset_type) as t:
            urllib.request.urlretrieve(url, filename=zip_path, reporthook=t.update_to)
        print("İndirme tamamlandı.")
    else:
        print(f"{zip_path} halihazırda mevcut.")

    target_dir = os.path.join(extract_path, dataset_type)
    if not os.path.exists(target_dir):
        print("Zip dosyası çıkarılıyor...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Çıkarma işlemi tamamlandı.")
    else:
        print(f"{target_dir} klasörü zaten çıkarılmış.")

if __name__ == "__main__":
    # İster sadece val2017, ister train2017 indir
    download_coco_dataset("val2017")
    download_coco_dataset("train2017") # 18 GB'lık dev veri seti