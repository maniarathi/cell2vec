import urllib.request

BASE_URL = "https://covid19.cog.sanger.ac.uk/"

LUNG_DATASETS = ["lukassen20_airway_orig.processed.h5ad", "lukassen20_lung_orig.processed.h5ad",
                 "vieira19_Alveoli_and_parenchyma_anonymised.processed.h5ad",
                 "vieira19_Bronchi_anonymised.processed.h5ad", "vieira19_Nasal_anonymised.processed.h5ad"]

for dataset_name in LUNG_DATASETS:
    print(f"Fetching dataset: {dataset_name}")
    url = BASE_URL + dataset_name
    urllib.request.urlretrieve(url, dataset_name)
