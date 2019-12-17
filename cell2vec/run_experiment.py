from training.train import Train

DATA_ROOT = "../data/"
DATA_FILES = {"endoderm": "endoderm.h5ad"}

if __name__ == "__main__":
    data_file = DATA_ROOT + DATA_FILES.get("endoderm")

    trainer = Train(data_file)
