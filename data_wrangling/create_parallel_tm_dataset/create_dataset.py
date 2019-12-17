import csv
import loompy
from os import listdir
from os.path import isfile, join
import pandas as pd

class ParallelTMDatasetCreator:
    """ Creates a parallel tabula muris dataset combining smartseq2 and 10x data equally """
    def __init__(self, organs=["bladder"]):
        self.organs = organs
        self.desired_cell_attributes = ["Cell_Type", "Mouse_Id"]
        self.desired_gene_attributes = []

        self.smart_seq_2_annotations_file = "annotations_FACS.csv"
        self.smart_seq_2_metadata_file = "metadata_FACS.csv"
        self.smart_seq_2_root = "data/tabula_muris_smartseq2/"
        self.smart_seq_2_data_root = "FACS/"

    def get_smart_seq_2_organ_files(self):
        annotation_file_name = self.smart_seq_2_root + self.smart_seq_2_annotations_file
        annotation_file_reader = pd.read_csv(annotation_file_name)

        # Get related raw gene x cell data files
        related_data_filenames = []
        for filename in listdir(self.smart_seq_2_root + self.smart_seq_2_data_root):
            if any(organ.lower() in filename.lower() for organ in self.organs):
                related_data_filenames.append(filename)
        print(f"Organ data files are: {related_data_filenames}")

        for data_filename in related_data_filenames:
            data_file = open(self.smart_seq_2_root + self.smart_seq_2_data_root + data_filename)
            data_file_reader = csv.DictReader(data_file)
            cell_names = data_file_reader.fieldnames
            cell_names = cell_names.remove("")

            data_file_reader_2 = pd.read_csv(self.smart_seq_2_root + self.smart_seq_2_data_root + data_filename, usecols=cell_names)
            for key in data_file_reader_2.keys():
                if key in annotation_file_reader.keys():
                    print(annotation_file_reader[key])
            #print(data_file_reader_2.keys())



        read_column_names = False
        for row in annotation_file_reader:
            if not read_column_names:
                print(f'Column names are {", ".join(row)}')
                read_column_names = True
        pass

class CreateMacaSubset:
    def __init__(self, gender=['m', 'f'], organs=['bladder'], tech=['10x', 'ss2']):
        self.gender = gender
        self.organs = organs
        self.tech = tech

        self.file =