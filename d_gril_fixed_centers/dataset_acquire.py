'''
Script is based on the following tutorial:
https://projects.volkamerlab.org/teachopencadd/talktorials/T001_query_chembl.html
'''
import math
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from chembl_webresource_client.new_client import new_client
from tqdm.auto import tqdm
import argparse
import os

def convert_ic50_to_pic50(IC50_value):
    pIC50_value = 9 - math.log10(IC50_value)
    return pIC50_value

class DataAcquisition:
    def __init__(self, chembl_id) -> None:
        self.targets_api, self.compounds_api, self.bioactivities_api = self.setup_apis()
        self.chembl_id = chembl_id
        self.bioactivities = None
        self.bioactivities_df = None

    def setup_apis(self):
        targets_api = new_client.target
        compounds_api = new_client.molecule
        bioactivities_api = new_client.activity
        return targets_api, compounds_api, bioactivities_api

    def get_bioactivity_data(self):
        print(f"The target ChEMBL ID is {self.chembl_id}")
        bioactivities = self.bioactivities_api.filter(
        target_chembl_id=self.chembl_id, type="IC50", relation="=", assay_type="B"
        ).only(
            "activity_id",
            "assay_chembl_id",
            "assay_description",
            "assay_type",
            "molecule_chembl_id",
            "type",
            "standard_units",
            "relation",
            "standard_value",
            "target_chembl_id",
            "target_organism",
        )
        self.bioactivities = bioactivities
        print(f"Length and type of bioactivities object: {len(self.bioactivities)}, {type(self.bioactivities)}")

    def download_data(self):
        bioactivities_df = pd.DataFrame.from_dict(self.bioactivities)
        print(f"DataFrame shape: {bioactivities_df.shape}")
        self.bioactivities_df = bioactivities_df

    def filter_and_preprocess(self):
        bioactivities_df = self.bioactivities_df.copy()
        bioactivities_df.drop(["units", "value"], axis=1, inplace=True)
        bioactivities_df = bioactivities_df.astype({"standard_value": "float64"})
        bioactivities_df.dropna(axis=0, how="any", inplace=True)
        print(f"Units in downloaded data: {bioactivities_df['standard_units'].unique()}")
        print(
            f"Number of non-nM entries:\
            {bioactivities_df[bioactivities_df['standard_units'] != 'nM'].shape[0]}"
        )
        bioactivities_df = bioactivities_df[bioactivities_df["standard_units"] == "nM"]
        print(f"Units after filtering: {bioactivities_df['standard_units'].unique()}")
        print(f"DataFrame shape: {bioactivities_df.shape}")
        bioactivities_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
        print(f"DataFrame shape after deduping: {bioactivities_df.shape}")
        bioactivities_df.reset_index(drop=True, inplace=True)
        bioactivities_df.rename(
            columns={"standard_value": "IC50", "standard_units": "units"}, inplace=True
        )
        self.bioactivities_df = bioactivities_df
    
    def get_compounds(self): 
        compounds_provider = self.compounds_api.filter(
            molecule_chembl_id__in=list(self.bioactivities_df["molecule_chembl_id"])
        ).only("molecule_chembl_id", "molecule_structures")
        compounds = list(tqdm(compounds_provider))
        compounds_df = pd.DataFrame.from_records(
         compounds,
        )
        print(f"Compounds dataframe shape: {compounds_df.shape}")
        compounds_df.dropna(axis=0, how="any", inplace=True)
        print(f"Compounds dataframe shape after filtering nan: {compounds_df.shape}")
        compounds_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
        print(f"Compounds dataframe shape after deduping: {compounds_df.shape}")
        canonical_smiles = []
        print("Extracting canonical SMILES...")
        for i, compounds in compounds_df.iterrows():
            try:
                canonical_smiles.append(compounds["molecule_structures"]["canonical_smiles"])
            except KeyError:
                canonical_smiles.append(None)

        compounds_df["smiles"] = canonical_smiles
        compounds_df.drop("molecule_structures", axis=1, inplace=True)
        print(f"DataFrame shape: {compounds_df.shape}")
        compounds_df.dropna(axis=0, how="any", inplace=True)
        print(f"Compounds dataframe shape after filtering nan: {compounds_df.shape}")
        self.compounds_df = compounds_df
    
    def merge_data(self):
        output_df = pd.merge(
        self.bioactivities_df[["molecule_chembl_id", "IC50", "units"]],
        self.compounds_df,
        on="molecule_chembl_id",
        )

        # Reset row indices
        output_df.reset_index(drop=True, inplace=True)

        print(f"Dataset with {output_df.shape[0]} entries.")
        output_df = output_df[output_df["IC50"] != 0]
        output_df["pIC50"] = output_df.apply(lambda x: convert_ic50_to_pic50(x.IC50), axis=1)
        self.output_df = output_df

    def save_data(self, output_dir):
        # PandasTools.AddMoleculeColumnToFrame(self.output_df, smilesCol="smiles")
        # Sort molecules by pIC50
        self.output_df.sort_values(by="pIC50", ascending=False, inplace=True)
        # Reset index
        self.output_df.reset_index(drop=True, inplace=True)
        self.output_df.to_csv(os.path.join(output_dir, f'{self.chembl_id}.csv'), index=False)

def main(args):
    chembl_id = args.chembl_id
    output_dir = args.output_dir
    if os.path.exists(os.path.join(output_dir, f'{chembl_id}.csv')):
        print(f'Data for {chembl_id} already exists')
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    data_acquisition = DataAcquisition(chembl_id)
    data_acquisition.get_bioactivity_data()
    print(f'Bioactivity data for {chembl_id} accessed')
    data_acquisition.download_data()
    print(f'Bioactivity data for {chembl_id} downloaded')
    data_acquisition.filter_and_preprocess()
    print(f'Bioactivity data for {chembl_id} filtered')
    data_acquisition.get_compounds()
    print(f'Compounds for {chembl_id} downloaded')
    data_acquisition.merge_data()
    print(f'Data for {chembl_id} merged')
    data_acquisition.save_data(output_dir)
    print(f'Data for {chembl_id} saved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download ChEMBL data for a given target')
    parser.add_argument('--chembl_id', required=True, type=str, help='ChEMBL ID of the target')
    parser.add_argument('--output_dir', type=str, help='Output directory', default='dataset')
    args = parser.parse_args()
    main(args)






