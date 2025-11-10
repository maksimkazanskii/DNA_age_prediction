import os
import pandas as pd
import numpy as np
import joblib
import json


class DamagePredictorInference:
    """A class for making predictions using the trained Damage Predictor model."""

    def __init__(self, folder, config):
        self.model_folder = folder
        self.basic_features = True
        self.scaler = joblib.load(os.path.join(folder, "best_scaler.pkl"))
        self.pca = joblib.load(os.path.join(folder, "best_pca.pkl"))
        self.model = joblib.load(os.path.join(folder, "best_model.pkl"))
        self.CONFIG = config

    def get_damage_features(self, folder_path):
        """Retrieve damage features from a specified folder.

        This method reads data from a file named 'misincorporation.txt' located in
        the specified folder. It then processes the data and returns relevant
        features

        Args:
            folder_path (str): The path to the folder containing the data file.

        Returns:
            pd.DataFrame or None: A DataFrame containing damage features, or None if
                an error occurs or the file is not found.
        """
        file_name = "misincorporation.txt"
        full_file_path = os.path.join(folder_path, file_name)
        try:
            df = pd.read_csv(full_file_path, sep="\t", skiprows=3)
            if self.basic_features:
                rows = df[df["Pos"].astype(int) < 60]
                features = rows[self.CONFIG['feature_names']]
                features = features.replace(r'\.(\d+)\.', r'.\1', regex=True)
                features = features.astype(float)
                normalized_features = features.copy()
                dict_norm = {
                    "A": "Total", "C": "Total", "G": "Total", "T": "Total", "Total": None,
                    "A>T": "A", "G>A": "G", "C>T": "C", "A>G": "A", "T>C": "T",
                    "A>C": "A", "C>G": "C", "C>A": "C", "T>G": "T", "T>A": "T",
                    "G>C": "G", "G>T": "G", "A>-": "A", "T>-": "T", "C>-": "C",
                    "G>-": "G", "->A": "A", "->T": "T", "->C": "C", "->G": "G",
                    "S": "Total"
                }
                reference_values = {col: normalized_features[col].copy() for col in set(dict_norm.values()) if
                                    col is not None}
                for col in features.columns:
                    reference_col = dict_norm.get(col)
                    if reference_col is not None:
                        normalized_features.loc[normalized_features['Total'] != 0, col] = (
                                normalized_features.loc[normalized_features['Total'] != 0, col] /
                                reference_values[reference_col][normalized_features['Total'] != 0]
                        )

                    normalized_features[col] = normalized_features[col].fillna(normalized_features[col].mean())
                normalized_features = normalized_features.drop(columns=['Total'])
                return normalized_features
            else:
                return None
        except FileNotFoundError:
            print(f"The file '{file_name}' was not found in the specified folder. {folder_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        return None

    def get_metadata(self, bam_name):

        """Get metadata information for a given BAM file name.

        Args:
            bam_name (str): The BAM file name.

        Returns:
            float or None: The age from the metadata for the specified BAM file, or
            None if not found.
        """
        print(bam_name)
        metadata_file = self.CONFIG['metadata_file']
        metadata_groups = self.CONFIG['batch_metadata_file']
        metadata_df = pd.read_csv(metadata_file, delimiter='\t')
        metadata_df_groups = pd.read_csv(metadata_groups, delimiter='\t')
        try:
            matching_rows = metadata_df[metadata_df['Run accession'] == bam_name]
        except:
            matching_rows = metadata_df[metadata_df['bam_name'] == bam_name]
        if matching_rows.empty:
            return None
        else:
            matching_rows_groups = metadata_df_groups[metadata_df_groups['sample'] == bam_name]
            if matching_rows.empty:
                return None
            else:
                age = float(matching_rows['age'].iloc[0])
                batch_name = matching_rows_groups['article_group'].iloc[0]
                return age, batch_name

    def prepare_dataset(self, m_damage_folder, verbose=True):
        """Prepare a dataset for damage prediction.

        Args:
            basic_features (bool, optional): Whether to include basic features. Default is True.
            verbose(bool) : Whether to print intermediate statistics.
        Returns:
            pd.DataFrame: The prepared dataset for damage prediction.
        """
        data = {}
        full_df = pd.DataFrame(data)
        count = 0
        features = []
        for root, directories, files in os.walk(m_damage_folder):
            for directory in directories:
                count += 1
                path = os.path.join(root, directory)
                bam_name = path.split("/")[-1].split(".")[0]
                (age, batch_name) = self.get_metadata(bam_name)
                features_concat = pd.DataFrame({
                                                'bam_name': [bam_name],
                                                'batch_name': [batch_name]})
                features = self.get_damage_features(path)
                if features is not None:
                    flattened_array = features.values.ravel()
                    flattened_df = pd.DataFrame([flattened_array])
                    new_column_names = [f"{col}_{i}" for i in range(features.shape[0]) for col in features.columns]
                    flattened_df.columns = new_column_names
                    features_concat = pd.concat([features_concat, flattened_df], axis=1)
                full_df = pd.concat([full_df, features_concat], ignore_index=True)
        if verbose:
            print("\nTOTAL COUNT OF THE BAM FOLDERS ", count)
            print("\nFILTERING OUT NOT PROCESSED DAMAGE FILES ")

        cols_to_convert = full_df.columns.difference(['batch_name', 'bam_name'])
        full_df[cols_to_convert] = full_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        columns_to_log = []
        for col in full_df.columns:
            if col not in ['label', 'batch_name', 'bam_name']:
                columns_to_log.append(col)
        full_df['bam_name'] = full_df['bam_name'].astype(str)
        full_df = full_df.fillna(full_df.median())
        not_corrected_data = full_df
        not_corrected_data.to_csv(os.path.join("data/data_comparison", "batch_effect_main.csv"))
        if verbose:
            print("\nLENGTH OF FULL_DATAFRAME AFTER CLEANING not existing Damage folders", len(full_df))
            print("\nHEAD OF THE DF_FULL\n", full_df.head(10))
        bam_name_col = full_df['bam_name']
        return bam_name_col, full_df

    def predict(self, data):
        """
        Makes predictions on the input data.

        Args:
            data (pd.DataFrame): The input data for prediction.

        Returns:
            np.ndarray: The predicted target values.
        """
        X_scaled = self.scaler.transform(data)
        X_pca = self.pca.transform(X_scaled)
        predictions = self.model.predict(X_pca)
        return predictions


def inference_bam_files():
    with open("config/config_harvard_60_cv5.json", 'r') as file:
        CONFIG = json.load(file)
    model_folder = os.path.join(CONFIG['exp_folder'], "best_model")
    predictor = DamagePredictorInference(model_folder, CONFIG)
    mDamage_folder = "/Users/maksimkazanskii2/PycharmProjects/GEN_AGE/data/full_test"
    (filenames, data) = predictor.prepare_dataset(mDamage_folder)
    predictions = predictor.predict(data)
    dict_ages = dict(zip(filenames, predictions))
    print(dict_ages)

if __name__ == '__main__':
    inference_bam_files()
