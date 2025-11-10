import os
import pandas as pd
import numpy as np
import random
from joblib import load
import json
from sklearn.metrics import make_scorer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class AverageModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.mean_value = None

    def fit(self, x, y):
        self.mean_value = np.mean(y)
        return self

    def predict(self, x):
        return np.full((x.shape[0],), self.mean_value)


class DamagePredictor:
    """A class for predicting damage based on input data and configuration settings.

       This class provides methods for various damage prediction tasks.

       Args:
           config (dict): A dictionary containing configuration settings.

       Attributes:
           CONFIG (dict): A dictionary containing configuration settings.
    """
    def __init__(self, config):
        """
        Initializes a new DamagePredictor instance with the given configuration.

        Args:
            config (dict): A dictionary containing configuration settings.
        """
        self.CONFIG = config
        local_folder = os.path.join(CONFIG['exp_folder'])
        DamagePredictor.create_folder(local_folder)

    @staticmethod
    def create_folder(folder_name):
        """
        Create or replace a folder with the given name.

        This method checks if the folder already exists. If it does, it removes the
        existing folder and creates a new one. If it doesn't exist, it creates a
        new folder.

        Args:
            folder_name (str): The name of the folder to be created or replaced.
        """
        if os.path.exists(folder_name):
            pass
        else:
            os.mkdir(folder_name)
            print(f"Created new folder: {folder_name}")

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
                    "A>T": "Total", "G>A": "Total", "C>T": "Total", "A>G": "Total", "T>C": "Total",
                    "A>C": "Total", "C>G": "Total", "C>A": "Total", "T>G": "Total", "T>A": "Total",
                    "G>C": "Total", "G>T": "Total", "A>-": "Total", "T>-": "Total", "C>-": "Total",
                    "G>-": "Total", "->A": "Total", "->T": "Total", "->C": "Total", "->G": "Total",
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

        metadata_file = self.CONFIG['metadata_file']
        metadata_groups = self.CONFIG['batch_metadata_file']
        metadata_df = pd.read_csv(metadata_file, delimiter='\t')
        metadata_df_groups = pd.read_csv(metadata_groups, delimiter='\t' )
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


    def prepare_dataset(self, basic_features=True, verbose=True):
        """Prepare a dataset for damage prediction.

        Args:
            basic_features (bool, optional): Whether to include basic features. Default is True.
            verbose(bool) : Whether to print intermediate statistics.
        Returns:
            pd.DataFrame: The prepared dataset for damage prediction.
        """
        self.basic_features = basic_features
        m_damage_folder = self.CONFIG['mDamage_folder']
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
                features_concat = pd.DataFrame({'label': [age],
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
            print("\n The number of rows for each BAM file : ", len(features))
            print("\nFILTERING OUT NOT PROCESSED DAMAGE FILES ")

        cols_to_convert = full_df.columns.difference(['batch_name', 'bam_name'])
        full_df[cols_to_convert] = full_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
        columns_to_log = []
        for col in full_df.columns:
            if col not in ['label', 'batch_name','bam_name']:
                columns_to_log.append(col)

        full_df['batch_name'] = full_df['batch_name'].astype(str)
        full_df['bam_name'] = full_df['bam_name'].astype(str)
        numeric_cols = full_df.select_dtypes(include='number').columns
        full_df[numeric_cols] = full_df[numeric_cols].fillna(full_df[numeric_cols].median())
        not_corrected_data = full_df
        not_corrected_data.to_csv(os.path.join("data/data_comparison", "batch_effect_main.csv"))
        if verbose:
            print("\nLENGTH OF FULL_DATAFRAME AFTER CLEANING not existing Damage folders", len(full_df))
            print("\nHEAD OF THE DF_FULL\n", full_df.head(10))
        return full_df

    def feature_selection(self,  df, local_folder):

        local_folder = os.path.join(CONFIG['exp_folder'], 'feature_selection')
        DamagePredictor.create_folder(local_folder)
        model_folder = os.path.join(self.CONFIG['exp_folder'], 'best_model')
        DamagePredictor.create_folder(local_folder)
        params_file_path = os.path.join(model_folder, "best_model_params.json")
        best_params = None
        with open(params_file_path, 'rb') as file:
            model_params = json.load(file)
        model = None
        best_params = None
        if model_params["model_name"] == "XGBoost":
            best_params = model_params["best_params"]  # Use the parameters as-is
            model = XGBRegressor(**{key.replace('model__', ''): value for key, value in best_params.items()})
        print(best_params)
        pca_model = load(os.path.join(model_folder, 'best_pca.pkl'))
        n_components = pca_model.n_components_

        print("Number of components for PCA: ", n_components)
        col_names = df.columns
        dict_names = {}

        for name in self.CONFIG['feature_names']:
            arr = []
            for col_name in col_names:
                if col_name.startswith(name + "_"):
                    arr.append(col_name)
                    dict_names[name] = arr
        groups = df['batch_name']
        y = df['label']
        diff_arr = []
        diff_std = []
        feature_names_df = [feature for feature in self.CONFIG['feature_names'] if feature != 'Total']
        best_performance = 0
        for name in ["NO_PERMUTE"] + feature_names_df:
            print(f"******** Analysis of the property ************ {name}")
            df_permute = df.copy()
            if name == "NO_PERMUTE":
                pass
            else:
                for col_name in dict_names[name]:
                    df_permute[col_name] = np.random.permutation(df_permute[col_name])
            X_permute = df_permute.drop(columns=['label', 'batch_name'])
            scaler = StandardScaler()
            pca = PCA(n_components=n_components)
            pipeline = Pipeline([('scaler', scaler), ('pca', pca), ('model', model)])
            param_grid = {key: [value] for key, value in best_params.items()}
            mae = []
            std = []
            N_exp = 12
            for _ in range(0, N_exp):
                grid = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    cv=self.CONFIG['cv'],
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                grid.fit(X_permute, y, groups=groups)
                mae.append(-grid.cv_results_['mean_test_score'])
                std.append(grid.cv_results_['std_test_score'])

            if name == "NO_PERMUTE":
                best_performance = np.mean(mae)
            else:
                difference = np.mean(mae) - best_performance
                std_difference = np.std(mae, ddof=1)
                diff_arr.append(difference)
                diff_std.append(std_difference)
                print(f"\nFor the {name}, the average difference in performance is { difference, std_difference }")

        df_result = pd.DataFrame({'Feature name': feature_names_df, 'Difference': diff_arr, 'Difference_std': diff_std})
        df_result.to_csv(os.path.join(local_folder, "feature_selection.csv"))

    def partial_data_model(self, df, local_folder):
        """Evaluate a model by training and testing with varying percentages of the dataset.

        Args:
            df (pd.DataFrame): The dataset for model evaluation.
            local_folder (str): The local folder to store results.

        Returns:
            None
        """
        model_folder = os.path.join(self.CONFIG['exp_folder'], local_folder)
        DamagePredictor.create_folder(model_folder)
        params_file_path = os.path.join(os.path.join(self.CONFIG['exp_folder'], "best_model"), "best_model_params.json")
        with open(params_file_path, 'rb') as file:
            model_params = json.load(file)
        model = None
        best_params = None
        if model_params["model_name"] == "XGBoost":
            best_params = model_params["best_params"]
            model = XGBRegressor(**{key.replace('model__', ''): value for key, value in best_params.items()})

        print("Number of components for PCA: ", 70)
        mae_arr_part = []
        std_arr_part = []
        percentage_arr = [30, 40, 50, 60, 70, 80, 90, 100]

        for part in percentage_arr:
            print(f"******* Calculating for {part}% of the dataset *************")
            unique_batches = df['batch_name'].unique()
            sampled_batches = np.random.choice(
                unique_batches,
                size=int(len(unique_batches) * part / 100),
                replace=False
            )
            df_partial = df[df['batch_name'].isin(sampled_batches)]
            groups = df_partial['batch_name']
            X_partial = df_partial.drop(columns=['label', 'batch_name'])
            y_partial = df_partial['label']

            pca_arr = [item for item in self.CONFIG['pca_arr'] if item < len(X_partial)]
            pca_mae_arr = []
            pca_mae_std_arr = []

            for n_components in pca_arr:
                print(f"Calculating for pca {n_components}")
                mae = []

                n_groups = len(np.unique(groups))
                n_splits = min(self.CONFIG['cv'], n_groups)
                if n_splits < 2:
                    print(f"Skipping PCA={n_components} for {part}%: only {n_groups} groups available.")
                    continue

                for i in range(10):
                    scaler = StandardScaler()
                    pca = PCA(n_components=n_components)
                    param_grid = {key: [value] for key, value in best_params.items()}
                    pipeline = Pipeline([('scaler', scaler), ('pca', pca), ('model', model)])
                    grid = GridSearchCV(
                        pipeline,
                        param_grid=param_grid,
                        cv=GroupKFold(n_splits=n_splits),
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1
                    )
                    grid.fit(X_partial, y_partial, groups=groups)
                    mae.append(-grid.cv_results_['mean_test_score'])

                mae_mean = np.mean(mae)
                mae_std = np.std(mae, ddof=1)
                print(f"Run {part}, pca {n_components},  MAE: {mae_mean}")
                pca_mae_arr.append(mae_mean)
                pca_mae_std_arr.append(mae_std)

            if pca_mae_arr:
                min_value_index = np.argmin(pca_mae_arr)
                min_pca_mean = pca_mae_arr[min_value_index]
                corresponding_std = pca_mae_std_arr[min_value_index]
                mae_arr_part.append(min_pca_mean)
                std_arr_part.append(corresponding_std)
                print(f"Best for all pca: Percentage {part}%: Mean MAE: {min_pca_mean}, Std MAE: {corresponding_std}")
            else:
                print(f"Skipping result collection for {part}% due to insufficient PCA evaluations.")
                mae_arr_part.append(None)
                std_arr_part.append(None)

        df_results = pd.DataFrame({'percentage': percentage_arr, 'mae': mae_arr_part, 'mae_std': std_arr_part})
        df_results.to_csv(os.path.join(model_folder, "mae.csv"))

    @staticmethod
    def create_random(df, local_folder, val_frac=0.2, seed=42):
        """
        Create and evaluate a random regressor and an average regressor model,
        ensuring groups (batch_name) are fully in train or val.

        Args:
            df (pd.DataFrame): The dataset for model evaluation.
            local_folder (str): The local folder to store results.
            val_frac (float): Fraction of groups to use for validation.
            seed (int): Random seed for reproducibility.

        Returns:
            None
        """
        print("Creating random and average regressor")

        # Prepare directories
        df_perm = df.copy()
        local_full_folder = os.path.join(CONFIG['exp_folder'], local_folder)
        DamagePredictor.create_folder(local_full_folder)

        # ----------------------
        # Group-aware splitting
        # ----------------------
        unique_batches = df['batch_name'].unique()
        train_batches, val_batches = train_test_split(
            unique_batches,
            test_size=val_frac,
            random_state=seed
        )

        df_perm['split'] = df_perm['batch_name'].apply(
            lambda x: 'val' if x in val_batches else 'train'
        )

        # -----------------------------------
        # Random regressor evaluation on val
        # -----------------------------------
        mae_arr_random = []
        for i in range(5):
            df_perm_val = df_perm[df_perm['split'] == 'val'].copy()
            df_perm_val['permuted_label'] = np.random.permutation(df_perm_val['label'].values)
            df_perm_val['difference'] = (df_perm_val['permuted_label'] - df_perm_val['label']).abs()
            mae_random = df_perm_val['difference'].mean()
            mae_arr_random.append(mae_random)

        mae_random = np.mean(mae_arr_random)

        # -------------------------------------
        # Average regressor evaluation on val
        # -------------------------------------
        average_label = df_perm[df_perm['split'] == 'train']['label'].mean()
        df_perm_val = df_perm[df_perm['split'] == 'val'].copy()
        df_perm_val['average_label'] = average_label
        df_perm_val['difference'] = (df_perm_val['average_label'] - df_perm_val['label']).abs()
        mae_average = df_perm_val['difference'].mean()

        # ---------------------
        # Save results to CSV
        # ---------------------
        csv_path = os.path.join(local_full_folder, "random_and_average_regressor.csv")
        results = {
            "Name": ["Random Regressor", "Average Regressor"],
            "MAE": [mae_random, mae_average]
        }
        df_results = pd.DataFrame(results)
        df_results.to_csv(csv_path, index=False)

        print(f"Random regressor MAE: {mae_random}")
        print(f"Average regressor MAE: {mae_average}")
        return None

    def evaluate_models_with_pca(self, df, CONFIG, local_folder, seed=42, save_model=True, verbose=True):
        X = df.drop(columns=['label', 'batch_name'])
        y = df['label']
        groups = df['batch_name']
        print("Length of samples:", len(groups))
        print("Number of unique groups: ", groups.nunique())
        corrected_X = X # skip the pycombat

        models = [
            ('AverageModel', AverageModel(), {}),
            ('LinearRegression', LinearRegression(), {}),
            ('Ridge', Ridge(), {
                'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr']
            }),
            ('Lasso', Lasso(), {
                'alpha': [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0],
                'max_iter': [100, 200, 500, 1000]
            }),
            ('ElasticNet', ElasticNet(), {
                'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
                'l1_ratio': [0.2, 0.5, 0.7, 1.0],
                'max_iter': [100, 200, 500, 1000]
            }),
            ('RandomForest', RandomForestRegressor(), {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt'],
                'bootstrap': [True, False],
                'random_state': [seed]
            }),
            ('GradientBoosting', GradientBoostingRegressor(), {
                'n_estimators': [50, 100, 150, 200],
                'learning_rate': [0.01, 0.03, 0.05, 0.1],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 4, 6, 8],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 1.0],
                'random_state': [seed]
            }),
            ('SVR', SVR(), {
                'C': [0.1, 0.5, 1, 5.0, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }),
            ('KNN', KNeighborsRegressor(), {
                'n_neighbors': [1, 2, 3, 5, 7],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }),
            ('BayesianRidge', BayesianRidge(), {
                'alpha_1': [1e-6, 1e-4, 1e-2, 1],
                'alpha_2': [1e-6, 1e-4, 1e-2, 1],
                'lambda_1': [1e-6, 1e-4, 1e-2, 1],
                'lambda_2': [1e-6, 1e-4, 1e-2, 1]
            }),
            ('XGBoost', XGBRegressor(), {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'gamma': [0, 0.1],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 10],
                'random_state': [seed]
            }),
            ('DecisionTree', DecisionTreeRegressor(), {
                'max_depth': [None, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': [None, 'sqrt'],
                'splitter': ['best', 'random'],
                'random_state': [seed]
            })
        ]

        best_mae = float('inf')
        best_model = None
        best_scaler = None
        best_pca = None
        best_model_params = None

        model_folder = os.path.join(CONFIG['exp_folder'], "best_model")
        DamagePredictor.create_folder(model_folder)
        pca_main_folder = os.path.join(CONFIG['exp_folder'], local_folder)
        DamagePredictor.create_folder(pca_main_folder)

        scoring_mae = make_scorer(mean_absolute_error, greater_is_better=False)
        scoring_rmse = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False)

        for n_components in CONFIG['pca_arr']:
            pca_folder = os.path.join(CONFIG['exp_folder'], local_folder, f'PCA_{n_components}')
            DamagePredictor.create_folder(pca_folder)
            result_arr = []

            for model_name, model, params in models:
                scaler = StandardScaler()
                pca = PCA(n_components=n_components)
                pipeline = Pipeline([('scaler', scaler), ('pca', pca), ('model', model)])
                kf = GroupKFold(n_splits=CONFIG['cv'])
                grid = GridSearchCV(pipeline, param_grid={'model__' + key: value for key, value in params.items()},
                                    cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
                grid.fit(corrected_X, y, groups=groups)
                best_pipeline = grid.best_estimator_

                mae_scores = -cross_val_score(best_pipeline, corrected_X, y, cv=kf, groups=groups, scoring=scoring_mae)
                rmse_scores = -cross_val_score(best_pipeline, corrected_X, y, cv=kf, groups=groups,
                                               scoring=scoring_rmse)
                r2_scores = cross_val_score(best_pipeline, corrected_X, y, cv=kf, groups=groups, scoring='r2')

                result_arr.append({
                    'model': model_name,
                    'mae': mae_scores.mean(), 'mae_std': mae_scores.std(),
                    'rmse': rmse_scores.mean(), 'rmse_std': rmse_scores.std(),
                    'r2': r2_scores.mean(), 'r2_std': r2_scores.std()
                })

                if mae_scores.mean() < best_mae:
                    best_mae = mae_scores.mean()
                    best_model = best_pipeline
                    best_scaler = scaler
                    best_pca = pca
                    best_model_params = {'model_name': model_name, 'best_params': grid.best_params_,
                                         'pca_components': n_components}

                if verbose:
                    print(f"Model: {model_name} | PCA components: {n_components} | "
                          f"MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f} | "
                          f"RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f} | "
                          f"R2: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

            df_results = pd.DataFrame(result_arr)
            results_file = os.path.join(pca_folder, 'results.csv')
            df_results.to_csv(results_file, index=False)

        if save_model and (best_model is not None):
            with open(os.path.join(model_folder, "best_model_params.json"), 'w') as f:
                json.dump(best_model_params, f)
            best_scaler.fit(corrected_X)
            X_scaled = best_scaler.transform(corrected_X)
            best_pca.fit(X_scaled)
            X_pca = best_pca.transform(X_scaled)
            best_model.named_steps['model'].fit(X_pca, y)
            joblib.dump(best_scaler, os.path.join(model_folder, "best_scaler.pkl"))
            joblib.dump(best_pca, os.path.join(model_folder, "best_pca.pkl"))
            joblib.dump(best_model.named_steps['model'], os.path.join(model_folder, "best_model.pkl"))

        return None

    """
    def perform_batch_normalization(self, CONFIG, df_features):
        print("Removing batch effects... This could take a while")
        batch_correction_folder = os.path.join(CONFIG['exp_folder'], 'batch_correction')
        DamagePredictor.create_folder(batch_correction_folder)
        df_features_tsne = df_features.copy()
        draw_tsne(df_features_tsne, os.path.join(batch_correction_folder, "TSNE_pre_normalization.png"))
        data = df_features.drop(columns=['label', 'batch_name']).values.T
        meta_data = pd.DataFrame({'batch': df_features['batch_name']})
        vars_use = ['batch']
        theta = 5.0
        max_iter_harmony = 10
        harmony_out = hm.run_harmony(
            data,
            meta_data,
            vars_use=vars_use,
            theta=theta,
            max_iter_harmony=max_iter_harmony
        )
        df_corrected_features = pd.DataFrame(harmony_out.Z_corr.T,
                                             columns=df_features.drop(columns=['label', 'batch_name']).columns)
        df_corrected = pd.concat(
            [df_features[['label', 'batch_name']].reset_index(drop=True), df_corrected_features.reset_index(drop=True)],
            axis=1)
        draw_tsne(CONFIG, df_corrected, os.path.join(batch_correction_folder, "TSNE_post_normalization.png"))
        return df_corrected
    """
    @staticmethod
    def set_random_seed(seed_value=42):
        """Set the random seed for reproducibility.

        Args:
            seed_value (int): The seed value to be used for random number generation. Default is 42.
        """
        print(f"Setting random seed to {seed_value}")
        random.seed(seed_value)
        np.random.seed(seed_value)

    def partition_and_save(self, CONFIG, df, group_column):
        """Partition dataset into train/validation and test, ensuring the same batch_name stays in one set.

        Args:
            df (pd.DataFrame): The dataset to be split.
            group_column (str): The column name to group by (e.g., batch_name).
            CONFIG :  the config json file

        Returns:
            None
        """
        # GroupShuffleSplit to split based on the group_column (batch_name)
        data_folder = os.path.join(self.CONFIG['exp_folder'], "SPLIT_DATA")
        DamagePredictor.create_folder(data_folder)
        gss = GroupShuffleSplit(n_splits=1, train_size=CONFIG['train_val_size'])
        train_val_idx, test_idx = next(gss.split(df, groups=df[group_column]))
        print("Length of train / test : ", len(train_val_idx), len(test_idx))
        df_train_val = df.iloc[train_val_idx]
        df_test = df.iloc[test_idx]
        print("TRAIN_DF names", df_train_val['batch_name'])
        print("TEST_DF names", df_test['batch_name'])

        train_val_file = os.path.join(data_folder, "trainval.csv")
        test_file = os.path.join(data_folder, "test.csv")
        df_train_val.to_csv(train_val_file, index=False)
        df_test.to_csv(test_file, index=False)
        df_train_val.pop('bam_name')
        df_test.pop('bam_name')
        print(f"Train/Validation set saved to {train_val_file} with {len(df_train_val)} records.")
        print(f"Test set saved to {test_file} with {len(df_test)} records.")
        return df_train_val, df_test

if __name__ == '__main__':
    with open("config/config_harvard_60_cv5.json", 'r') as file:
        CONFIG = json.load(file)
    DP = DamagePredictor(CONFIG)
    seed = 7777
    DP.set_random_seed(seed)
    df_features = DP.prepare_dataset(basic_features=True, verbose=True)
    df_features_train_val, df_features_test = DP.partition_and_save(CONFIG, df_features, "batch_name")
    # DP.create_random(df_features, local_folder='RANDOM_REGRESSOR')
    DP.evaluate_models_with_pca(df_features_train_val, CONFIG, "MODEL_COMPARISON",seed=777, save_model=True, verbose=True)
    # DP.feature_selection(df_features_train_val, local_folder='LINEAR_MODEL_FEATURE_SELECTION')
    # DP.partial_data_model(df_features_train_val, local_folder='PARTIAL_EXPERIMENT')
