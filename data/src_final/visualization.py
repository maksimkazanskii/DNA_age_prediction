import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import shutil
import warnings
warnings.filterwarnings("ignore")


def create_or_replace_folder(folder_name):
    """
    Create or replace a folder with the given name.

    This method checks if the folder already exists. If it does, it removes the
    existing folder and creates a new one. If it doesn't exist, it creates a
    new folder.

    Args:
        folder_name (str): The name of the folder to be created or replaced.
    """
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        print(f"Removed existing folder: {folder_name}")
    os.mkdir(folder_name)
    print(f"Created new folder: {folder_name}")


def create_dummy_data(num_algorithms=10, num_hyperparameters=5):
    """
    Generates a DataFrame containing dummy data for testing purposes.
    Data represents a set of algorithms and their associated hyperparameter values.

    Args:
        num_algorithms (int): The number of different algorithms to generate data for, defaults to 10.
        num_hyperparameters (int): The number of different hyperparameters per algorithm, defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an algorithm and each column to a hyperparameter set.
    """

    np.random.seed(0)
    data = {
        'Algorithm': [f'Algorithm testv {i}' for i in range(1, num_algorithms + 1)],
    }
    for i in range(1, num_hyperparameters + 1):
        data[f'Hyperparameter Set {i}'] = (np.random.uniform(1000, 2000, num_algorithms))
    df = pd.DataFrame(data)
    df.set_index('Algorithm', inplace=True)
    return df


def create_and_save_plot(df, final_folder):
    """
    Creates and saves a heatmap plot representing algorithm performance based on the provided DataFrame.
    Each cell displays the average MAE and standard deviation as 'value ± std' in a smaller font.

    Args:
        df (pd.DataFrame): The DataFrame containing performance data for each algorithm.
        final_folder (str): The directory path where the plot image will be saved.

    Returns:
        None: The plot is saved to the specified folder.
    """
    print("... Creating algorithm comparison plot")
    plt.figure(figsize=(16, 16))  # Set the figure size
    cmap = 'RdBu'
    vmin = 1200
    vmax = 1800

    numeric_df = df[[col for col in df.columns if "MAE" in col and "std" not in col]]
    annotation_df = numeric_df.copy()

    for i in range(0, df.shape[1], 4):
        mae_col = df.columns[i]
        std_col = df.columns[i + 1]
        r2_col = df.columns[i + 2]
        r2_std_col = df.columns[i + 3]
        annotation_df[mae_col] = (
                r"$\mathbf{" + (df[mae_col].round()).astype(int).astype(str) + " ± " +
                (df[std_col].round()).astype(int).astype(str)
                + "}$" + "\n\n" + df[r2_col].round(3).astype(str) + " ± " + df[r2_std_col].round(3).astype(str)
        )

    annotation_df = annotation_df.transpose()
    numeric_df = numeric_df.transpose()
    ax = sns.heatmap(numeric_df,
                     annot=annotation_df,
                     cmap=cmap,
                     fmt="",  # Required for string formatting in annotations
                     cbar=True,
                     linewidths=0.5,
                     annot_kws={"fontsize": 7},  # Smaller font size for better fit
                     vmin=vmin,
                     vmax=vmax)

    # Customize axis labels and title
    plt.xlabel('Algorithm', labelpad=16, fontsize=16)  # Set fontsize for x-label
    plt.ylabel('Collinear threshold', labelpad=16, fontsize=16)  # Set fontsize for y-label
    plt.title('Algorithm performance (MAE ± std, R² ± std) \n for different number of components of PCA',
              pad=40, fontsize=16)  # Set fontsize for title

    # Set tick label sizes
    plt.xticks(fontsize=16, rotation=75)  # Set fontsize for x-tick labels
    plt.yticks(fontsize=16)  # Set fontsize for y-tick labels

    # Adjust color bar label font size
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=16)  # Set fontsize for color bar ticks
    colorbar.set_label('MAE', fontsize=14)  # Set fontsize for color bar label
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(final_folder, "algorithms_with_mae_std_heatmap.png"), dpi=300)
    plt.clf()


def create_feature_importance_plot(original_csv, final_folder):
    """
    Creates and saves a plot showing the importance of different features
    based on the provided CSV, including standard deviation intervals.

    Args:
        original_csv (str): Path to the CSV file containing features and their
        importance scores.
        final_folder (str): The directory path where the plot image will be saved.

    Returns:
        None: The plot is saved to the specified folder.
    """
    print("... Creating a feature importance plot")
    df = pd.read_csv(original_csv)

    # Extract importance scores and features
    importance_scores = df['Difference']
    importance_std = df['Difference_std']
    feature_names = df['Feature name']

    # Create a DataFrame for sorting
    importance_df = pd.DataFrame({
        'Feature name': feature_names,
        'Importance': importance_scores,
        'Importance_std': importance_std
    })

    # Sort features by importance scores
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Prepare data for plotting
    sorted_features = importance_df['Feature name']
    sorted_importance_scores = importance_df['Importance']
    sorted_std_scores = importance_df['Importance_std']

    plt.figure(figsize=(16, 16))
    fig, ax = plt.subplots()

    # Plotting horizontal bars for importance scores
    ax.barh(sorted_features, sorted_importance_scores, color='orange',
            xerr=sorted_std_scores, capsize=3, edgecolor='black', alpha=0.7)

    # Set labels and titles
    ax.set_xlabel('Importance Score (Difference in MAE)')
    ax.set_title('Permutation Feature Importance with Standard Deviation')
    plt.grid(axis='x', linestyle='--', linewidth=0.5, color="grey")

    # Save the figure
    plt.savefig(os.path.join(final_folder, "feature_importance_with_std.png"), dpi=300)
    plt.clf()


def create_linear_model_plot(df, dummy_performance, final_folder):
    """
    Creates and saves a plot showing the performance of a linear model across different numbers of PCA components.

    Args:
        df (pd.DataFrame): The DataFrame containing the number of components, mean average error (MAE),
         and standard deviation.
        dummy_performance (float): The performance metric of a baseline model for comparison.
        final_folder (str): The directory path where the plot image will be saved.

    Returns:
        None: The plot is saved to the specified folder.
    """
    print("... Creating a linear model plot")
    n_components = df['n_components']
    mae = df['mae']
    std = df['mae_std']
    plt.plot(n_components, mae)
    plt.plot(n_components, mae - std, color='steelblue', linestyle=':', linewidth=1)
    plt.plot(n_components, mae + std, color='steelblue', linestyle=':', linewidth=1)
    plt.axhline(y=dummy_performance, color='red', linestyle='--')
    plt.scatter(n_components, mae, color='steelblue', s=3)
    plt.fill_between(n_components, mae - std, mae + std, alpha=.1)
    plt.xlabel("Number of Components of PCA")
    plt.ylabel("MAE")
    plt.title("The Validation Mean Average Error")
    plt.grid(axis='both', linestyle='--')
    plt.tight_layout()
    plt.ylim((0, 2700))
    plt.savefig(os.path.join(final_folder, "linear_model.png"), dpi=300)
    plt.clf()


def create_algorithm_efficiency(df, dummy_performance, final_folder):
    """
    Creates and saves a plot showing the efficiency of an algorithm based on the provided DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the percentage of original data, MAE, and standard deviation.
        dummy_performance (float): The performance metric of a baseline model for comparison.
        final_folder (str): The directory path where the plot image will be saved.

    Returns:
        None: The plot is saved to the specified folder.
    """
    print("... Creating the algorithm efficiency plot")
    percentage = df['percentage']
    mae = df['mae']
    std = df['mae_std']

    # Set font sizes for consistency
    plt.plot(percentage, mae, color='orange', label='MAE')
    plt.plot(percentage, mae - std, color='orange', linestyle=':', linewidth=1, label='MAE - Std')
    plt.plot(percentage, mae + std, color='orange', linestyle=':', linewidth=1, label='MAE + Std')
    plt.axhline(y=dummy_performance, color='red', linestyle='--', label='Baseline Performance')
    plt.scatter(percentage, mae, color='orange', s=3)
    plt.fill_between(percentage, mae - std, mae + std, alpha=.1, color='orange')

    plt.xlabel("Percentage of the original data")
    plt.ylabel("MAE")
    plt.title("Algorithm Efficiency")
    plt.legend(fontsize=9)  # Include legend to reflect changes
    plt.tight_layout()
    plt.grid(linestyle='--', linewidth=0.5, color="grey")
    plt.ylim((0, 2700))

    plt.savefig(os.path.join(final_folder, "algorithm_efficiency.png"), dpi=300)
    plt.clf()


def get_random_results(random_csv):
    """
    Reads a CSV file containing performance metrics and returns the Mean Average Error (MAE) of a random model.

    Args:
        random_csv (str): Path to the CSV file containing the MAE of the random model.

    Returns:
        float: The MAE of the random model.
    """
    df = pd.read_csv(random_csv)
    random_mae = df['MAE'][0]
    return random_mae


def get_average_results(random_csv):
    """
    Reads a CSV file containing performance metrics and returns the Mean Average Error (MAE) of a random model.

    Args:
        random_csv (str): Pfath to the CSV file containing the MAE of the random model.

    Returns:
        float: The MAE of the random model.
    """
    df = pd.read_csv(random_csv)
    random_mae = df['MAE'][1]
    return random_mae


def create_age_plot(df):
    """
    Generates and saves an age distribution histogram.

    Args:
        df (pandas.DataFrame): Data frame containing 'age' column.
    """
    print("... Creating the age diagram")
    plt.figure(figsize=(10, 6))
    plt.hist(df['Age'], bins=30, color='red', edgecolor='red', alpha=0.3, density=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.xticks(fontsize=10, rotation=30)  # Adjust fontsize and rotation as needed
    plt.grid(axis='y', linestyle='--', linewidth=0.5, color="grey")  # Add horizontal dotted grid lines
    plt.savefig(os.path.join(final_folder, "age.png"), dpi=300)


def parse_the_data(folder_results, dummy_performance):
    """
    Parses result data from a set of folders, compares different models, and returns a formatted DataFrame.

    Args:
        folder_results (str): The directory path containing subfolders with model comparison results.
        dummy_performance (float): The performance metric of a dummy model for comparison.

    Returns:
        pd.DataFrame: A DataFrame where each row represents an algorithm and each column
         represents a different model's performance.
    """
    df_new = pd.DataFrame([])
    for root, dirs, files in os.walk(folder_results):
        for subfolder in dirs:
            subfolder_path = os.path.join(root, subfolder)
            model_results_csv = os.path.join(subfolder_path, "results.csv")
            df_results = pd.read_csv(model_results_csv)
            df_results = df_results.sort_values(by='model', ascending=False)
            algorithm_list = df_results['model'].tolist()
            mae_list = df_results['mae'].tolist()
            r2_list = df_results['r2'].tolist()
            r2_std_list = df_results['r2_std'].tolist()
            mae_std_list = df_results['mae_std'].tolist()
            df_new['Algorithms'] = algorithm_list
            df_new[subfolder + " MAE"] = mae_list
            df_new[subfolder + " MAE std"] = mae_std_list
            df_new[subfolder + " R2"] = r2_list
            df_new[subfolder + " R2 std"] = r2_std_list
    df_new.set_index('Algorithms', inplace=True)
    columns_to_sort = [col for col in df_new.columns
                       if (not ("R2" in col) and not ("std" in col))]
    df_new['Min_Value'] = df_new[columns_to_sort].min(axis=1)
    df_new = df_new.sort_values(by='Min_Value')
    df_new.drop(columns=['Min_Value'], inplace=True)
    column_numbers = [int(col.split(" ")[0].split('_')[1]) for col in df_new.columns]
    sorted_columns = [col for _, col in sorted(zip(column_numbers, df_new.columns))]
    df_new = df_new[sorted_columns]
    return df_new


def create_jitter_plot(csv_file, csv_actual_age, folder_to_write):
    print("Creating jitter plot ...")
    data = pd.read_csv(csv_file)
    x = data.iloc[:, 0].str.split("_").str[0]
    y = data.iloc[:, 1]

    # Step 2: Process Actual values
    df_actual = pd.read_csv(csv_actual_age)
    actual_keys = df_actual['Key'].values
    actual_values = df_actual['Value'].values

    # Check if unique keys match
    x_unique = pd.Series(x).unique()
    if len(actual_keys) != len(x_unique):
        raise ValueError("The number of actual values must match the number of unique BAM files.")

    # Create a mapping and retrieve actual values based on keys
    actual_mapping = dict(zip(actual_keys, actual_values))
    actual_values_mapped = [actual_mapping.get(key, None) for key in x_unique]
    print(actual_values_mapped)
    # Create mapping for unique values
    x_numeric = pd.Series(range(len(x_unique)))
    x_mapping = dict(zip(x_unique, x_numeric))
    x_mapped = x.map(x_mapping)
    jitter_strength = 0.05
    x_jittered = x_mapped + np.random.normal(0, jitter_strength, size=x_mapped.shape)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_jittered, y, alpha=0.7, label='Jittered Predictions', color='steelblue')

    legend_added = False
    print(actual_values_mapped)
    for i, val in enumerate(actual_values_mapped):
        if val is not None:
            plt.hlines(y=val,
                       xmin=i- jitter_strength,
                       xmax=i + jitter_strength,
                       color='red',
                       alpha=0.7,
                       linewidth=4,
                       label='Actual Values' if not legend_added else "")  # Add label only once
            legend_added = True  # Set flag to True after adding the label

    plt.xticks(ticks=np.arange(len(x_unique)), labels=x_unique, rotation=30)
    plt.xlabel("Sample name")
    plt.ylabel("Predicted age")
    plt.title("Jitter Plot with Actual Values")
    plt.grid(linestyle='--', linewidth=0.5, color="grey")

    plt.legend()  # Now this will include the 'Actual Values' if legended properly

    # Create a margin at the bottom for x-axis label
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin

    os.makedirs(folder_to_write, exist_ok=True)
    output_file_path = os.path.join(folder_to_write, "jitter.png")
    plt.savefig(output_file_path, dpi=300)
    plt.close()
    print(f"Jitter plot saved as: {output_file_path}")


def create_reads_plot(file_path, final_folder):
    print("Creating reads plot ...")
    data = pd.read_csv(file_path, sep="\t")
    plt.figure(figsize=(8, 6))

    # Define colors for the lines
    colors = ['steelblue', 'red', 'orange', 'brown']

    for (sample_name, group), color in zip(data.groupby('Sample name'), colors):
        plt.plot(group['Number of reads'], group['Predicted age'],
                 marker='o', alpha=0.7, label=sample_name,
                 color=color, linewidth=3)  # Set line width to 3

    plt.title('Predicted Age vs Number of Reads')
    plt.xlabel('Number of Reads')
    plt.ylabel('Predicted Age')
    plt.xscale('log')  # Using logarithmic scale for better visualization
    plt.legend(title='Sample Name')
    plt.grid(linestyle='--', linewidth=0.5, color="grey")

    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    plt.savefig(os.path.join(final_folder, 'reads_plot.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    with open("config/config_harvard_60_cv5.json", 'r') as file:
        CONFIG = json.load(file)
    final_folder = os.path.join(CONFIG['exp_folder'], "FINAL_REPORT")
    create_or_replace_folder(final_folder)

    reads_filename = os.path.join(CONFIG['exp_folder'], "results_for_article/reads_variation.csv")
    if os.path.exists(reads_filename):
        create_reads_plot(reads_filename,  final_folder)

    jitter_data_file = os.path.join(CONFIG['exp_folder'], "results_for_article/data_jitter.csv")
    jitter_actual_file = os.path.join(CONFIG['exp_folder'], "results_for_article/data_jitter_actual.csv")
    if os.path.exists(jitter_data_file):
        create_jitter_plot(jitter_data_file, jitter_actual_file, final_folder)

    # Get the random regressor performance
    random_csv = os.path.join(CONFIG['exp_folder'], "RANDOM_REGRESSOR/random_and_average_regressor.csv")
    dummy_performance = None
    if os.path.exists(random_csv):
        dummy_performance = get_average_results(random_csv)

    # Creating heatMap.
    folder_results = os.path.join(CONFIG['exp_folder'], "MODEL_COMPARISON")
    if os.path.exists(folder_results):
        df = parse_the_data(folder_results, dummy_performance)
        create_and_save_plot(df, final_folder)

    # Creating feature importance plot
    features_importance_csv = os.path.join(CONFIG['exp_folder'], "feature_selection/feature_selection.csv")
    if os.path.exists(features_importance_csv):
        create_feature_importance_plot(features_importance_csv, final_folder)

    # Creating algorithm efficiency plot
    algorithm_eff_file = os.path.join(CONFIG['exp_folder'], "PARTIAL_EXPERIMENT/mae.csv")
    if os.path.exists(algorithm_eff_file):
        df = pd.read_csv(algorithm_eff_file)
        create_algorithm_efficiency(df, dummy_performance, final_folder)
