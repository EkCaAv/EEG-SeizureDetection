import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_cleaned_data(folder_path):
    cleaned_csv_path = os.path.join(folder_path, 'cleaned_data.csv')
    return pd.read_csv(cleaned_csv_path)

def visualize_data(df, title, output_path):
    # Correlation Heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation Heatmap of {title} Data')
    plt.savefig(os.path.join(output_path, f'{title}_correlation_heatmap.png'))
    plt.close()

    # Histograms of each feature
    df.hist(figsize=(14, 12), bins=30)
    plt.suptitle(f'Histograms of {title} Data')
    plt.savefig(os.path.join(output_path, f'{title}_histograms.png'))
    plt.close()

    # Boxplot to detect outliers
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df)
    plt.title(f'Boxplots of {title} Data')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output_path, f'{title}_boxplots.png'))
    plt.close()

if __name__ == "__main__":
    base_path = '../../../../data_EGG'
    output_base_path = './output_images'

    # Visualizar datos limpios de personas sanas
    # healthy_folder = 'healthy'
    # healthy_path = os.path.join(base_path, healthy_folder)
    # healthy_output_path = os.path.join(output_base_path, healthy_folder)
    # os.makedirs(healthy_output_path, exist_ok=True)
    
    # if os.path.exists(os.path.join(healthy_path, 'cleaned_data.csv')):
    #     healthy_df = load_cleaned_data(healthy_path)
    #     visualize_data(healthy_df, 'Healthy', healthy_output_path)
    
    # Visualizar datos limpios de personas con convulsiones
    seizures_folder = 'seizures'
    seizures_path = os.path.join(base_path, seizures_folder)
    seizures_output_path = os.path.join(output_base_path, seizures_folder)
    os.makedirs(seizures_output_path, exist_ok=True)

    if os.path.exists(os.path.join(seizures_path, 'cleaned_data.csv')):
        seizures_df = load_cleaned_data(seizures_path)
        visualize_data(seizures_df, 'Seizures', seizures_output_path)
