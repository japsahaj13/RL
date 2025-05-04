import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = 'data/retail_store_inventory.csv'

# Load data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        print(f"ERROR: Data file not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df

def print_summary(df):
    print("\n--- Summary Statistics ---")
    print(df.describe(include='all'))
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Number of Samples per Category ---")
    print(df['Category'].value_counts())

def check_zeros_negatives(df):
    print("\n--- Zero/Negative Value Checks ---")
    for col in ['Units Sold', 'Price', 'Competitor Pricing']:
        zeros = (df[col] == 0).sum()
        negatives = (df[col] < 0).sum()
        print(f"{col}: {zeros} zeros, {negatives} negatives")
        if zeros > 0 or negatives > 0:
            print(f"WARNING: {col} has {zeros} zeros and {negatives} negatives!")

def check_outliers(df):
    print("\n--- Outlier Checks (using 1.5*IQR rule) ---")
    for col in ['Units Sold', 'Price', 'Competitor Pricing']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"{col}: {outliers} outliers (lower={lower:.2f}, upper={upper:.2f})")
        if outliers > 0:
            print(f"WARNING: {col} has {outliers} outliers!")

def plot_histograms(df):
    for col in ['Units Sold', 'Price', 'Competitor Pricing']:
        plt.figure(figsize=(8, 4))
        plt.hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"hist_{col.replace(' ', '_').lower()}.png")
        plt.close()
        print(f"Saved histogram for {col} as hist_{col.replace(' ', '_').lower()}.png")

def check_duplicates(df):
    print("\n--- Duplicate Checks ---")
    dups = df.duplicated().sum()
    print(f"Number of duplicate rows: {dups}")
    if dups > 0:
        print("WARNING: There are duplicate rows in the data!")

def main():
    df = load_data()
    if df is None:
        return
    print_summary(df)
    check_zeros_negatives(df)
    check_outliers(df)
    check_duplicates(df)
    plot_histograms(df)
    print("\nData profiling complete. Please review the warnings and plots for data quality issues.")

if __name__ == "__main__":
    main() 