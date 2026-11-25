"""
Data Splitting Module
Partitions cleaned dataset into train (70%), validation (15%), and test (15%) sets
using stratified sampling to avoid data leakage and maintain consistent class distributions.
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def split_hostel_data(input_csv: str = "data/processed/hostel_complaints_cleaned.csv",
                      output_dir: str = "data/processed",
                      seed: int = 42):
    """
    Performs 70/15/15 stratified train/val/test split.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    print(f"Total records to split: {len(df)}")
    
    # 70% train, 30% temp (which gets split into 15% val and 15% test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["category"]
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["category"]
    )
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train split saved: {train_path} ({len(train_df)} rows, {len(train_df)/len(df)*100:.1f}%)")
    print(f"Val split saved:   {val_path} ({len(val_df)} rows, {len(val_df)/len(df)*100:.1f}%)")
    print(f"Test split saved:  {test_path} ({len(test_df)} rows, {len(test_df)/len(df)*100:.1f}%)")
    
    print("\nTest set class distribution:")
    print(test_df["category"].value_counts())
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into 70/15/15 train/val/test splits.")
    parser.add_argument("--input", type=str, default="data/processed/hostel_complaints_cleaned.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    split_hostel_data(input_csv=args.input, output_dir=args.output_dir, seed=args.seed)
