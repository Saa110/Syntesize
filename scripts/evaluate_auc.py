import pandas as pd
from sklearn.metrics import roc_auc_score
import argparse

def calculate_auc(validate_path, submission_path):
    print(f"Reading truth labels from: {validate_path}")
    val_df = pd.read_csv(validate_path, usecols=['SK_ID_CURR', 'TARGET'])
    val_df.rename(columns={'TARGET': 'TRUE_TARGET'}, inplace=True)
    
    print(f"Reading predictions from: {submission_path}")
    sub_df = pd.read_csv(submission_path)
    
    print("Merging data...")
    merged_df = pd.merge(val_df, sub_df, on='SK_ID_CURR', how='inner')
    
    if merged_df.empty:
        print("Error: No matching SK_ID_CURR found between validation and submission files.")
        return
        
    print(f"Successfully matched {len(merged_df)} rows.")
    
    auc_score = roc_auc_score(merged_df['TRUE_TARGET'], merged_df['TARGET'])
    print(f"\nROC AUC Score: {auc_score:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate AUC of predictions against true labels")
    parser.add_argument("--validate", default="data/raw/validate.csv", help="Path to the validation CSV (with true TARGET)")
    parser.add_argument("--submission", default="data/submissions/submission.csv", help="Path to the submission CSV (with predicted TARGET)")
    args = parser.parse_args()
    
    calculate_auc(args.validate, args.submission)
