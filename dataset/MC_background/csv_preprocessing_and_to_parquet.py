# %%
import pandas as pd
import numpy as np
import pandas as pd
import uproot as ur
import glob
import os
import shutil
# pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)


# %%
def find_files(search_dir=".", file_pattern="*.csv"):
    search_pattern = os.path.join(search_dir, file_pattern)
    return sorted(glob.glob(search_pattern))

FILE_PATHS = find_files(".", "output_*.csv")
FILE_PATHS

# %%
WEIGHT_VALUE = 0.000359148056


# %%
# One-hot encodes the l0_id, l1_id (e-/e+) and q0_id, q1_id (dq/uq) columns
def encode_class_columns(df):
    df_encoded = df.copy()
    
    # Encode l0_id and l1_id (11/-11 columns)
    for col in ['l0_id', 'l1_id']:
        df_encoded[f'{col}_e-'] = (df[col] == 11).astype(int)
        df_encoded[f'{col}_e+'] = (df[col] == -11).astype(int)
        
    # Encode q0_id and q1_id (1/2 columns)
    for col in ['q0_id', 'q1_id']:
        df_encoded[f'{col}_dq'] = (df[col] == 1).astype(int)
        df_encoded[f'{col}_uq'] = (df[col] == 2).astype(int)
    
    df_encoded = df_encoded.drop(columns=['l0_id', 'l1_id', 'q0_id', 'q1_id'])
    
    return df_encoded

# %%
problematic_files = []

for file_path in FILE_PATHS:
    print(f"Processing {os.path.basename(file_path)}...")
    
    if 'output_zz_background.csv' in file_path:
        chw_value = 0.0
        chw_str = 'sm'
    else:
        # Extract number after "bsm_" and before ".csv"
        chw_str = file_path.split('_')[-1].replace('.csv','')
        try:
            # Check if it's in the format '0X' or '-0X' (and not just '0')
            if len(chw_str) > 1 and chw_str.startswith(('0', '-0')) and '.' not in chw_str:
                 # Convert '01' -> 0.1, '-01' -> -0.1, etc.
                 chw_value = float(chw_str) / 10.0
            else:
                 # Standard conversion for '1', '-1', '10', etc.
                 chw_value = float(chw_str)
        except ValueError:
             print(f"  Error converting '{chw_str}' to float for file {file_path}. Skipping.")
             problematic_files.append(file_path)
             continue

    print(f"  Extracted string: {chw_str}")
    print(f"  Interpreted cHW value: {chw_value}")

    weight_value = WEIGHT_VALUE # zz background weight

    try:
        df = pd.read_csv(file_path)
        df_encoded = encode_class_columns(df)
        
        df_encoded["weight"] = weight_value
        df_encoded["cHW"] = chw_value
        
        df_encoded.to_csv(file_path, index=False)
        print(f"  Successfully processed and saved (in place): {os.path.basename(file_path)}\n")
        
    except Exception as e:
        print(f"  Error during processing or saving {file_path}: {e}")
        problematic_files.append(file_path)


print("\nProcessing complete.")
if problematic_files:
    print("Problematic files encountered:")
    for f in problematic_files:
        print(f" - {f}")

# %%
import pandas as pd
import glob
import os

csv_directory = ''
csv_pattern = os.path.join(csv_directory, '*.csv')
output_parquet_file = 'zz_background.parquet'

csv_files = glob.glob(csv_pattern)

print(f"Found {len(csv_files)} CSV files to combine.")

all_dfs = []
print("Starting to read CSV files...")
for i, f in enumerate(csv_files):
    df_part = pd.read_csv(f)
    all_dfs.append(df_part)
    print(f"  Read {i+1}/{len(csv_files)}: {os.path.basename(f)}")

print("\nConcatenating DataFrames...")
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"Concatenation complete. Final DataFrame shape: {combined_df.shape}")
print(f"\nWriting combined data to {output_parquet_file}...")
combined_df.to_parquet(output_parquet_file, engine='pyarrow', compression='snappy', index=False)
print("Successfully wrote combined Parquet file.")


