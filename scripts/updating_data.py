import sys
import os
import pickle
import pandas
import pandas.core.indexes.base


if __name__ == "__main__":
    # 1. Mock the old module structure one last time to read the file
    mock_module = type(sys)("pandas.core.indexes.numeric")
    mock_module.NumericIndex = pandas.core.indexes.base.Index
    mock_module.Int64Index = pandas.core.indexes.base.Index
    sys.modules["pandas.core.indexes.numeric"] = mock_module

    # 2. Define your paths
    RAW_DATA_DIR = "../data/original_raw/"
    for file in os.listdir(RAW_DATA_DIR):
        print(f"Processing file: {file}")
        old_file_path = os.path.join(RAW_DATA_DIR, file)
        new_file_path = os.path.join("../data/raw/", file)
        # 3. Load the old data
        data = pandas.read_pickle(old_file_path)
        # 4. Save it using your current version of Pandas
        data.to_pickle(new_file_path)
        print(f"Success! Saved version-compatible file to: {new_file_path}")
