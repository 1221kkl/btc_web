import pandas as pd
import os
import logging

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def preprocess_data(input_file, output_file):
    try:
        logging.info(f"加载原始数据文件: {input_file}")
        df = pd.read_csv(input_file)
        df = df.drop_duplicates(subset=["open_time"])
        df = df.sort_values("open_time")
        df = df.fillna(method='ffill').fillna(method='bfill')
        ensure_directory_exists(os.path.dirname(output_file))
        df.to_csv(output_file, index=False)
        logging.info(f"数据预处理完成，输出文件: {output_file}")
    except Exception as e:
        logging.error(f"数据预处理失败: {e}")

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(root_dir, "data/raw/bitcoin_data.csv")
    cleaned_data_path = os.path.join(root_dir, "data/processed/bitcoin_data_cleaned.csv")
    preprocess_data(raw_data_path, cleaned_data_path)