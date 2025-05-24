import pandas as pd
import os
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def add_technical_indicators(df):
    logging.info("添加技术指标...")
    df['MA_7'] = df['close'].rolling(window=7).mean()
    df['MA_30'] = df['close'].rolling(window=30).mean()
    df['EMA_7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Volatility'] = df['close'].rolling(window=14).std()
    df.fillna(method='bfill', inplace=True)
    logging.info("技术指标添加完成")
    return df

def prepare_lstm_data(df, lookback=30):
    logging.info("准备 LSTM 输入数据...")
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'MA_7', 'MA_30', 'EMA_7', 'EMA_30', 'RSI', 'MACD', 'Volatility'
    ]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, 3])
    logging.info("LSTM 输入数据准备完成")
    return np.array(X), np.array(y), scaler

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_data_path = os.path.join(root_dir, "data/processed/bitcoin_data_cleaned.csv")
    lstm_data_path = os.path.join(root_dir, "data/processed/lstm_data.npz")
    scaler_path = os.path.join(root_dir, "data/processed/scaler.npz")
    try:
        logging.info(f"加载清理后的数据: {cleaned_data_path}")
        df = pd.read_csv(cleaned_data_path)
        df = add_technical_indicators(df)
        X, y, scaler = prepare_lstm_data(df)
        output_dir = os.path.dirname(lstm_data_path)
        ensure_directory_exists(output_dir)
        np.savez(lstm_data_path, X=X, y=y)
        logging.info(f"LSTM 数据已保存到: {lstm_data_path}")
        scaler_params = {"min": scaler.data_min_, "max": scaler.data_max_}
        np.savez(scaler_path, **scaler_params)
        logging.info(f"Scaler已保存到: {scaler_path}")
    except Exception as e:
        logging.error(f"特征工程失败: {e}")