import os
import random
import numpy as np
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)
logging.info("启用混合精度训练以提升性能。")

def create_tf_dataset(X, y, batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_lstm_model(input_shape, units_layer1=64, dropout_layer1=0.1, units_layer2=96, dropout_layer2=0.1, learning_rate=0.001):
    model = Sequential([
        LSTM(units=units_layer1, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_layer1),
        LSTM(units=units_layer2, return_sequences=False),
        Dropout(dropout_layer2),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error"
    )
    return model

if __name__ == "__main__":
    units_layer1 = 64
    units_layer2 = 96
    dropout_layer1 = 0.1
    dropout_layer2 = 0.1
    learning_rate = 0.001
    batch_size = 128
    epochs = 50
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_dir, "data/processed/lstm_data.npz")
    model_save_path = os.path.join(root_dir, "models/lstm_baseline.h5")
    logging.info("LSTM Baseline Model Hyperparameters:")
    logging.info(f"  LSTM层数: 2")
    logging.info(f"  第一层单元数 units_layer1: {units_layer1}")
    logging.info(f"  第一层Dropout: {dropout_layer1}")
    logging.info(f"  第二层单元数 units_layer2: {units_layer2}")
    logging.info(f"  第二层Dropout: {dropout_layer2}")
    logging.info(f"  学习率 learning_rate: {learning_rate}")
    logging.info(f"  批量大小 batch_size: {batch_size}")
    logging.info(f"  训练轮数 epochs: {epochs}")
    logging.info(f"加载数据: {data_path}")
    data = np.load(data_path)
    n_samples = data["X"].shape[0]
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.9)
    X_train, y_train = data["X"][:train_end], data["y"][:train_end]
    X_val, y_val = data["X"][train_end:val_end], data["y"][train_end:val_end]
    X_test, y_test = data["X"][val_end:], data["y"][val_end:]
    logging.info(f"数据集划分: 总样本数={n_samples}, 训练集={len(X_train)}, 验证集={len(X_val)}, 测试集={len(X_test)}")
    train_dataset = create_tf_dataset(X_train, y_train, batch_size=batch_size)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size=batch_size)
    logging.info("开始训练 LSTM 基线模型...")
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units_layer1=units_layer1,
        dropout_layer1=dropout_layer1,
        units_layer2=units_layer2,
        dropout_layer2=dropout_layer2,
        learning_rate=learning_rate
    )
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=1)
    ensure_directory_exists(os.path.dirname(model_save_path))
    model.save(model_save_path)
    logging.info(f"基线模型已保存至: {model_save_path}")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"测试集评估指标:")
    logging.info(f"  RMSE: {rmse:.6f}")
    logging.info(f"  MAE:  {mae:.6f}")
    logging.info(f"  R²  :  {r2:.6f}")