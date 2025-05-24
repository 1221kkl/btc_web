import os
import random
import numpy as np
os.environ['PYTHONHASHSEED'] = '98'
random.seed(98)
np.random.seed(98)
import tensorflow as tf
tf.random.set_seed(98)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
import logging
from keras_tuner.tuners import BayesianOptimization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import shutil
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def configure_only_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logging.error("未检测到可用的 GPU，将使用 CPU（模型性能可能受限）。")
        return
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logging.info(f"已禁用 CPU，仅使用 GPU: {gpus[0].name}")
    except RuntimeError as e:
        logging.error(f"无法配置 GPU: {e}")

configure_only_gpu()

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)
logging.info("启用混合精度训练以提升性能。")

def create_tf_dataset(X, y, batch_size=128):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model(hp, input_shape):
    model = Sequential([
        LSTM(hp.Int("units_layer1", min_value=32, max_value=128, step=16), return_sequences=True, input_shape=input_shape),
        Dropout(hp.Float("dropout_layer1", min_value=0.1, max_value=0.5, step=0.1)),
        LSTM(hp.Int("units_layer2", min_value=32, max_value=128, step=16), return_sequences=False),
        Dropout(hp.Float("dropout_layer2", min_value=0.1, max_value=0.5, step=0.1)),
        Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5])),
        loss="mean_squared_error"
    )
    return model

if __name__ == "__main__":
    # 只用当前目录作为根目录
    root_dir = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(root_dir, "data/processed/lstm_data.npz")
    model_dir = os.path.join(root_dir, "models/")
    tuner_dir = os.path.join(root_dir, "models/tuning_logs/")
    model_save_path = os.path.join(model_dir, "tuned_lstm_model.h5")

    logging.info(f"加载数据: {data_path}")
    data = np.load(data_path)
    n_samples = data["X"].shape[0]
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.9)
    X_train, y_train = data["X"][:train_end], data["y"][:train_end]
    X_val, y_val = data["X"][train_end:val_end], data["y"][train_end:val_end]
    X_test, y_test = data["X"][val_end:], data["y"][val_end:]

    logging.info(f"数据集划分: 总样本数={n_samples}, 训练集={len(X_train)}, 验证集={len(X_val)}, 测试集={len(X_test)}")

    train_dataset = create_tf_dataset(X_train, y_train, batch_size=128)
    val_dataset = create_tf_dataset(X_val, y_val, batch_size=128)

    ensure_directory_exists(model_dir)

    # 每次都自动重新搜索（不再询问，直接清理旧的调优状态）
    if os.path.exists(tuner_dir):
        shutil.rmtree(tuner_dir)
        logging.info(f"旧的调优状态文件已清理: {tuner_dir}")

    tuner = BayesianOptimization(
        lambda hp: build_model(hp, input_shape=(X_train.shape[1], X_train.shape[2])),
        objective="val_loss",
        max_trials=20,
        executions_per_trial=3,
        directory=tuner_dir,
        project_name="lstm_hyperparameter_tuning_bayesian"
    )
    tuner.search(train_dataset, epochs=50, validation_data=val_dataset)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_params = {
        "units_layer1": best_hps.get("units_layer1"),
        "dropout_layer1": best_hps.get("dropout_layer1"),
        "units_layer2": best_hps.get("units_layer2"),
        "dropout_layer2": best_hps.get("dropout_layer2"),
        "learning_rate": best_hps.get("learning_rate")
    }
    logging.info("LSTM Tuning Best Hyperparameters:")
    logging.info(f"  第一层单元数 units_layer1: {best_params['units_layer1']}")
    logging.info(f"  第一层Dropout: {best_params['dropout_layer1']}")
    logging.info(f"  第二层单元数 units_layer2: {best_params['units_layer2']}")
    logging.info(f"  第二层Dropout: {best_params['dropout_layer2']}")
    logging.info(f"  学习率 learning_rate: {best_params['learning_rate']}")
    logging.info("使用最佳超参数训练最终模型...")
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_dataset, epochs=50, validation_data=val_dataset, verbose=1)
    if os.path.exists(model_save_path):
        os.remove(model_save_path)
        logging.info(f"已删除旧的超参数优化模型: {model_save_path}")
    model.save(model_save_path)
    logging.info(f"调优后的模型已保存至: {model_save_path}")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"测试集评估指标:")
    logging.info(f"  RMSE: {rmse:.6f}")
    logging.info(f"  MAE:  {mae:.6f}")
    logging.info(f"  R²  :  {r2:.6f}")