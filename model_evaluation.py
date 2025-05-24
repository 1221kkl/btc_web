import numpy as np
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}, y_pred

def evaluate_market_conditions(y_true, y_pred, market_condition):
    if market_condition == "bull_market":
        mask = y_true > np.roll(y_true, shift=1)
    elif market_condition == "bear_market":
        mask = y_true < np.roll(y_true, shift=1)
    else:
        raise ValueError("未知的市场情境：仅支持 'bull_market' 和 'bear_market'")
    y_true_condition = y_true[mask]
    y_pred_condition = y_pred[mask]
    if len(y_true_condition) == 0:
        return {"RMSE": None, "MAE": None, "R2": None}
    rmse = np.sqrt(mean_squared_error(y_true_condition, y_pred_condition))
    mae = mean_absolute_error(y_true_condition, y_pred_condition)
    r2 = r2_score(y_true_condition, y_pred_condition)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def plot_results(y_true, y_pred, dates, title, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="真实值", color="blue")
    plt.plot(dates, y_pred, label="预测值", color="orange")
    plt.legend()
    plt.title(title)
    plt.xlabel("日期")
    plt.ylabel("真实收盘价（USD）")
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        ensure_directory_exists(os.path.dirname(save_path))
        delete_file_if_exists(save_path)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_performance_comparison(baseline_metrics, tuned_metrics, save_path):
    metrics_names = ["RMSE", "MAE", "R2"]
    baseline_values = [baseline_metrics["RMSE"], baseline_metrics["MAE"], baseline_metrics["R2"]]
    tuned_values = [tuned_metrics["RMSE"], tuned_metrics["MAE"], tuned_metrics["R2"]]
    def percent_improve(baseline, tuned, higher_better=False):
        if higher_better:
            return 100 * (tuned - baseline) / abs(baseline) if baseline != 0 else 0
        else:
            return 100 * (baseline - tuned) / abs(baseline) if baseline != 0 else 0
    improvements = [
        percent_improve(baseline_values[0], tuned_values[0]),
        percent_improve(baseline_values[1], tuned_values[1]),
        percent_improve(baseline_values[2], tuned_values[2], higher_better=True)
    ]
    x = np.arange(len(metrics_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8,5))
    rects1 = ax.bar(x - width/2, baseline_values, width, label='基础模型')
    rects2 = ax.bar(x + width/2, tuned_values, width, label='调优模型')
    for i, (b, t, imp) in enumerate(zip(baseline_values, tuned_values, improvements)):
        ax.text(i + width/2, t, f'{imp:+.1f}%', ha='center', va='bottom', fontsize=11, color='red')
    ax.set_ylabel('指标值')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_title('模型性能对比及百分比提升')
    ax.legend()
    plt.tight_layout()
    ensure_directory_exists(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_dir, "data/processed/lstm_data.npz")
    model_dir = os.path.join(root_dir, "models/")
    output_dir = os.path.join(root_dir, "output/")
    scaler_path = os.path.join(root_dir, "data/processed/scaler.npz")
    scaler = np.load(scaler_path)
    y_min = scaler["min"][0]
    y_max = scaler["max"][0]
    def inverse_transform(y):
        return y * (y_max - y_min) + y_min
    csv_path = os.path.join(root_dir, "data/processed/bitcoin_data_cleaned.csv")
    df = pd.read_csv(csv_path)
    raw_dates = pd.to_datetime(df["open_time"]).values
    data = np.load(data_path)
    X_test, y_test = data["X"][-1000:], data["y"][-1000:]
    test_dates = raw_dates[-len(data["y"]):][-1000:]
    baseline_model_path = os.path.join(model_dir, "lstm_baseline.h5")
    if not os.path.exists(baseline_model_path):
        raise FileNotFoundError(f"基础模型不存在: {baseline_model_path}")
    baseline_model = load_model(baseline_model_path)
    baseline_metrics, y_pred_baseline = evaluate_model(baseline_model, X_test, y_test)
    tuned_model_path = os.path.join(model_dir, "tuned_lstm_model.h5")
    if not os.path.exists(tuned_model_path):
        raise FileNotFoundError(f"调优模型文件不存在: {tuned_model_path}")
    tuned_model = load_model(tuned_model_path)
    tuned_metrics, y_pred_tuned = evaluate_model(tuned_model, X_test, y_test)
    y_test_real = inverse_transform(y_test)
    y_pred_baseline_real = inverse_transform(y_pred_baseline)
    y_pred_tuned_real = inverse_transform(y_pred_tuned)
    print("综合性能对比：")
    print(f"基础模型（lstm_baseline.h5）评估结果: {baseline_metrics}")
    print(f"调优模型（tuned_lstm_model.h5）评估结果: {tuned_metrics}")
    bull_market_metrics_baseline = evaluate_market_conditions(y_test, y_pred_baseline, "bull_market")
    bull_market_metrics_tuned = evaluate_market_conditions(y_test, y_pred_tuned, "bull_market")
    bear_market_metrics_baseline = evaluate_market_conditions(y_test, y_pred_baseline, "bear_market")
    bear_market_metrics_tuned = evaluate_market_conditions(y_test, y_pred_tuned, "bear_market")
    print("\n牛市（Bull Market）性能对比：")
    print(f"基础模型: {bull_market_metrics_baseline}")
    print(f"调优模型: {bull_market_metrics_tuned}")
    print("\n熊市（Bear Market）性能对比：")
    print(f"基础模型: {bear_market_metrics_baseline}")
    print(f"调优模型: {bear_market_metrics_tuned}")
    plot_results(
        y_test_real, y_pred_baseline_real, test_dates, "基础模型真实与预测对比",
        save_path=os.path.join(output_dir, "baseline_results.png")
    )
    plot_results(
        y_test_real, y_pred_tuned_real, test_dates, "调优模型真实与预测对比",
        save_path=os.path.join(output_dir, "tuned_results.png")
    )
    plot_performance_comparison(baseline_metrics, tuned_metrics, os.path.join(output_dir, "performance_compare.png"))
    print(f"\n对比图、性能对比图已保存到 {output_dir}")

    # 假设 y_pred_tuned_real[-1] 为今日预测收盘价
    predicted_today_close = y_pred_tuned_real[-1]
    with open(os.path.join(output_dir, "predicted_today_close.txt"), "w") as f:
        f.write(f"{predicted_today_close:.2f}")