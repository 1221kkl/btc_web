import os
import logging
import pandas as pd
import time
import re

from binance.client import Client

# 兼容新旧 python-binance 异常类型
try:
    from binance.error import ClientError, APIError
except ImportError:
    from binance.exceptions import BinanceAPIException as APIError
    ClientError = APIError

RAW_DATA_DIR = "./data/raw/"
RAW_DATA_FILE = "bitcoin_data.csv"
PROXY_ENABLE = True
PROXY_HTTP = "http://127.0.0.1:7890"
PROXY_HTTPS = "http://127.0.0.1:7890"
BINANCE_API_KEY = "fmUcrtCCC6mvq1hLeQ5yIbqlEsJgyt0KKREQG3j92FHSvywThvz8t3aneN2EVjRy"      # ← 填写你的API Key
BINANCE_API_SECRET = "cjQ1Fmq9aH9NsJDAgOZrWbJvLLt5xznjLydWAcMYNqPq5kyU2TiK4UI3hqfvcGNE"  # ← 填写你的API Secret

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def wait_until_unban(message):
    ts = re.search(r"banned until (\d+)", message)
    if ts:
        ban_until = int(ts.group(1)) / 1000  # ms转为秒
        now = time.time()
        wait_seconds = int(ban_until - now) + 1
        if wait_seconds > 0:
            logging.warning(f"IP已被Binance封禁，自动等待{wait_seconds}秒后重试...")
            time.sleep(wait_seconds)
            return True
    return False

def collect_data():
    logging.info("启动数据收集脚本...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, RAW_DATA_DIR, RAW_DATA_FILE)

    try:
        ensure_directory_exists(os.path.join(base_dir, RAW_DATA_DIR))
        logging.info(f"目标目录已存在或创建成功: {RAW_DATA_DIR}")
    except Exception as e:
        logging.error(f"创建目录失败: {e}")
        return

    try:
        if delete_file_if_exists(raw_data_path):
            logging.info(f"旧数据文件已删除: {raw_data_path}")
        else:
            logging.debug(f"旧数据文件不存在，无需删除: {raw_data_path}")
    except Exception as e:
        logging.error(f"删除旧文件失败: {e}")
        return

    # 只设置proxies，不加headers
    requests_params = {}
    if PROXY_ENABLE:
        requests_params["proxies"] = {
            "http": PROXY_HTTP,
            "https": PROXY_HTTPS
        }

    client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, requests_params=requests_params)

    symbol = "BTCUSDT"
    interval = Client.KLINE_INTERVAL_1DAY
    start_str = "1 Jan, 2017"
    end_str = "31 Dec, 2025"

    logging.info("开始从 Binance API 获取数据...")
    all_data = []
    error_count = 0
    max_error_count = 30  # 遇到非ban异常也可多试几次

    while True:
        try:
            klines = client.get_historical_klines(symbol, interval, start_str, end_str)
            if not klines:
                logging.error("未获取到任何数据，脚本终止。")
                return
            all_data = klines
            logging.info(f"成功获取到 {len(all_data)} 条K线数据")
            break
        except (APIError, ClientError) as e:
            msg = str(e)
            logging.error(f"Binance API异常: {msg}")
            # 自动等待ban解封
            if "banned until" in msg:
                if wait_until_unban(msg):
                    continue
            # 其它高频请求异常也自动重试
            if any(code in msg for code in ["418", "429", "-1003"]):
                error_count += 1
                if error_count > max_error_count:
                    logging.error("连续多次 418/429/-1003 错误，终止爬取。")
                    return
                time.sleep(10)
                continue
            else:
                return
        except Exception as e:
            logging.error(f"未知异常: {e}")
            error_count += 1
            if error_count > max_error_count:
                logging.error("连续多次未知异常，终止爬取。")
                return
            time.sleep(10)
            continue

    logging.info("开始处理数据...")
    try:
        processed_data = [
            {
                "open_time": entry[0],
                "open": entry[1],
                "high": entry[2],
                "low": entry[3],
                "close": entry[4],
                "volume": entry[5],
            }
            for entry in all_data
        ]
        df = pd.DataFrame(processed_data)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.to_csv(raw_data_path, index=False)
        logging.info(f"数据已成功保存到: {raw_data_path}")
    except Exception as e:
        logging.error(f"数据处理失败: {e}")

if __name__ == "__main__":
    try:
        collect_data()
    except Exception as e:
        logging.error(f"程序运行时发生未知错误: {e}")