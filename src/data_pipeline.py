import yfinance as yf
import numpy as np
import pandas as pd

class FinancialDataPipeline:
        def __init__(self):
            pass
        def fetch_data(self, ticker, start_date, end_date):
            try:
                data = yf.download(
                    tickers=ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    multi_level_index=False,
                )

                if data.empty:
                    print(f"WARNING: Không có dữ liệu cho mã {ticker}")
                    return pd.DataFrame()
                data = data.dropna()
                return data

            except Exception as e:
                print(f"ERROR: Lỗi khi tải dữ liệu: {e}")
                return pd.DataFrame()

        def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Nhận vào DataFrame chứa OHLCV.
            Trả về DataFrame đã được tính toán thêm các cột Features (Phi).
            """
            # Tránh can thiệp trực tiếp vào df gốc
            data = df.copy()

            # 1. Feature 1: Log Return
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))

            # 2. Feature 2: RSI 14
            delta = data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-8)
            data['RSI_14'] = 100 - (100 / (1 + rs))

            # 3. Feature 3: Volume Z-Score 20
            vol_mean = data['Volume'].rolling(window=20).mean()
            vol_std = data['Volume'].rolling(window=20).std()
            data['Vol_ZScore_20'] = (data['Volume'] - vol_mean) / (vol_std + 1e-8)

            # 4. Cắt bỏ các hàng bị thiếu dữ liệu do độ trễ của hàm rolling/shift
            data = data.dropna()

            return data

        def compute_labels(self, df):
            data = df.copy()
            data['Future_Return'] = (data['Close'].shift(-3) - data['Close']) / data['Close']
            data = data.dropna()

            conditions = [
                (data['Future_Return'] >= 0.05),
                (data['Future_Return'] >= 0.0) & (data['Future_Return'] < 0.05),
                (data['Future_Return'] >= -0.05) & (data['Future_Return'] < 0.0),
                (data['Future_Return'] < -0.05)
            ]

            choices = [0, 1, 2, 3]
            data['Label_Class'] = np.select(conditions, choices, default=1)
            one_hot = pd.get_dummies(data['Label_Class'], prefix='Class').astype(int)
            data = pd.concat([data, one_hot], axis=1)
            return data
        def prepare_matrices(self, data, test_size):
            N = len(data)

            split_idx = int(N * (1 - test_size))

            feature_cols = ['Log_Return', 'RSI_14', 'Vol_ZScore_20']

            label_cols = ['Class_0', 'Class_1', 'Class_2', 'Class_3']

            X = data[feature_cols].values
            y = data[label_cols].values

            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            train_mean = np.mean(X_train, axis=0)
            train_std = np.std(X_train, axis=0) + 1e-8

            X_train_scaled = (X_train - train_mean) / train_std
            X_test_scaled = (X_test - train_mean) / train_std

            Phi_train = np.c_[np.ones(len(X_train_scaled)), X_train_scaled]
            Phi_test = np.c_[np.ones(len(X_test_scaled)), X_test_scaled]

            return Phi_train, y_train, Phi_test, y_test

if __name__ == "__main__":
    financial_data_pipeline = FinancialDataPipeline()
    financial_data_pipeline = FinancialDataPipeline()
    data = financial_data_pipeline.fetch_data("AAPL", "2024-01-01", "2026-03-01")
    print(len(data))
    features = financial_data_pipeline.compute_features(data)

    data_clean = financial_data_pipeline.compute_labels(features)
    print(data_clean.columns)

    Phi_train, T_train, Phi_test, T_test = financial_data_pipeline.prepare_matrices(data_clean, 0.2)

    print(Phi_train.shape)
    print(T_train.shape)

