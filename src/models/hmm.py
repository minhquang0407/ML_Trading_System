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

        def compute_features(self, data):
            p

        def compute_labels(self, data):
            pass

        def prepare_matrices(self, data, test_size):
            pass

financial_data_pipeline = FinancialDataPipeline()
data = financial_data_pipeline.fetch_data("AAPL", "2024-01-01", "2026-03-01")
print(data.head())