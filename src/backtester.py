import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Backtester:
    def __init__(self, initial_capital: float = 10000.0, fee_rate: float = 0.001):
        """
        initial_capital: Vốn ban đầu (VD: 10,000 USD)
        fee_rate: Phí giao dịch mỗi chiều (VD: 0.1% cho Crypto/Chứng khoán)
        """
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate

    def run(self, signals: np.ndarray, actual_returns: np.ndarray):
        """
        signals: Mảng tín hiệu (0: Mua Mạnh, 1: Mua, 2: Đứng Ngoài, 3: Bán/Short)
        actual_returns: Tỷ suất sinh lời thực tế của tài sản trong kỳ hạn đó
        """
        # Ánh xạ tín hiệu thành Vị thế (Position):
        # 0, 1 -> Long (+1) | 2 -> Đứng ngoài (0) | 3 -> Short (-1)
        positions = np.zeros_like(signals)
        positions[signals == 0] = 1
        positions[signals == 1] = 1
        positions[signals == 2] = 0
        positions[signals == 3] = -1

        # Tính toán chi phí giao dịch khi thay đổi vị thế (Turnover)
        position_changes = np.abs(np.diff(positions, prepend=0))
        fees = position_changes * self.fee_rate

        # Lợi nhuận gộp = Vị thế * Lợi nhuận tài sản - Phí giao dịch
        strategy_returns = (positions * actual_returns) - fees

        # Tính toán Đường cong Tài sản (Equity Curve)
        equity_curve = self.initial_capital * np.cumprod(1 + strategy_returns)

        # --- Tính các chỉ số hiệu suất (Metrics) ---
        total_return = (equity_curve[-1] / self.initial_capital) - 1

        # Maximum Drawdown (Mức sụt giảm tối đa)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)

        return {
            "Total Return": total_return,
            "Max Drawdown": max_drawdown,
            "Equity Curve": equity_curve,
            "Strategy Returns": strategy_returns
        }

    def plot_results(self, results: dict):
        plt.figure(figsize=(12, 6))
        plt.plot(results["Equity Curve"], label="Bayesian Bot V1.0", color="blue")
        plt.title("Đường cong Tài sản (Equity Curve)")
        plt.xlabel("Thời gian (Đơn vị: Ngày/Trade)")
        plt.ylabel("Giá trị Tài khoản (USD)")
        plt.legend()
        plt.grid(True)
        plt.show()