import numpy as np

class UncertaintyRiskEngine:
    def __init__(self, num_samples: int = 100, uncertainty_threshold: float = 0.05):
        """
        num_samples: Số lượng kịch bản trọng số ảo (Monte Carlo samples).
        uncertainty_threshold: Ngưỡng phương sai tối đa cho phép.
        """
        self.num_samples = num_samples
        self.threshold = uncertainty_threshold

    def predict_with_uncertainty(self, model, Phi_new: np.ndarray):
        """
        Nhận vào mô hình Bayesian đã huấn luyện (chứa W_MAP và S_N) và Dữ liệu mới.
        Trả về: Xác suất trung bình và Độ bất định (Phương sai) của dự đoán.
        """
        M, K = model.W_MAP.shape

        # [TODO 1]: Duỗi thẳng W_MAP thành vector 1D
        W_MAP_flat = model.W_MAP.flatten()

        # [TODO 2]: Lấy mẫu ngẫu nhiên 100 vector trọng số từ phân phối Multivariate Normal
        # Sử dụng: np.random.multivariate_normal
        sampled_weights_flat = np.random.multivariate_normal(W_MAP_flat, model.S_N, self.num_samples) # Kích thước sẽ là (num_samples, M*K)

        predictions = []

        # Chạy 100 kịch bản dự đoán
        for i in range(self.num_samples):
            # [TODO 3]: Lấy vector trọng số thứ i, gấp (reshape) nó lại thành ma trận (M x K)
            W_sampled = sampled_weights_flat[i].reshape(M,K)
            # Tính Logits và Softmax cho kịch bản này
            A_sampled = Phi_new.dot(W_sampled)
            Y_sampled = model._softmax(A_sampled)
            predictions.append(Y_sampled)

        # Chuyển list thành Numpy array kích thước (num_samples, N_new, K)
        predictions = np.array(predictions)

        # [TODO 4]: Tính Xác suất Trung bình (mean) và Độ bất định (variance) dọc theo trục num_samples (axis=0)
        mean_probs = np.mean(predictions, axis=0)
        uncertainty = np.var(predictions, axis=0)

        return mean_probs, uncertainty

    def get_trading_signal(self, mean_probs: np.ndarray, uncertainty: np.ndarray):
        """
        Quyết định MUA (0), MUA NHẸ (1), ĐỨNG NGOÀI (2), BÁN (3)
        dựa trên Xác suất và Độ bất định.
        """
        signals = []
        for i in range(len(mean_probs)):
            prob = mean_probs[i]
            uncert = uncertainty[i]

            # Lấy index của Lớp có xác suất cao nhất
            best_class = np.argmax(prob)

            # Lấy phương sai của chính Lớp đó làm thước đo độ hoang mang
            class_uncert = uncert[best_class]

            # [TODO 5]: Viết Logic Quản trị Rủi ro
            if class_uncert > self.threshold:
            #     Bắt buộc tín hiệu phải là ĐỨNG NGOÀI (Lớp 2) bất chấp xác suất.
                signals.append(2)
            else:
            #     Tín hiệu là best_class.
            # Thêm tín hiệu vào mảng signals.
                signals.append(best_class)

        return np.array(signals)