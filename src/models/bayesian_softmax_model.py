import numpy as np
from src.models.base_model import BaseModel


class BayesianSoftmaxRegression(BaseModel):
    def __init__(self, alpha: float = 1.0, learning_rate: float = 0.01, max_iter: float = 1000):
        """
        alpha: Hệ số phạt L2 (Precision của Gaussian Prior). 
               Alpha càng lớn, Prior càng kéo trọng số về 0 (chống Overfit).
        """
        self.alpha = alpha
        self.lr = learning_rate
        self.max_iter = max_iter

        self.W_MAP = None  # Ma trận trọng số tối ưu (M x K)
        self.S_N = None  # Ma trận Hiệp phương sai Hậu nghiệm sau khi Flatten W thành (MK x MK)
        self.loss_history = []

    def _softmax(self, A: np.ndarray) -> np.ndarray:
        """ Tính Softmax với thủ thuật ổn định số học (Numerical Stability) """
        A_shifted = A - np.max(A, axis=1, keepdims=True)
        exp_A = np.exp(A_shifted)
        return exp_A / np.sum(exp_A, axis=1, keepdims=True)

    def fit(self, Phi: np.ndarray, T: np.ndarray) -> None:
        """
        Tìm Ma trận trọng số W_MAP bằng Gradient Descent.
        Hàm Mục tiêu E(W) = Cross-Entropy + (alpha/2) * ||W||^2
        """
        N, M = Phi.shape
        K = T.shape[1]

        # Khởi tạo ma trận trọng số W bằng số 0
        self.W_MAP = np.zeros((M, K))

        for i in range(self.max_iter):
            # [TODO 1]: Tính Ma trận Logits A (Kích thước N x K)
            A = Phi.dot(self.W_MAP)

            # Tính Xác suất Y bằng hàm _softmax
            Y = self._softmax(A)

            # Tính Loss để theo dõi hội tụ (Đã viết sẵn cho bạn)
            loss = -np.mean(np.sum(T * np.log(Y + 1e-15), axis=1)) + (self.alpha / 2) * np.sum(self.W_MAP**2)
            self.loss_history.append(loss)

            # [TODO 2]: Tính Gradient của Hàm mục tiêu
            Gradient = (Phi.T.dot(Y - T)) / N + self.alpha * self.W_MAP

            # [TODO 3]: Cập nhật trọng số W_MAP theo thuật toán Gradient Descent
            self.W_MAP = self.W_MAP - self.lr * Gradient


    def laplace_approximation(self, Phi: np.ndarray):
        """
        Xấp xỉ Laplace: Tính Ma trận Hessian tại đỉnh W_MAP, 
        sau đó nghịch đảo để tìm Hiệp phương sai Hậu nghiệm S_N.
        """
        N, M = Phi.shape
        # Lấy K từ số cột của W_MAP
        K = self.W_MAP.shape[1]

        # 1. Tính xác suất Y_MAP tại đỉnh tối ưu
        A_MAP = Phi.dot(self.W_MAP)
        Y_MAP = self._softmax(A_MAP)

        # Khởi tạo Ma trận Hessian toàn cục kích thước (M*K) x (M*K)
        # Vì W là ma trận (M x K), ta phải "duỗi" nó ra thành vector độ dài M*K để dùng Laplace.
        Hessian = np.zeros((M * K, M * K))

        # [TODO 4]: Xây dựng Ma trận Hessian (Thử thách Đại số lớn nhất!)
        # Công thức Block Hessian H_{kj} (kích thước M x M) giữa lớp k và lớp j:
        # H_{kj} = sum_{n=1}^N  y_{nk} * (I_{kj} - y_{nj}) * (phi_n * phi_n^T)
        # Gợi ý vòng lặp:
        for k in range(K):
            for j in range(K):
                # Tính I_kj (Kronecker Delta: 1 nếu k==j, ngược lại 0)
                I_kj = 1.0 if k == j else 0.0

                # Tính ma trận trọng số đường chéo R cho cặp (k, j)
                R_diag = Y_MAP[:, k] * (I_kj - Y_MAP[:, j])

                # Tính Block Hessian H_kj = Phi^T * R * Phi
                H_kj = (Phi.T * R_diag).dot(Phi)

                # Lắp Block H_kj vào đúng vị trí trong Ma trận Hessian toàn cục
                Hessian[k*M : (k+1)*M, j*M : (j+1)*M] = H_kj

        # [TODO 5]: Thêm Đạo hàm bậc 2 của Prior (L2 Penalty) vào Hessian
        # Gợi ý: Cộng thêm self.alpha * Ma trận đơn vị (Identity Matrix) kích thước (MK x MK)
        Hessian += self.alpha * np.eye(M*K)

        # [TODO 6]: Tính Ma trận Hiệp phương sai Hậu nghiệm S_N
        # Gợi ý: S_N chính là nghịch đảo của Hessian. Dùng np.linalg.inv
        self.S_N = np.linalg.inv(Hessian)


    def predict_proba(self, Phi: np.ndarray) -> np.ndarray:
        A = Phi.dot(self.W_MAP)
        return self._softmax(A)