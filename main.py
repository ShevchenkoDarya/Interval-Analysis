# interval_analysis_lab1.py
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import itertools


class IntervalMatrix:
    """Класс для работы с интервальными матрицами"""

    def __init__(self, mid: np.ndarray, rad: np.ndarray):
        self.mid = np.array(mid, dtype=float)
        self.rad = np.array(rad, dtype=float)
        self.shape = self.mid.shape

    def get_interval(self, i: int, j: int) -> tuple:
        center = self.mid[i, j]
        radius = self.rad[i, j]
        return (center - radius, center + radius)

    def contains_matrix(self, A: np.ndarray) -> bool:
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                lower, upper = self.get_interval(i, j)
                if not (lower - 1e-10 <= A[i, j] <= upper + 1e-10):
                    return False
        return True


def check_rectangular_singularity_optimized(mid_A: np.ndarray, rad_A: np.ndarray) -> tuple:
    """
    Проверить, содержит ли интервальная матрица вырожденную
    Использует оптимизацию для нахождения минимального сингулярного числа
    """
    n_rows, n_cols = mid_A.shape

    def objective(params):
        # params - это коэффициенты для каждого элемента (от -1 до 1)
        A_test = mid_A + params.reshape(n_rows, n_cols) * rad_A
        # Вычисляем наименьшее сингулярное число
        _, S, _ = np.linalg.svd(A_test, full_matrices=False)
        return S[-1]  # Наименьшее сингулярное число

    # Начальная точка - центры интервалов
    x0 = np.zeros(n_rows * n_cols)

    # Границы: каждый параметр от -1 до 1
    bounds = [(-1, 1)] * (n_rows * n_cols)

    # Минимизируем наименьшее сингулярное число
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    min_sv = result.fun
    best_params = result.x
    best_A = mid_A + best_params.reshape(n_rows, n_cols) * rad_A

    return min_sv < 1e-6, best_A, min_sv


def find_singular_delta_rectangular(mid_A: np.ndarray, rad_pattern: np.ndarray,
                                    tol: float = 1e-8) -> float:
    """
    Найти минимальное delta для прямоугольной матрицы
    """
    delta_min = 0.0
    delta_max = 1.0

    # Находим верхнюю границу
    while True:
        rad_A = rad_pattern * delta_max
        is_singular, _, min_sv = check_rectangular_singularity_optimized(mid_A, rad_A)
        if is_singular:
            break
        delta_max *= 2
        if delta_max > 100:
            raise ValueError(
                f"Не удалось найти сингулярное значение delta. Минимальное SV при delta={delta_max}: {min_sv}")

    # Бинарный поиск
    for _ in range(100):  # 100 итераций для высокой точности
        delta_mid = (delta_min + delta_max) / 2
        rad_A = rad_pattern * delta_mid
        is_singular, _, _ = check_rectangular_singularity_optimized(mid_A, rad_A)

        if is_singular:
            delta_max = delta_mid
        else:
            delta_min = delta_mid

        if delta_max - delta_min < tol:
            break

    return delta_max


def find_singular_matrix_rectangular(mid_A: np.ndarray, delta: float,
                                     rad_pattern: np.ndarray) -> np.ndarray:
    """
    Найти конкретную вырожденную матрицу
    """
    rad_A = rad_pattern * delta
    _, best_A, _ = check_rectangular_singularity_optimized(mid_A, rad_A)
    return best_A


def check_square_singularity_optimized(mid_A: np.ndarray, rad_A: np.ndarray) -> tuple:
    """
    Проверить, содержит ли квадратная интервальная матрица вырожденную
    """
    n = mid_A.shape[0]

    def objective(params):
        A_test = mid_A + params.reshape(n, n) * rad_A
        return abs(np.linalg.det(A_test))

    x0 = np.zeros(n * n)
    bounds = [(-1, 1)] * (n * n)

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    min_det = result.fun
    best_params = result.x
    best_A = mid_A + best_params.reshape(n, n) * rad_A

    return min_det < 1e-6, best_A, min_det


def find_singular_delta_square(mid_A: np.ndarray, rad_pattern: np.ndarray,
                               tol: float = 1e-8) -> float:
    """
    Найти минимальное delta для квадратной матрицы
    """
    delta_min = 0.0
    delta_max = 1.0

    while True:
        rad_A = rad_pattern * delta_max
        is_singular, _, min_det = check_square_singularity_optimized(mid_A, rad_A)
        if is_singular:
            break
        delta_max *= 2
        if delta_max > 100:
            raise ValueError(
                f"Не удалось найти сингулярное значение delta. Минимальный det при delta={delta_max}: {min_det}")

    for _ in range(100):
        delta_mid = (delta_min + delta_max) / 2
        rad_A = rad_pattern * delta_mid
        is_singular, _, _ = check_square_singularity_optimized(mid_A, rad_A)

        if is_singular:
            delta_max = delta_mid
        else:
            delta_min = delta_mid

        if delta_max - delta_min < tol:
            break

    return delta_max


def find_singular_matrix_square(mid_A: np.ndarray, delta: float,
                                rad_pattern: np.ndarray) -> np.ndarray:
    """
    Найти конкретную вырожденную квадратную матрицу
    """
    rad_A = rad_pattern * delta
    _, best_A, _ = check_square_singularity_optimized(mid_A, rad_A)
    return best_A


def main():
    """Основная функция"""
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №1 ПО ИНТЕРВАЛЬНОМУ АНАЛИЗУ")
    print("=" * 60)

    # ===== ЧАСТЬ 1: ПРЯМОУГОЛЬНАЯ МАТРИЦА =====
    print("\n1. ПРЯМОУГОЛЬНАЯ МАТРИЦА A1 (3x2)")
    print("-" * 60)

    mid_A1 = np.array([
        [0.95, 1.00],
        [1.05, 1.00],
        [1.10, 1.00]
    ])

    # Вариант 1: томография
    rad_pattern1_A1 = np.array([
        [1.00, 1.00],
        [1.00, 1.00],
        [1.00, 1.00]
    ])

    # Вариант 2: регрессия
    rad_pattern2_A1 = np.array([
        [1.00, 0.00],
        [1.00, 0.00],
        [1.00, 0.00]
    ])

    print("\nВариант 1 (томография):")
    delta1_A1 = find_singular_delta_rectangular(mid_A1, rad_pattern1_A1)
    print(f"Минимальное delta: {delta1_A1:.6f}")

    singular_A1_1 = find_singular_matrix_rectangular(mid_A1, delta1_A1, rad_pattern1_A1)
    print(f"Вырожденная матрица:\n{singular_A1_1}")

    U, S, Vt = np.linalg.svd(singular_A1_1, full_matrices=False)
    print(f"Сингулярные числа: {S}")
    print(f"Наименьшее сингулярное число: {S[-1]:.2e}")

    interval_A1_1 = IntervalMatrix(mid_A1, rad_pattern1_A1 * delta1_A1)
    print(f"Принадлежит интервальной матрице: {interval_A1_1.contains_matrix(singular_A1_1)}")

    print("\nВариант 2 (регрессия):")
    delta2_A1 = find_singular_delta_rectangular(mid_A1, rad_pattern2_A1)
    print(f"Минимальное delta: {delta2_A1:.6f}")

    singular_A1_2 = find_singular_matrix_rectangular(mid_A1, delta2_A1, rad_pattern2_A1)
    print(f"Вырожденная матрица:\n{singular_A1_2}")

    U, S, Vt = np.linalg.svd(singular_A1_2, full_matrices=False)
    print(f"Сингулярные числа: {S}")
    print(f"Наименьшее сингулярное число: {S[-1]:.2e}")

    interval_A1_2 = IntervalMatrix(mid_A1, rad_pattern2_A1 * delta2_A1)
    print(f"Принадлежит интервальной матрице: {interval_A1_2.contains_matrix(singular_A1_2)}")

    # ===== ЧАСТЬ 2: КВАДРАТНАЯ МАТРИЦА =====
    print("\n\n2. КВАДРАТНАЯ МАТРИЦА A2 (3x3)")
    print("-" * 60)

    mid_A2 = np.array([
        [1.10, 0.90, 1.10],
        [1.40, 1.00, 0.80],
        [0.80, 1.40, 1.20]
    ])

    rad_pattern_A2 = np.ones((3, 3))

    delta_A2 = find_singular_delta_square(mid_A2, rad_pattern_A2)
    print(f"Минимальное delta: {delta_A2:.6f}")

    singular_A2 = find_singular_matrix_square(mid_A2, delta_A2, rad_pattern_A2)
    print(f"Вырожденная матрица:\n{singular_A2}")

    det_A2 = np.linalg.det(singular_A2)
    print(f"Определитель: {det_A2:.2e}")

    interval_A2 = IntervalMatrix(mid_A2, rad_pattern_A2 * delta_A2)
    print(f"Принадлежит интервальной матрице: {interval_A2.contains_matrix(singular_A2)}")

    # ===== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ =====
    print("\n\n3. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("-" * 60)

    results = {
        'rectangular_tomography_delta': delta1_A1,
        'rectangular_regression_delta': delta2_A1,
        'square_delta': delta_A2,
        'rectangular_tomography_matrix': singular_A1_1,
        'rectangular_regression_matrix': singular_A1_2,
        'square_matrix': singular_A2
    }

    np.savez('interval_analysis_results.npz', **results)
    print("Результаты сохранены в файл 'interval_analysis_results.npz'")

    return results


if __name__ == "__main__":
    results = main()