def test_regression_model():
    import numpy as np
    import matplotlib.pyplot as plt

    TB = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)

    NT = np.array([10, 20, 30, 40, 50, 60, 70, 80, 100], dtype=float)

    log_TB = np.log(TB)
    log_NT = np.log(NT)

    A = np.vstack([log_TB, np.ones(len(log_TB))]).T
    m, c = np.linalg.lstsq(A, log_NT, rcond=None)[0]

    print(f"Model regresi pangkat: NT = e^({c:.4f}) * TB^{m:.4f}")

    NT_pred = np.exp(c) * TB**m

    error = NT - NT_pred

    RMS_error = np.sqrt(np.mean(error**2))

    print(f"Galat RMS: {RMS_error:.4f}")

    plt.scatter(TB, NT, color='blue', label='Data Sebenarnya')
    plt.plot(TB, NT_pred, color='red', label='Model Prediksi')
    plt.xlabel('Durasi Waktu Belajar (TB)')
    plt.ylabel('Nilai Ujian (NT)')
    plt.title('Regresi Model Pangkat Sederhana')
    plt.legend()
    plt.show()

    expected_RMS_error = 2.6961
    assert np.isclose(RMS_error, expected_RMS_error, atol=0.01), f"Expected RMS error to be close to {expected_RMS_error}, but got {RMS_error}"

    expected_m = 1.097  
    expected_c = 2.233  
    assert np.isclose(m, expected_m, atol=0.01), f"Expected slope (m) to be close to {expected_m}, but got {m}"
    assert np.isclose(c, expected_c, atol=0.01), f"Expected intercept (c) to be close to {expected_c}, but got {c}"

    print("All tests passed.")

test_regression_model()
