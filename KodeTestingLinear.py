import numpy as np
import matplotlib.pyplot as plt

TB_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
NT_train = np.array([10, 20, 30, 40, 50, 60, 70, 80, 100])

TB_test = np.array([2, 4, 6, 8, 10])
NT_test = np.array([15, 35, 55, 75, 95])

def linear_regression(TB, NT):
    mean_TB = np.mean(TB)
    mean_NT = np.mean(NT)
    numerator = np.sum((TB - mean_TB) * (NT - mean_NT))
    denominator = np.sum((TB - mean_TB)**2)
    m = numerator / denominator
    c = mean_NT - m * mean_TB
    return m, c

def predict(TB, m, c):
    return m * TB + c

def rms_error(actual, predicted):
    return np.sqrt(np.mean((actual - predicted)**2))

m, c = linear_regression(TB_train, NT_train)

NT_train_pred = predict(TB_train, m, c)

RMS_train = rms_error(NT_train, NT_train_pred)

NT_test_pred = predict(TB_test, m, c)

RMS_test = rms_error(NT_test, NT_test_pred)

print(f"Koefisien regresi (m): {m}")
print(f"Intersep (c): {c}")
print(f"Galat RMS untuk data pelatihan: {RMS_train}")
print(f"Galat RMS untuk data pengujian: {RMS_test}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(TB_train, NT_train, color='blue', label='Data Pelatihan Asli')
plt.plot(TB_train, NT_train_pred, color='red', label='Garis Regresi Linear')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Linear - Data Pelatihan')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(TB_test, NT_test, color='green', label='Data Pengujian Asli')
plt.plot(TB_test, NT_test_pred, color='orange', label='Garis Regresi Linear')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Linear - Data Pengujian')
plt.legend()

plt.show()
