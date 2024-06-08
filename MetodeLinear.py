import numpy as np
import matplotlib.pyplot as plt

TB = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
NT = np.array([10, 20, 30, 40, 50, 60, 70, 80, 100])

mean_TB = np.mean(TB)
mean_NT = np.mean(NT)

numerator = np.sum((TB - mean_TB) * (NT - mean_NT))
denominator = np.sum((TB - mean_TB)**2)
m = numerator / denominator
c = mean_NT - m * mean_TB

NT_pred = m * TB + c

RMS = np.sqrt(np.mean((NT - NT_pred)**2))

print(f"Koefisien regresi (m): {m}")
print(f"Intersep (c): {c}")
print(f"Galat RMS: {RMS}")

plt.scatter(TB, NT, color='blue', label='Data Asli')
plt.plot(TB, NT_pred, color='red', label='Garis Regresi Linear')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Linear Sederhana')
plt.legend()
plt.show()
