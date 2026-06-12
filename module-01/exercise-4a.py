import numpy as np

# Ten "measurements" — pretend these are repeated readings of one quantity.
x = np.array([12.1, 9.8, 11.5, 10.2, 13.0, 8.7, 10.9, 12.4, 9.1, 11.8])
n = x.size

mean = x.sum() / n

deviations = x - mean
sq_deviations = deviations**2
variance = sq_deviations.sum() / n
corrected_variance = sq_deviations.sum() / (n - 1)  # Bessel's correction

print("Mean:                ", mean)
print("Variance:            ", variance)
print("Standard deviation:  ", np.sqrt(variance))
print("Corrected variance:  ", corrected_variance)
print("Standard dev. corr.: ", np.sqrt(corrected_variance))

print("n / (n - 1):                   ", n / (n - 1))
print("Corrected variance / Variance: ", corrected_variance / variance)
