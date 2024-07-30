# this code development by Eng/ Shaif Alameri

import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# System parameters
M = 4  # Number of BS antennas
N = 10 # Number of RIS elements
K = 4  # Number of users
sigma2 = 1e-3  # Noise power
distances = np.linspace(50, 300, 6)  # Distance from BS to RIS in meters
transmit_powers = np.linspace(0.1, 1, 10)  # Transmit power range in Watts

# Generate random channels
def generate_channels(D):
    np.random.seed(42)  # Fixing random state for reproducibility
    path_loss_d = (D / 100) ** (-3.76)  # Path loss for direct link
    path_loss_r = (D / 100) ** (-2.2)  # Path loss for RIS link

    h_d = np.sqrt(path_loss_d) * (np.random.randn(K, M) + 1j * np.random.randn(K, M))
    h_r = np.sqrt(path_loss_r) * (np.random.randn(K, N) + 1j * np.random.randn(K, N))
    H = np.sqrt(path_loss_r) * (np.random.randn(N, M) + 1j * np.random.randn(N, M))

    return h_d, h_r, H

# Objective function to maximize sum-rate with RIS
def sum_rate_with_ris(x, h_d, h_r, H, P_T):
    theta = np.exp(1j * x[:N])  # Phase shifts for RIS
    g = x[N:].reshape((K, M))   # Active beamforming matrix

    sum_rate = 0
    for k in range(K):
        interference = sigma2
        for j in range(K):
            if j != k:
                interference += np.abs(h_d[k] @ g[j] + h_r[k] @ np.diag(theta) @ H @ g[j])**2
        signal = np.abs(h_d[k] @ g[k] + h_r[k] @ np.diag(theta) @ H @ g[k])**2
        sinr = signal / interference
        sum_rate += np.log2(1 + sinr)
    return -sum_rate  # Minimize the negative sum-rate

# Objective function to maximize sum-rate without RIS
def sum_rate_without_ris(x, h_d, P_T):
    g = x.reshape((K, M))   # Active beamforming matrix

    sum_rate = 0
    for k in range(K):
        interference = sigma2
        for j in range(K):
            if j != k:
                interference += np.abs(h_d[k] @ g[j])**2
        signal = np.abs(h_d[k] @ g[k])**2
        sinr = signal / interference
        sum_rate += np.log2(1 + sinr)
    return -sum_rate  # Minimize the negative sum-rate

# Optimizer with RIS
def optimize_beamforming_with_ris(h_d, h_r, H, P_T):
    bounds = [(-np.pi, np.pi)] * N + [(-1, 1)] * (K * M)  # Bounds for the variables
    args = (h_d, h_r, H, P_T)
    result = differential_evolution(sum_rate_with_ris, bounds, args=args, maxiter=100)
    return -result.fun

# Optimizer without RIS
def optimize_beamforming_without_ris(h_d, P_T):
    bounds = [(-1, 1)] * (K * M)  # Bounds for the variables
    args = (h_d, P_T)
    result = differential_evolution(sum_rate_without_ris, bounds, args=args, maxiter=100)
    return -result.fun

# Simulation for varying transmit powers
spectral_efficiency_vs_power_with_ris = []
spectral_efficiency_vs_power_without_ris = []
for P_T in transmit_powers:
    h_d, h_r, H = generate_channels(100)  # Fixed distance for this plot
    max_sum_rate_with_ris = optimize_beamforming_with_ris(h_d, h_r, H, P_T)
    max_sum_rate_without_ris = optimize_beamforming_without_ris(h_d, P_T)
    spectral_efficiency_vs_power_with_ris.append(max_sum_rate_with_ris)
    spectral_efficiency_vs_power_without_ris.append(max_sum_rate_without_ris)

# Simulation for varying distances
spectral_efficiency_vs_distance_with_ris = []
spectral_efficiency_vs_distance_without_ris = []
for D in distances:
    h_d, h_r, H = generate_channels(D)
    max_sum_rate_with_ris = optimize_beamforming_with_ris(h_d, h_r, H, 1)  # Fixed transmit power for this plot
    max_sum_rate_without_ris = optimize_beamforming_without_ris(h_d, 1)
    spectral_efficiency_vs_distance_with_ris.append(max_sum_rate_with_ris)
    spectral_efficiency_vs_distance_without_ris.append(max_sum_rate_without_ris)

# Plot results for transmit power
plt.figure()
plt.plot(transmit_powers, spectral_efficiency_vs_power_with_ris, marker='o', label='With RIS')
plt.plot(transmit_powers, spectral_efficiency_vs_power_without_ris, marker='o', label='Without RIS')
plt.xlabel('Transmit Power (W)')
plt.ylabel('Achievable Spectral Efficiency (bps/Hz)')
plt.title('Achievable Spectral Efficiency vs. Transmit Power')
plt.grid()
plt.legend()
plt.savefig('spectral_efficiency_vs_power.png')
plt.show()

# Plot results for distance
plt.figure()
plt.plot(distances, spectral_efficiency_vs_distance_with_ris, marker='o', label='With RIS')
plt.plot(distances, spectral_efficiency_vs_distance_without_ris, marker='o', label='Without RIS')
plt.xlabel('Distance (m)')
plt.ylabel('Achievable Spectral Efficiency (bps/Hz)')
plt.title('Achievable Spectral Efficiency vs. Distance')
plt.grid()
plt.legend()
plt.savefig('spectral_efficiency_vs_distance.png')
plt.show()