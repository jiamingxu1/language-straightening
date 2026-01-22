import numpy as np
import statsmodels.api as sm
import os 


def make_pink_noise(nT, beta=1):
    # beta: power law exponent
    freqs = np.fft.rfftfreq(nT)
    freqs[0] = 1
    spectrum = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
    spectrum /= freqs ** (beta/2)  # power spectrum follos 1/f^beta, amplitude spectrum follows this
    pink_noise = np.fft.irfft(spectrum, n=nT)
    return pink_noise / np.std(pink_noise)


def make_signal_innov(n_voxs, nT, private_signal_std, corr, beta=0.5):
    """
    n_voxs, nT:    number of voxels and time points
    private_signal_std: signal innovation std (should be the total std)
    corr:          fraction of innovation variance that is shared across voxels
    beta:          fraction of shared variance going to shared_signal_inno
                   (rest 1-beta goes to shared_noise)
    Returns
    -------
    private_signal_innov  : (n_voxs, nT)
    shared_signal_innov    : (nT,)
    shared_noise_innov     : (nT,)
    """

    sigma = private_signal_std  # just renaming

    # Variance allocations
    var_private        = (1.0 - corr) * sigma**2
    var_shared_signal  = corr * beta * sigma**2
    var_shared_noise   = corr * (1.0 - beta) * sigma**2

    # Corresponding stds
    std_private        = np.sqrt(var_private)
    std_shared_signal  = np.sqrt(var_shared_signal)
    std_shared_noise   = np.sqrt(var_shared_noise)

    # Draw components
    private_signal_innov      = np.random.randn(n_voxs, nT) * std_private          # voxel-specific
    shared_signal_innov       = np.random.randn(nT) * std_shared_signal            # shared across voxels & runs
    shared_noise_innov        = np.random.randn(nT) * std_shared_noise             # shared across voxels, per run
    # shared_noise_innov        = np.random.randn(n_voxs, nT) * std_shared_noise     # NOT shared across voxels, per run

    return private_signal_innov, shared_signal_innov, shared_noise_innov

# simulate noise-less AR processes
def make_noiseless_arp(ar_coeff, nT, n_voxs, private_signal_innov):
    time_series = np.zeros((n_voxs, nT))
    time_series[:,0] = private_signal_innov[:,0]
    for t in range(1, nT):
        time_series[:, t] = ar_coeff * time_series[:, t - 1] + private_signal_innov[:,t]
        
    return time_series #(n_voxs, nT)


# simulate noisy AR processes 
def make_additive_noisy_arp(ar_coeff, nT, n_voxs, signal_innov, measurement_noise_std):
    white_noise = np.random.randn(n_voxs, nT) * measurement_noise_std

    time_series = np.zeros((n_voxs, nT))
    time_series[:,0] = signal_innov[:,0]

    for t in range(1, nT):
        time_series[:, t] = ar_coeff * time_series[:, t - 1] + signal_innov[:,t]


    noisy_time_series = time_series + white_noise

    return noisy_time_series

def avg_pop_curvature(X):
    # X: (n_voxs, nT)
    V = X[:, 1:] - X[:, :-1] #(n_series, nT-1)
    norms = np.linalg.norm(V, axis=0) #(nT-1,)
    dots = (V[:, 1:] * V[:, :-1]).sum(0)
    coss = dots / (norms[1:] * norms[:-1])
    angles = np.degrees(np.arccos(coss))
    return angles.mean()  # return mean curvature across series

zs = lambda v: (v-v.mean(0))/v.std(0)