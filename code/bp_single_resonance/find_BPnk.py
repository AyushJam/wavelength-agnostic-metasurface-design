import numpy as np
import matplotlib.pyplot as plt
import csv


def load_k_data():
    """Loads k data from the CSV file.

    Returns:
        tuple: A tuple containing two NumPy arrays, the first for k values
               and the second for corresponding wavelengths.
    """

    with open("BP_k_data.csv", "r") as f:
        data = list(csv.reader(f, delimiter=","))
        data = np.array(data[1:]).astype(float)

    k_data = data[:, 0]
    wav_data = data[:, 1]

    return k_data, wav_data


def get_BP_nk(freq):
    """Retrieves the complex refractive index (n-ik) for a given frequency.

    Args:
        freq: Frequency in micrometers^-1.

    Returns:
        complex: The complex refractive index (n + ik).
    """

    k_data, wav_data = load_k_data()
    wav_ind = np.argmin(np.abs(wav_data - 1 / freq))
    k = k_data[wav_ind]

    return BP_n + 1j * k


BP_n = 3.5


if __name__ == "__main__":
    # Load k data
    k_data, wav_data = load_k_data()

    # Plot n and k data
    plt.plot(wav_data, k_data, label='k')
    plt.plot(wav_data, np.ones(len(wav_data)) * BP_n, label='n')
    plt.grid(True)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Refractive Index")  # Combined label for n and k
    plt.title("BP n-k Data for 7 Layers")
    plt.legend()
    plt.show()
    plt.savefig("BP_nk.png")
