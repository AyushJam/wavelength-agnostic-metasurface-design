from find_BPnk import get_BP_nk


def get_graphene_nk(f):
    """
    Calculates the complex refractive index of graphene for a given frequency.

    Args:
        f: Frequency in micrometers^-1.

    Returns:
        complex: The complex refractive index (n + ik).
    """

    ref_ind = 3 + 1j * (5.446 / (3 * f))
    return ref_ind


def get_BAs_nk(f):
    """
    Approximates the complex refractive index of BAs for a given frequency.

    Args:
        f: Frequency in micrometers^-1.

    Returns:
        complex: The complex refractive index (n + ik).
    """

    n = 3.1
    k = (0.25 + (5 / 8) * (2.5 - 1 / f)) * 2
    return n + k * 1j


# materials.py
refractive_indices = {
    "Silicon": 3.5,
    "Air": 1,
    "MoO2": 1.6 + 0.5j,
    "Graphene": get_graphene_nk,
    "PEC": (-1e8) ** 0.5,
    "TiO2": (2.5263),
    "BAs": get_BAs_nk,
    "BP": get_BP_nk,
    "Sb2S3": 3.2,
    "SiO2": 1.44,
}