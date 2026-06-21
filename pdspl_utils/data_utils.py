# pdspl_utils/data_utils.py

"""
Shared data loading and preprocessing utilities for the PDSPL analysis pipeline.
Import this module instead of re-defining preprocess_ggl_table in every notebook.
"""

import numpy as np
from astropy.table import Table


def preprocess_ggl_table(filepath):
    """
    Load a GGL catalog FITS file and apply all standard column transformations
    needed for downstream pairing and inference.

    Transformations applied
    -----------------------
    * Photometric color columns (color_D_gr, color_D_ri)
    * Linear pseudo-flux columns (flux_D_i) and flux-ratio columns
      (flux_ratio_D_gr, flux_ratio_D_ri) — used as pairing features in log-space
    * Log columns for convenience (log_R_e_kpc, log_Sigma_half, log_sigma_v_D)
    * Rename 'size_D' → 'R_e_arcsec'  (SLSim stores angular half-light radius as size_D)

    Parameters
    ----------
    filepath : str
        Path to the FITS catalog produced by notebook 02.

    Returns
    -------
    astropy.table.Table
        Pre-processed lens catalog.
    """
    table = Table.read(filepath, format="fits")

    # --- color columns (magnitude differences) ---
    table["color_D_gr"] = table["mag_D_g"] - table["mag_D_r"]
    table["color_D_ri"] = table["mag_D_r"] - table["mag_D_i"]

    # --- linear pseudo-flux columns (no zero-point correction; for pairing only) ---
    # Converting to flux so that log-space Euclidean distance ≈ fractional difference,
    # consistent with the kD-Tree log-metric and the RMS dissimilarity metric.
    table["flux_D_i"]        = 10 ** (-0.4 * table["mag_D_i"])
    table["flux_ratio_D_gr"] = 10 ** (-0.4 * table["mag_D_g"]) / 10 ** (-0.4 * table["mag_D_r"])
    table["flux_ratio_D_ri"] = 10 ** (-0.4 * table["mag_D_r"]) / 10 ** (-0.4 * table["mag_D_i"])

    # --- log columns (useful for corner plots and inspection) ---
    table["log_R_e_kpc"]              = np.log10(table["R_e_kpc"])
    table["log_Sigma_half_Msun/pc2"]  = np.log10(table["Sigma_half_Msun/pc2"])
    table["log_sigma_v_D"]            = np.log10(table["sigma_v_D"])

    # --- rename angular effective radius column ---
    # SLSim writes it as 'size_D'; we call it 'R_e_arcsec' everywhere else.
    if "size_D" in table.colnames and "R_e_arcsec" not in table.colnames:
        table.rename_column("size_D", "R_e_arcsec")

    return table