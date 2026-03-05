from astropy.table import Table
import numpy as np
from tqdm import tqdm
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from pdspl_utils.inference import beta_double_source_plane, beta2theta_e_ratio

############################################################################
# GENERIC PAIRING ALGORITHMS
############################################################################

def normalize_data(data, type='minmax', data_min=None, data_max=None):
    """Normalize the data to the range [0, 1] or by Z-score."""
    data = np.array(data, dtype=float)
    if type == 'minmax':
        data_min = np.nanmin(data) if data_min is None else data_min
        data_max = np.nanmax(data) if data_max is None else data_max
        if data_max == data_min: return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    elif type == 'zscore':
        std = np.nanstd(data)
        if std == 0: return np.zeros_like(data)
        return (data - np.nanmean(data)) / std
    else:
        raise ValueError("Unsupported normalization type. Use 'minmax' or 'zscore'.")

def get_kdtree_pairs(table, pairing_keys, norm_type='zscore', n_neighbors=2, unique_pairs=True):
    """
    General function to find near-identical pairs in ANY table based on specified keys.
    """
    # 1. Normalize and stack the features into a multi-dimensional space
    points = np.stack([normalize_data(table[pk], type=norm_type) for pk in pairing_keys], axis=1)
    
    # 2. Build and query KDTree
    tree = spatial.KDTree(points)
    distances, indices = tree.query(points, k=n_neighbors)
    distances = distances[:, 1] # exclude self-distance
    
    # 3. Filter unique pairs
    if unique_pairs:
        indices = np.sort(indices, axis=1) # Ensure (i, j) is same as (j, i)
        indices, unique_pairs_idxs = np.unique(indices, axis=0, return_index=True)
        distances = distances[unique_pairs_idxs]

    return indices, distances

def build_generic_pairs_table(table, pair_indices, sort_by=None):
    """
    Takes an input table and pair indices, returning a new table where each row is a pair.
    Every column 'X' in the original table becomes 'X_1' and 'X_2'.
    
    :param sort_by: Optional column name. If provided, ensures entity '1' has the lower value.
    """
    idx1 = pair_indices[:, 0].copy()
    idx2 = pair_indices[:, 1].copy()

    # Optional: Enforce ordering (e.g., ensuring z_S1 < z_S2)
    if sort_by and sort_by in table.colnames:
        swap_mask = table[idx1][sort_by] > table[idx2][sort_by]
        idx1[swap_mask], idx2[swap_mask] = idx2[swap_mask], idx1[swap_mask]

    pairs_dict = {"index_1": idx1, "index_2": idx2}
    
    for col in table.colnames:
        pairs_dict[f"{col}_1"] = table[col][idx1]
        pairs_dict[f"{col}_2"] = table[col][idx2]
        
    return Table(pairs_dict)

############################################################################
# DOMAIN-SPECIFIC (PDSPL) EXTENSIONS
############################################################################

def calculate_rel_diff(val1, val2):
    """Helper to calculate fractional difference."""
    return 2 * (val2 - val1) / (val2 + val1)

def get_pairs_table_PDSPL(data_table, pair_indices, cosmo):
    """
    Builds the PDSPL-specific table on top of the generic pairs table.
    """
    # 1. Build the generic table, enforcing z_S1 < z_S2
    pt = build_generic_pairs_table(data_table, pair_indices, sort_by="z_S")
    
    # 2. Add convenient combined properties
    pt["z_D"] = 0.5 * (pt["z_D_1"] + pt["z_D_2"])
    
    # Rename some columns back to your original naming convention if necessary
    pt.rename_column('z_S_1', 'z_S1')
    pt.rename_column('z_S_2', 'z_S2')
    pt.rename_column('theta_E_1', 'theta_E1')
    pt.rename_column('theta_E_2', 'theta_E2')
    
    # 3. Calculate DSPL beta properties
    _beta_E_DSPL_D1 = beta_double_source_plane(pt["z_D_1"], pt["z_S1"], pt["z_S2"], cosmo)
    _beta_E_DSPL_D2 = beta_double_source_plane(pt["z_D_2"], pt["z_S1"], pt["z_S2"], cosmo)
    
    beta_E_DSPL_D1 = beta2theta_e_ratio(_beta_E_DSPL_D1, gamma_pl=pt["gamma_pl_1"], lambda_mst=1)
    beta_E_DSPL_D2 = beta2theta_e_ratio(_beta_E_DSPL_D2, gamma_pl=pt["gamma_pl_2"], lambda_mst=1)
    
    pt["beta_E_DSPL"] = 0.5 * (beta_E_DSPL_D1 + beta_E_DSPL_D2)
    pt["beta_E_pseudo"] = pt["theta_E1"] / pt["theta_E2"]
    
    # 4. Add calculated relative differences
    pt["rel_diff_beta_E"] = 1 - pt['beta_E_pseudo'] / pt['beta_E_DSPL']
    
    diff_keys = ['sigma_v_D', 'R_e_kpc', 'R_e_arcsec', 'Sigma_half_Msun/pc2', 
                 'mag_D_i', 'z_D', 'gamma_pl', 'color_D_gr', 'color_D_ri']
    
    for key in diff_keys:
        if f"{key}_1" in pt.colnames and f"{key}_2" in pt.colnames:
            pt[f"rel_diff_{key}"] = calculate_rel_diff(pt[f"{key}_1"], pt[f"{key}_2"])
            
    return pt

def inject_observational_errors(table, sample_key):
    """Generates a new table with added Gaussian noise based on sample assumptions."""
    noisy_table = table.copy()
    
    # z_D errors
    if sample_key in ['lsst_y1', 'lsst_y10']:
        err_z_D = 0.03 * (1 + table['z_D']) # Photo-z
    elif sample_key in ['lsst_4most_spec-z', 'lsst_4most_spec-z_sigma_v']:
        err_z_D = np.full(len(table), 1e-4) # Spec-z
    else:
        err_z_D = np.zeros(len(table))
        
    noisy_table['z_D'] += np.random.normal(0, err_z_D)
    noisy_table['err_z_D'] = err_z_D

    # sigma_v_D errors
    err_sigma = np.full(len(table), 10.0)
        
    noisy_table['sigma_v_D'] += np.random.normal(0, err_sigma)
    noisy_table['err_sigma_v_D'] = err_sigma

    # R_e errors
    err_R_e = 0.05 * table['R_e_arcsec']
    noisy_table['R_e_arcsec'] += np.random.normal(0, err_R_e)
    noisy_table['err_R_e_arcsec'] = err_R_e

    # Magnitude errors
    err_mag = 1e-3 # millimag precision
    noisy_table['mag_D_i'] += np.random.normal(0, err_mag)
    noisy_table['err_mag_D_i'] = err_mag
    
    return noisy_table

def compute_dissimilarity(pairs_table, dissimilarity_keys, method='rms'):
    """
    Computes the dissimilarity metric between paired lenses.

    Parameters
    ----------
    pairs_table : astropy.table.Table
        The table containing the paired lenses and their properties.
    dissimilarity_keys : list of str
        The keys (e.g., 'rel_diff_z_D') used to calculate the metric.
    method : str
        'rms'  -> Root-Mean-Square of relative fractional differences.
        'chi2' -> Error-Weighted distance (Mahalanobis/Chi-Square approach).
    """
    if method == 'rms':
        diffs = np.array([pairs_table[k] for k in dissimilarity_keys])
        return np.sqrt(np.mean(diffs**2, axis=0))

    elif method == 'chi2':
        chi2_terms = []
        for key in dissimilarity_keys:
            # Extract the base physical parameter (e.g., 'z_D' from 'rel_diff_z_D')
            base_feature = key.replace('rel_diff_', '')
            
            err_col1 = f"err_{base_feature}_1"
            err_col2 = f"err_{base_feature}_2"
            
            # Use explicit observational errors if they exist in the mock table
            if err_col1 in pairs_table.colnames and err_col2 in pairs_table.colnames:
                val1 = pairs_table[f"{base_feature}_1"]
                val2 = pairs_table[f"{base_feature}_2"]
                
                variance = pairs_table[err_col1]**2 + pairs_table[err_col2]**2
                # Prevent division by zero if error is exactly 0
                variance = np.where(variance == 0, 1e-12, variance)
                
                term = ((val1 - val2)**2) / variance
            else:
                # Fallback: Treat the relative difference against an assumed 10% error.
                # This ensures features without explicit err_ columns (like colors) 
                # don't get mathematically ignored compared to the chi2 terms.
                assumed_fractional_error = 0.10 
                term = (pairs_table[key] / assumed_fractional_error)**2
                
            chi2_terms.append(term)
            
        return np.sqrt(np.mean(chi2_terms, axis=0))

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'rms' or 'chi2'.")

def plot_beta_E_vs_D_MC(pdspl_samples, mc_results, fit_type='linear', save_path=None, show_fit_eqn_label=True):
    """
    Plots the scatter of beta_E vs Dissimilarity along with marginal distributions 
    and MC realization statistics.
    
    :param fit_type: 'linear' or 'power_law'
    """
    alpha_vals = {
        "lsst_y10": 0.007, "lsst_y1": 0.01, 
        "lsst_4most_spec-z": 0.03, "lsst_4most_spec-z_sigma_v": 0.04
    }
    markers = {
        "lsst_y10": "o", "lsst_y1": "s", 
        "lsst_4most_spec-z": "s", "lsst_4most_spec-z_sigma_v": "D"
    }

    fig = plt.figure(figsize=(13, 6))
    gs = GridSpec(4, 9, figure=fig, wspace=-0.02, hspace=-0.02)

    ax_scatter = fig.add_subplot(gs[1:, 0:3])
    ax_histx   = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
    ax_histy   = fig.add_subplot(gs[1:, 3], sharey=ax_scatter)
    ax_fit     = fig.add_subplot(gs[1:, 5:9])

    # -------------------------------------------------------
    # Thicken Axis Frames and Ticks
    # -------------------------------------------------------
    frame_width = 2.0
    for ax in [ax_scatter, ax_fit]:
        for spine in ax.spines.values():
            spine.set_linewidth(frame_width)
        # Match tick mark thickness to the frame
        ax.tick_params(axis='both', which='major', width=frame_width, length=6)
        ax.tick_params(axis='both', which='minor', width=frame_width, length=4)

    for sample_key, s in pdspl_samples.items():
        if "pairs_analysis" not in s:
            continue
            
        color = s.get('color', 'black')
        marker = markers.get(sample_key, 'o')
        alpha = alpha_vals.get(sample_key, 0.01)

        # -------------------------------------------------------
        # LEFT PANEL: Single Realization Scatter & Distributions
        # -------------------------------------------------------
        tbl_err = s["pairs_analysis"]["pairs_table_with_errors"]
        tbl = s["pairs_analysis"]["pairs_table"]

        dissim = tbl_err['dissimilarity']
        delta_beta_E = 1 - tbl['beta_E_pseudo'] / tbl['beta_E_DSPL']

        # Apply mask
        mask = (dissim < 0.1) & np.isfinite(delta_beta_E)
        dissim_clean = dissim[mask]
        delta_beta_clean = delta_beta_E[mask]

        # 1. Scatter Plot
        ax_scatter.scatter(dissim_clean, delta_beta_clean, color=color, alpha=alpha, s=10)

        # 2. KDE Histograms
        try:
            x_grid = np.linspace(0, 0.1, 200)
            y_grid = np.linspace(-0.4, 0.4, 200)
            ax_histx.plot(x_grid, gaussian_kde(dissim_clean)(x_grid), color=color, lw=1.5)
            ax_histy.plot(gaussian_kde(delta_beta_clean)(y_grid), y_grid, color=color, lw=1.5)
        except np.linalg.LinAlgError:
            pass # Handle edge cases where KDE fails

        # 3. Binned Error Bars (Last Realization)
        percentiles = np.percentile(dissim_clean, np.arange(0, 101, 10))
        digitized = np.digitize(dissim_clean, percentiles)

        binned_x, binned_y, binned_yerr = [], [], []
        for i in range(1, len(percentiles)):
            bin_mask = (digitized == i)
            if np.any(bin_mask):
                binned_x.append(np.median(dissim_clean[bin_mask]))
                binned_y.append(np.nanmean(delta_beta_clean[bin_mask]))
                binned_yerr.append(np.nanstd(delta_beta_clean[bin_mask]))

        ax_scatter.errorbar(binned_x, binned_y, yerr=binned_yerr, fmt=marker, color=color,
                            markersize=8, label=s['name'], capsize=5, elinewidth=3)

        # -------------------------------------------------------
        # RIGHT PANEL: MC Statistics & Fitting
        # -------------------------------------------------------
        all_dissim = np.array(mc_results[sample_key]['binned_dissim'])
        all_scatter = np.array(mc_results[sample_key]['binned_scatter'])

        mean_dissim = np.nanmean(all_dissim, axis=0)
        mean_scatter = np.nanmean(all_scatter, axis=0)
        std_scatter = np.nanstd(all_scatter, axis=0)

        valid = np.isfinite(mean_dissim) & np.isfinite(mean_scatter) & (mean_scatter > 0)
        x_data, y_data, y_err = mean_dissim[valid], mean_scatter[valid], std_scatter[valid]

        ax_fit.scatter(x_data, y_data, marker=marker, color=color, alpha=0.8)

        # Perform chosen fit
        if len(x_data) > 2:
            if fit_type == 'linear':
                weights = np.where((y_err > 0), 1.0 / y_err, 0.0)
                coeffs, cov = np.polyfit(x_data, y_data, 1, w=weights, cov=True)
                y_fit = np.polyval(coeffs, x_data)
                eq_latex = r"$y = (%.2f \pm %.2f) x + (%.3f \pm %.3f)$" % (
                    coeffs[0], np.sqrt(cov[0, 0]), coeffs[1], np.sqrt(cov[1, 1]))
                
            elif fit_type == 'power_law':
                log_x, log_y = np.log10(x_data), np.log10(y_data)
                sigma_log_y = y_err / (y_data * np.log(10))
                weights = np.where((sigma_log_y > 0), 1.0 / sigma_log_y, 0.0)
                coeffs, cov = np.polyfit(log_x, log_y, 1, w=weights, cov=True)
                y_fit = 10**(np.polyval(coeffs, log_x))
                eq_latex = r"$y = 10^{%.2f \pm %.2f} x^{%.2f \pm %.2f}$" % (
                    coeffs[1], np.sqrt(cov[1, 1]), coeffs[0], np.sqrt(cov[0, 0]))

            # Plot Fit Line and Fill
            ax_fit.plot(x_data, y_fit, linestyle='--', color=color, label=eq_latex)
            
            # Save stats back to dictionary
            s['pairs_analysis']['scatter_vs_dissimilarity_fit_coeffs'] = coeffs
            s['pairs_analysis']['scatter_vs_dissimilarity_fit_cov'] = cov

        ax_fit.fill_between(x_data, y_data - y_err, y_data + y_err, color=color, alpha=0.2)

    # -------------------------------------------------------
    # Formatting and Cleanup
    # -------------------------------------------------------
    # Left Panel Labels and Ticks
    ax_scatter.set_xlabel(r"$\mathcal{D}_{\rm deflector}$", fontsize=18)
    ax_scatter.set_ylabel(r"$\Delta \beta_{E} / \beta_{E} = 1 - \beta_{\rm E, pseudo}/\beta_{\rm E, DSPL}$", fontsize=18)
    ax_scatter.tick_params(axis='both', which='major', labelsize=14)
    ax_scatter.set_xlim(0, 0.09)
    ax_scatter.set_ylim(-0.43, 0.43)
    ax_scatter.legend(frameon=True, fontsize=12)

    ax_histx.axis('off')
    ax_histy.axis('off')

    # Right Panel Labels and Ticks
    ax_fit.set_xlabel(r"$\mathcal{D}_{\rm deflector}$", fontsize=18)
    ax_fit.set_ylabel(r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}} = \sigma(\Delta \beta_{E} / \beta_{E})$", fontsize=18)
    ax_fit.tick_params(axis='both', which='major', labelsize=14)
    ax_fit.set_xlim(0, 0.06)
    ax_fit.set_ylim(0, 0.3)
    if show_fit_eqn_label:
        ax_fit.legend(frameon=True, fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig