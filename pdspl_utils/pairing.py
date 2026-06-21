# pdspl_utils/pairing.py

from astropy.table import Table
import numpy as np
from tqdm import tqdm
import corner
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.optimize import curve_fit

from pdspl_utils.inference import beta_double_source_plane, beta2theta_e_ratio

############################################################################
# PAIRING SIMULATION
############################################################################

def normalize_data(data, type='minmax', data_min=None, data_max=None):
    """Normalize the data to the range [0, 1] or by Z-score.
    Parameters
    ----------
    data : array-like
        The input data to be normalized.
    type : str, optional
        The type of normalization to be applied. Default is 'minmax'.
        Options are:
        - 'minmax': Min-Max normalization to the range [0, 1].
        - 'zscore': Z-score normalization (mean=0, std=1).
    data_min : float, optional
        The minimum value of the data for 'minmax' normalization. If None, it is
        computed from the data.
    data_max : float, optional
        The maximum value of the data for 'minmax' normalization. If None, it is
        computed from the data.
    Returns
    -------
    normalized_data : array-like
        The normalized data.
    """
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

# def get_kdtree_pairs(table, pairing_keys, norm_type='zscore', n_neighbors=2, unique_pairs=True):
#     """
#     General function to find near-identical pairs in ANY table based on specified keys.

#         :param table: Astropy Table containing the data.
#         :param pairing_keys: List of column names to use for pairing.
#         :param norm_type: Type of normalization ('minmax' or 'zscore').
#         :param n_neighbors: Number of nearest neighbors to find (including self). Set to 2 for just the closest pair.
#         :param unique_pairs: If True, ensures that pairs are unique (i.e., (i, j) is the same as (j, i) and only one is kept).
#     """
#     # 1. Normalize and stack the features into a multi-dimensional space
#     points = np.stack([normalize_data(table[pk], type=norm_type) for pk in pairing_keys], axis=1)
    
#     # 2. Build and query KDTree
#     tree = spatial.KDTree(points)
#     distances, indices = tree.query(points, k=n_neighbors)
#     distances = distances[:, 1] # exclude self-distance
    
#     # 3. Filter unique pairs
#     if unique_pairs:
#         indices = np.sort(indices, axis=1) # Ensure (i, j) is same as (j, i)
#         indices, unique_pairs_idxs = np.unique(indices, axis=0, return_index=True)
#         distances = distances[unique_pairs_idxs]

#     return indices, distances

def get_kdtree_pairs(table, pairing_keys, norm_type='zscore', n_neighbors=2, 
                     unique_pairs=True, use_log_metric=False):
    """
    Finds near-identical pairs in ANY table based on specified keys.

    :param use_log_metric: If True, log-transforms each pairing feature before building
                           the KD-Tree. Euclidean distance in log-space approximates the
                           RMS relative difference (dissimilarity metric), so KD-Tree
                           nearest-neighbor rankings are identical to exact dissimilarity
                           rankings. Validated at 99.8% pair overlap vs exact method.
                           All values in pairing_keys must be strictly positive.
                           When True, norm_type is ignored.
    """
    if use_log_metric:
        points = np.stack(
            [np.log(np.array(table[pk], dtype=float)) for pk in pairing_keys], axis=1
        )
        if not np.all(np.isfinite(points)):
            raise ValueError(
                "use_log_metric=True requires all pairing features to be strictly "
                "positive. Check for zeros or negatives in pairing_keys."
            )
    else:
        points = np.stack(
            [normalize_data(table[pk], type=norm_type) for pk in pairing_keys], axis=1
        )

    # 2. Build and query KDTree
    tree = spatial.KDTree(points)
    distances, indices = tree.query(points, k=n_neighbors)

    # Handle edge case where n_neighbors is just 1 (only finds itself)
    if n_neighbors < 2:
        return np.empty((0, 2), dtype=int), np.empty(0)

     # 3. Flatten the arrays to create actual pair combinations
    n_points         = len(points)

    # Repeat the base indices for however many neighbors we are extracting (excluding self)
    base_indices     = np.repeat(np.arange(n_points), n_neighbors - 1)

    # Flatten the neighbor indices and distances (skipping column 0, which is self-distance)
    neighbor_indices = indices[:, 1:].flatten()
    flat_distances   = distances[:, 1:].flatten()

    # Create the Nx2 array of pairs
    pair_indices     = np.column_stack((base_indices, neighbor_indices))

    # 4. Filter unique pairs
    if unique_pairs:
        # Sort each row so that (i, j) becomes (min(i,j), max(i,j))
        pair_indices = np.sort(pair_indices, axis=1)
        # Extract the unique rows
        pair_indices, unique_pairs_idxs = np.unique(pair_indices, axis=0, return_index=True)
        flat_distances = flat_distances[unique_pairs_idxs]

    return pair_indices, flat_distances

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
# PDSPL EXTENSIONS
############################################################################

def calculate_rel_diff(val1, val2):
    """Helper to calculate fractional difference."""
    return 2 * (val2 - val1) / (val2 + val1)

def calculate_absolute_diff(val1, val2):
    """Helper to calculate absolute difference."""
    return np.abs(val2 - val1)

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
                 'mag_D_i', 'flux_D_i', 'z_D', 'gamma_pl', 'color_D_gr', 'color_D_ri',
                 'flux_ratio_D_gr', 'flux_ratio_D_ri']
    
    for key in diff_keys:
        if f"{key}_1" in pt.colnames and f"{key}_2" in pt.colnames:
            pt[f"rel_diff_{key}"] = calculate_rel_diff(pt[f"{key}_1"], pt[f"{key}_2"])
            pt[f"abs_diff_{key}"] = calculate_absolute_diff(pt[f"{key}_1"], pt[f"{key}_2"])
            
    return pt

def inject_observational_errors(table, error_config):
    """
    Generates a new table with added Gaussian noise based on a provided configuration.
    
    Parameters
    ----------
    table : astropy.table.Table
        The input data table.
    error_config : dict
        A dictionary mapping column names to error specifications.
        Values can be:
        - A callable (e.g., lambda function) taking the table as input.
        - A float/int (constant error for all rows).
        - An array-like of the same length as the table.
    """
    noisy_table = table.copy()
    
    for col, err_spec in error_config.items():
        if col not in table.colnames:
            continue
            
        # Determine the error array based on the spec type
        if callable(err_spec):
            err_array = err_spec(table)
        elif isinstance(err_spec, (int, float)):
            err_array = np.full(len(table), float(err_spec))
        else:
            err_array = np.array(err_spec)
            
        # Inject Gaussian noise and save the error column
        noisy_table[col] += np.random.normal(0, err_array)
        noisy_table[f'err_{col}'] = err_array
        
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

def plot_beta_E_vs_D_MC(pdspl_samples, mc_results, fit_type='linear', save_path=None, show_fit_eqn_label=True,
                        custom_colors_dict=None, custom_markers_dict=None, plot_dissimilarity_range=(0, 0.1)):
    """
    Plots the scatter of beta_E vs Dissimilarity along with marginal distributions 
    and MC realization statistics.
    
    :param fit_type: 'linear' or 'power_law' or 'quadrature' for the fitting method on the right panel.
    """
    alpha_vals = {
        "lsst_y10": 0.007, "lsst_y1": 0.01, 
        "lsst_4most_spec-z": 0.03, "lsst_4most_spec-z_sigma_v": 0.04
    }
    markers = {
        "lsst_y10": "o", "lsst_y1": "*", 
        "lsst_4most_spec-z": "s", "lsst_4most_spec-z_sigma_v": "D",
        "lsst_y10_photo_z": "o", "lsst_y10_spec_z": "s"
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
            
        color = custom_colors_dict.get(sample_key, s.get('color', 'black')) if custom_colors_dict else s.get('color', 'black')
        marker = custom_markers_dict.get(sample_key, markers.get(sample_key, 'o')) if custom_markers_dict else markers.get(sample_key, 'o')
        alpha = alpha_vals.get(sample_key, 0.01)

        # -------------------------------------------------------
        # LEFT PANEL: Single Realization Scatter & Distributions
        # -------------------------------------------------------
        tbl_err = s["pairs_analysis"]["pairs_table_with_errors"]
        tbl = s["pairs_analysis"]["pairs_table"]

        dissim = tbl_err['dissimilarity']
        delta_beta_E = 1 - tbl['beta_E_pseudo'] / tbl['beta_E_DSPL']

        # Apply mask
        mask = (dissim < plot_dissimilarity_range[1]) & np.isfinite(delta_beta_E)
        dissim_clean = dissim[mask]
        delta_beta_clean = delta_beta_E[mask]

        # 1. Scatter Plot
        ax_scatter.scatter(dissim_clean, delta_beta_clean, color=color, alpha=alpha, s=10)

        # 2. KDE Histograms
        try:
            x_grid = np.linspace(0, plot_dissimilarity_range[1], 200)
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
            
            elif fit_type == 'quadrature':
                # Model: y = sqrt( c0^2 + (c1 * x)^2 )
                def quad_model(x, c0, c1):
                    return np.sqrt(c0**2 + (c1 * x)**2)
                
                # Fit the curve
                popt, pcov = curve_fit(quad_model, x_data, y_data, sigma=y_err, absolute_sigma=True, p0=[0.1, 1.0])
                c0, c1 = popt
                err_c0, err_c1 = np.sqrt(np.diag(pcov))
                
                y_fit = quad_model(x_data, c0, c1)
                eq_latex = r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}} = \sqrt{(%.3f \pm %.3f)^2 + [(%.2f \pm %.2f)\mathcal{D}_{\rm deflector}]^2}$" % (
                    c0, err_c0, c1, err_c1)
                
                # Save coefficients for the MCMC likelihood
                coeffs = [c1, c0] # Note: storing as [slope_equivalent, intercept_equivalent]
                cov = pcov

            # Plot Fit Line and Fill
            ax_fit.plot(x_data, y_fit, linestyle='--', color=color, label=eq_latex)
            
            # Save stats back to dictionary
            s['pairs_analysis']['scatter_vs_dissimilarity_fit_coeffs'] = coeffs
            s['pairs_analysis']['scatter_vs_dissimilarity_fit_cov'] = cov
            s['pairs_analysis']['scatter_vs_dissimilarity_fit_type'] = fit_type

        ax_fit.fill_between(x_data, y_data - y_err, y_data + y_err, color=color, alpha=0.2)

    # -------------------------------------------------------
    # Formatting and Cleanup
    # -------------------------------------------------------
    # Left Panel Labels and Ticks
    ax_scatter.set_xlabel(r"$\mathcal{D}_{\rm deflector}$", fontsize=18)
    ax_scatter.set_ylabel(r"$\Delta \beta_{E} / \beta_{E} = 1 - \beta_{\rm E, pseudo}/\beta_{\rm E, DSPL}$", fontsize=18)
    ax_scatter.tick_params(axis='both', which='major', labelsize=14)
    ax_scatter.set_xlim(plot_dissimilarity_range)
    ax_scatter.set_ylim(-0.43, 0.43)
    ax_scatter.legend(frameon=True, fontsize=12)

    ax_histx.axis('off')
    ax_histy.axis('off')

    # Right Panel Labels and Ticks
    ax_fit.set_xlabel(r"$\mathcal{D}_{\rm deflector}$", fontsize=18)
    ax_fit.set_ylabel(r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}} = \sigma(\Delta \beta_{E} / \beta_{E})$", fontsize=18)
    ax_fit.tick_params(axis='both', which='major', labelsize=14)
    ax_fit.set_xlim(plot_dissimilarity_range)
    ax_fit.set_ylim(0, 0.26)
    # if show_fit_eqn_label:
    #     ax_fit.legend(frameon=True, fontsize=9, loc='lower right', bbox_to_anchor=(1.0, 0.0))

    if show_fit_eqn_label:
        # bbox_to_anchor places the legend just above the axes top edge,
        # in the white space of the GridSpec row 0 (no marginal there).
        ax_fit.legend(
            frameon=True, fontsize=9,
            loc='lower left',
            bbox_to_anchor=(0.0, 1.01),
            borderaxespad=0.0,
        )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

############################################################################
# PLOTTING & VISUALIZATION
############################################################################

def plot_dataset_corner(pdspl_samples, samples_to_plot, key_list, key_latex_labels, plot_ranges, save_path=None,
                        custom_colors_dict=None,):
    """
    Generates a combined corner plot for the base properties of multiple GGL datasets.
    """
    fig_corner_ref = None

    for sample_key in samples_to_plot:
        data_corner_sample = [pdspl_samples[sample_key]['table'][key] for key in key_list]
        data_corner_sample = np.array(data_corner_sample).T

        if fig_corner_ref is None:
            fig_corner_ref = corner.corner(
                data_corner_sample,
                labels=[key_latex_labels[key] for key in key_list],
                range=plot_ranges, 
                hist_kwargs={"density": True},
                color=pdspl_samples[sample_key]['color'],
                smooth=1,
                plot_datapoints=False,
                fill_contours=True,
                levels=(0.68, 0.95)
            )
        else:
            corner.corner(
                data_corner_sample,
                labels=[key_latex_labels[key] for key in key_list],
                range=plot_ranges, 
                hist_kwargs={"density": True},
                color=custom_colors_dict.get(sample_key, pdspl_samples[sample_key]['color']) if custom_colors_dict else pdspl_samples[sample_key]['color'],
                smooth=1,
                fig=fig_corner_ref,
                plot_datapoints=False,
                fill_contours=True,
                levels=(0.68, 0.95)
            )

    legend_handles = []
    for sample_key in samples_to_plot:
        color = custom_colors_dict.get(sample_key, pdspl_samples[sample_key]['color']) if custom_colors_dict else pdspl_samples[sample_key]['color']
        name = pdspl_samples[sample_key]['name']
        patch = mpatches.Patch(color=color, alpha=0.8, label=name)
        legend_handles.append(patch)

    fig_corner_ref.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        fontsize=18,
        frameon=True, 
        facecolor='white',
        framealpha=1.0
    )

    for ax in fig_corner_ref.get_axes():
        ax.tick_params(axis='both', which='major', labelsize=17, pad=8)
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
        ax.title.set_size(22)

    if save_path:
        fig_corner_ref.savefig(save_path, bbox_inches='tight', dpi=300)
        
    return fig_corner_ref

def plot_reldiff_corner(pdspl_samples, samples_to_plot, key_list, key_latex_labels, 
                        figsize=(14, 14), show_multiple_titles=True, 
                        error_type='asymmetric', title_y_spacing=0.15,
                        custom_ranges=None, save_path=None,
                        custom_colors_dict=None,
                        label_fontsize=22, tick_fontsize=16, legend_fontsize=20):
    """
    Generates a combined corner plot for the relative difference properties of multiple samples.
    """
    plot_labels = [key_latex_labels.get(k, k) for k in key_list]
    truth_values = [0.0] * len(key_list) 
    
    if isinstance(custom_ranges, dict):
        corner_range = [custom_ranges.get(k, 0.99) for k in key_list]
    else:
        corner_range = custom_ranges
    
    fig = plt.figure(figsize=figsize)
    valid_samples = [s for s in samples_to_plot if s in pdspl_samples]

    for sample_key in valid_samples:
        color = custom_colors_dict.get(sample_key, pdspl_samples[sample_key].get('color', '#333333')) if custom_colors_dict else pdspl_samples[sample_key].get('color', '#333333')
        table = pdspl_samples[sample_key]['pairs_analysis']['pairs_table_with_errors']
        samples_2d = np.vstack([table[k] for k in key_list]).T
        
        mask = ~np.isnan(samples_2d).any(axis=1)
        samples_2d = samples_2d[mask]
        
        fig = corner.corner(
            samples_2d,
            fig=fig,
            labels=plot_labels,
            range=corner_range,
            color=color,
            weights=np.ones(len(samples_2d)) / len(samples_2d),
            smooth=1,
            plot_datapoints=False,  
            plot_density=False,     
            fill_contours=True,    
            show_titles=False,      
            levels=[0.68, 0.95],    
            hist_kwargs={"density": True, "linewidth": 2, "histtype": "step"},
            contour_kwargs={"linewidths": 2},
            truths=truth_values, 
            truth_color="#444444",
            label_kwargs={"fontsize": label_fontsize} # Updated axis label size
        )

    if show_multiple_titles and valid_samples:
        ndim = len(key_list)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        
        for col_idx, param_key in enumerate(key_list):
            ax = axes[col_idx, col_idx] 
            param_label_raw = plot_labels[col_idx]
            param_math_text = param_label_raw.replace('$', '')
            
            for row_idx, sample_key in enumerate(valid_samples):
                color = custom_colors_dict.get(sample_key, pdspl_samples[sample_key].get('color', '#333333')) if custom_colors_dict else pdspl_samples[sample_key].get('color', '#333333')
                table = pdspl_samples[sample_key]['pairs_analysis']['pairs_table_with_errors']
                samples_1d = np.array(table[param_key])
                samples_1d = samples_1d[~np.isnan(samples_1d)]
                
                if error_type == 'symmetric':
                    mean_val = np.mean(samples_1d)
                    std_val = np.std(samples_1d)
                    title_str = fr"${param_math_text} = {mean_val:.3f} \pm {std_val:.3f}$"
                else:
                    q16, q50, q84 = np.percentile(samples_1d, [16, 50, 84])
                    lower_err = q50 - q16
                    upper_err = q84 - q50
                    title_str = fr"${param_math_text} = {q50:.3f}^{{+{upper_err:.3f}}}_{{-{lower_err:.3f}}}$"
                
                y_offset = 1.05 + (title_y_spacing * row_idx) 
                # Slightly increased the title font size here as well to match the new scaling
                ax.text(0.5, y_offset, title_str, transform=ax.transAxes, 
                        ha='center', va='bottom', color=color, fontsize=label_fontsize - 4)

    # Loop through all axes to increase tick label sizes and optionally rotate them
    for ax in fig.axes:
        ax.tick_params(axis='both', which='both', labelsize=tick_fontsize, direction='in')
        
        # When tick fonts get larger, x-axis labels tend to overlap. 
        # Setting rotation to 45 degrees helps prevent this.
        if ax.get_xticklabels():
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    legend_handles = []
    for sample_key in valid_samples:
        name = pdspl_samples[sample_key].get('name', sample_key)
        color = custom_colors_dict.get(sample_key, pdspl_samples[sample_key].get('color', '#333333')) if custom_colors_dict else pdspl_samples[sample_key].get('color', '#333333')
        line = mlines.Line2D([], [], color=color, linewidth=3, label=name)
        legend_handles.append(line)

    fig.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98), 
        bbox_transform=fig.transFigure,
        fontsize=legend_fontsize, # Updated legend size
        frameon=False
    )

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig


def generate_latex_summary_table(pdspl_samples, keys_to_plot):
    """
    Prints a formatted LaTeX summary table of the pairing analysis results,
    reporting mean values across realizations and dynamic fit headers.
    """
    # Determine the fit type from the first valid sample
    header_fit_type = "Fit" 
    for key in keys_to_plot:
        s = pdspl_samples.get(key)
        if s and "pairs_analysis" in s:
            fit_type = s["pairs_analysis"].get("scatter_vs_dissimilarity_fit_type")
            if fit_type == "linear":
                header_fit_type = "Linear Fit"
            elif fit_type == "power_law":
                header_fit_type = "PL Fit"
            break

    print(r"\begin{tabular}{l | c c c | c c}")
    print(r"\hline")
    print(fr"\multirow{{2}}{{*}}{{\textbf{{Sample}}}} & \multicolumn{{3}}{{c|}}{{\textbf{{Pairing at 20K deg$^2$}}}} & \multicolumn{{2}}{{c}}{{\textbf{{{header_fit_type}}}}} \\")
    print(r" & \textbf{\# Lenses} & \textbf{Mean \# Pairs} & \textbf{Mean ${\Delta\beta_{\rm E}}/{\beta_{\rm E}}$} & \textbf{$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(0)}$} & \textbf{$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(1)}$} \\")
    print(r"\hline")

    for key in keys_to_plot:
        s = pdspl_samples.get(key)
        if not s or "pairs_analysis" not in s:
            continue
            
        pa = s["pairs_analysis"]
        fit_coeffs = pa.get('scatter_vs_dissimilarity_fit_coeffs', [0, 0])
        fit_covs = pa.get('scatter_vs_dissimilarity_fit_cov', np.zeros((2,2)))
        
        a, b = fit_coeffs[0], fit_coeffs[1]
        da, db = np.sqrt(fit_covs[0, 0]), np.sqrt(fit_covs[1, 1])
        
        # Fetch means (falling back to single realization values if means aren't present)
        mean_pairs = pa.get('mean_num_pairs', pa.get('num_pairs', 0))
        mean_scatter = pa.get('mean_scatter_in_beta_E', pa.get('scatter_in_beta_E', 0.0))
        
        print(
            f"{s['name']} & "
            f"{pa['num_lenses']} & "
            f"{mean_pairs:.0f} & "
            f"{mean_scatter:.3f} & "
            f"{b:.2f} $\\pm$ {db:.3f} & "
            f"{a:.2f} $\\pm$ {da:.2f} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")