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
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from pdspl_utils.inference import beta_double_source_plane, beta2theta_e_ratio


############################################################################
# SCATTER MODEL EVALUATION
############################################################################

def eval_scatter_model(coeffs, fit_type, dissimilarity):
    """
    Evaluate the calibrated scatter model σ_{β_E, D}(D) for given dissimilarity
    values.  Always use this function instead of np.polyval so that the correct
    formula is applied for every fit type.

    Parameters
    ----------
    coeffs : array-like
        Fit coefficients as stored in
        ``pairs_analysis['scatter_vs_dissimilarity_fit_coeffs']``.

        * ``'linear'``     → ``[slope, intercept]``  (np.polyfit convention)
        * ``'quadrature'`` → ``[c1_slope, c0_floor]``
          model: σ = sqrt(c0_floor² + (c1_slope · D)²)
        * ``'power_law'``  → ``[log-slope, log-intercept]``
          model: σ = 10^intercept · D^slope

    fit_type : str
        One of ``'linear'``, ``'quadrature'``, ``'power_law'``.

    dissimilarity : float or array-like
        Dissimilarity value(s) at which to evaluate the model.

    Returns
    -------
    float or numpy.ndarray
        Predicted scatter σ_{β_E, D}.
    """
    dissimilarity = np.asarray(dissimilarity, dtype=float)

    if fit_type == "linear":
        # coeffs = [slope, intercept]
        return np.polyval(coeffs, dissimilarity)

    elif fit_type == "quadrature":
        # coeffs = [c1_slope, c0_floor]
        c1, c0 = coeffs[0], coeffs[1]
        return np.sqrt(c0 ** 2 + (c1 * dissimilarity) ** 2)

    elif fit_type == "power_law":
        # coeffs = [log-slope, log-intercept]
        return 10 ** np.polyval(coeffs, np.log10(dissimilarity))

    else:
        raise ValueError(
            f"Unknown fit_type '{fit_type}'. Use 'linear', 'quadrature', or 'power_law'."
        )


############################################################################
# PAIRING SIMULATION
############################################################################

def normalize_data(data, type="minmax", data_min=None, data_max=None):
    """
    Normalize data to [0, 1] (minmax) or zero-mean unit-variance (zscore).

    Parameters
    ----------
    data : array-like
    type : str  'minmax' | 'zscore'
    data_min, data_max : float, optional  (minmax only)

    Returns
    -------
    numpy.ndarray
    """
    data = np.array(data, dtype=float)
    if type == "minmax":
        data_min = np.nanmin(data) if data_min is None else data_min
        data_max = np.nanmax(data) if data_max is None else data_max
        if data_max == data_min:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    elif type == "zscore":
        std = np.nanstd(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - np.nanmean(data)) / std
    else:
        raise ValueError("Unsupported normalization type. Use 'minmax' or 'zscore'.")


def get_kdtree_pairs(
    table,
    pairing_keys,
    norm_type="zscore",
    n_neighbors=2,
    unique_pairs=True,
    use_log_metric=False,
):
    """
    Find nearest-neighbour pairs in a table using a kD-Tree.

    Parameters
    ----------
    table : astropy.table.Table
    pairing_keys : list of str
        Column names used as pairing features.
    norm_type : str
        Normalisation applied when ``use_log_metric=False``.
    n_neighbors : int
        Number of neighbours to query per point (including self → effective
        neighbours = n_neighbors - 1).
    unique_pairs : bool
        If True, keep only (min_i, max_j) so each unordered pair appears once.
    use_log_metric : bool
        If True, log-transform every pairing feature before building the
        kD-Tree.  Euclidean distance in log-space approximates the RMS relative
        difference, making the kD-Tree distance metric consistent with the RMS
        dissimilarity metric used downstream.  All pairing feature values must
        be strictly positive.  When True, ``norm_type`` is ignored.

    Returns
    -------
    pair_indices : numpy.ndarray, shape (N_pairs, 2)
    flat_distances : numpy.ndarray, shape (N_pairs,)
    """
    if use_log_metric:
        points = np.stack(
            [np.log(np.array(table[pk], dtype=float)) for pk in pairing_keys], axis=1
        )
        if not np.all(np.isfinite(points)):
            raise ValueError(
                "use_log_metric=True requires all pairing features to be strictly "
                "positive.  Check for zeros or negatives in pairing_keys."
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
    n_points = len(points)
    # Repeat the base indices for however many neighbors we are extracting (excluding self)
    base_indices = np.repeat(np.arange(n_points), n_neighbors - 1)
    
    # Flatten the neighbor indices and distances (skipping column 0, which is self-distance)
    neighbor_indices = indices[:, 1:].flatten()
    flat_distances = distances[:, 1:].flatten()
    # Create the Nx2 array of pairs
    pair_indices = np.column_stack((base_indices, neighbor_indices))

    # 4. Filter unique pairs
    if unique_pairs:
        # Sort each row so that (i, j) becomes (min(i,j), max(i,j))
        pair_indices = np.sort(pair_indices, axis=1)
        # Extract the unique rows
        pair_indices, unique_idx = np.unique(pair_indices, axis=0, return_index=True)
        flat_distances = flat_distances[unique_idx]

    return pair_indices, flat_distances


def build_generic_pairs_table(table, pair_indices, sort_by=None):
    """
    Build a pairs table where every column 'X' becomes 'X_1' and 'X_2'.

    Parameters
    ----------
    table : astropy.table.Table
    pair_indices : numpy.ndarray, shape (N, 2)
    sort_by : str, optional
        If given, enforce that entity '1' has the lower value of this column.

    Returns
    -------
    astropy.table.Table
    """
    idx1 = pair_indices[:, 0].copy()
    idx2 = pair_indices[:, 1].copy()

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
    """Fractional difference: 2(v2-v1)/(v2+v1)."""
    return 2 * (val2 - val1) / (val2 + val1)


def calculate_absolute_diff(val1, val2):
    """Absolute difference |v2 - v1|."""
    return np.abs(val2 - val1)


def get_pairs_table_PDSPL(data_table, pair_indices, cosmo):
    """
    Build the PDSPL-specific pairs table, adding DSPL β observables and
    relative/absolute differences for all standard columns.

    Parameters
    ----------
    data_table : astropy.table.Table
        Pre-processed GGL catalog (output of ``preprocess_ggl_table``).
    pair_indices : numpy.ndarray, shape (N, 2)
        Output of ``get_kdtree_pairs``.
    cosmo : astropy.cosmology instance

    Returns
    -------
    astropy.table.Table
    """
    # Build generic pairs table, enforcing z_S1 < z_S2
    pt = build_generic_pairs_table(data_table, pair_indices, sort_by="z_S")

    # Convenient combined redshift
    pt["z_D"] = 0.5 * (pt["z_D_1"] + pt["z_D_2"])

    # Rename for downstream convenience
    pt.rename_column("z_S_1", "z_S1")
    pt.rename_column("z_S_2", "z_S2")
    pt.rename_column("theta_E_1", "theta_E1")
    pt.rename_column("theta_E_2", "theta_E2")

    # DSPL beta values for each deflector
    _beta_E_DSPL_D1 = beta_double_source_plane(pt["z_D_1"], pt["z_S1"], pt["z_S2"], cosmo)
    _beta_E_DSPL_D2 = beta_double_source_plane(pt["z_D_2"], pt["z_S1"], pt["z_S2"], cosmo)

    beta_E_DSPL_D1 = beta2theta_e_ratio(_beta_E_DSPL_D1, gamma_pl=pt["gamma_pl_1"], lambda_mst=1)
    beta_E_DSPL_D2 = beta2theta_e_ratio(_beta_E_DSPL_D2, gamma_pl=pt["gamma_pl_2"], lambda_mst=1)

    pt["beta_E_DSPL"]   = 0.5 * (beta_E_DSPL_D1 + beta_E_DSPL_D2)
    pt["beta_E_pseudo"] = pt["theta_E1"] / pt["theta_E2"]

    # Relative scatter in beta_E (the key PDSPL observable noise metric)
    pt["rel_diff_beta_E"] = 1 - pt["beta_E_pseudo"] / pt["beta_E_DSPL"]

    # Relative and absolute differences for all standard columns
    diff_keys = [
        "sigma_v_D", "R_e_kpc", "R_e_arcsec", "Sigma_half_Msun/pc2",
        "mag_D_i", "flux_D_i", "z_D", "gamma_pl",
        "color_D_gr", "color_D_ri",
        "flux_ratio_D_gr", "flux_ratio_D_ri",
    ]
    for key in diff_keys:
        if f"{key}_1" in pt.colnames and f"{key}_2" in pt.colnames:
            pt[f"rel_diff_{key}"] = calculate_rel_diff(pt[f"{key}_1"], pt[f"{key}_2"])
            pt[f"abs_diff_{key}"] = calculate_absolute_diff(pt[f"{key}_1"], pt[f"{key}_2"])

    return pt


def inject_observational_errors(table, error_config):
    """
    Return a copy of ``table`` with Gaussian noise added according to
    ``error_config``.

    Parameters
    ----------
    table : astropy.table.Table
    error_config : dict
        Keys are column names; values are one of:
        * callable(table) → array of per-row σ values
        * float / int     → constant σ for all rows
        * array-like      → per-row σ array

    Returns
    -------
    astropy.table.Table  (copy with noise added and ``err_<col>`` columns)
    """
    noisy_table = table.copy()

    for col, err_spec in error_config.items():
        if col not in table.colnames:
            continue

        if callable(err_spec):
            err_array = err_spec(table)
        elif isinstance(err_spec, (int, float)):
            err_array = np.full(len(table), float(err_spec))
        else:
            err_array = np.array(err_spec)

        noisy_table[col] += np.random.normal(0, err_array)
        noisy_table[f"err_{col}"] = err_array

    return noisy_table


def compute_dissimilarity(pairs_table, dissimilarity_keys, method="rms"):
    """
    Compute the scalar dissimilarity metric for each pair.

    Parameters
    ----------
    pairs_table : astropy.table.Table
    dissimilarity_keys : list of str
        Column names of per-dimension difference metrics (e.g. ``rel_diff_z_D``).
    method : str
        ``'rms'``  → RMS of the listed difference columns.
        ``'chi2'`` → error-weighted distance (requires ``err_*`` columns).

    Returns
    -------
    numpy.ndarray
    """
    if method == "rms":
        diffs = np.array([pairs_table[k] for k in dissimilarity_keys])
        return np.sqrt(np.mean(diffs ** 2, axis=0))

    elif method == "chi2":
        chi2_terms = []
        for key in dissimilarity_keys:
            # Extract the base physical parameter (e.g., 'z_D' from 'rel_diff_z_D')
            base_feature = key.replace("rel_diff_", "")
            err_col1 = f"err_{base_feature}_1"
            err_col2 = f"err_{base_feature}_2"

            # Use explicit observational errors if they exist in the mock table
            if err_col1 in pairs_table.colnames and err_col2 in pairs_table.colnames:
                val1 = pairs_table[f"{base_feature}_1"]
                val2 = pairs_table[f"{base_feature}_2"]
                variance = pairs_table[err_col1] ** 2 + pairs_table[err_col2] ** 2
                variance = np.where(variance == 0, 1e-12, variance)
                term = ((val1 - val2) ** 2) / variance
            else:
                # Fallback: Treat the relative difference against an assumed 10% error.
                # This ensures features without explicit err_ columns (like colors) 
                # don't get mathematically ignored compared to the chi2 terms.
                assumed_fractional_error = 0.10
                term = (pairs_table[key] / assumed_fractional_error) ** 2

            chi2_terms.append(term)

        return np.sqrt(np.mean(chi2_terms, axis=0))

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'rms' or 'chi2'.")


############################################################################
# PLOTTING
############################################################################

def plot_beta_E_vs_D_MC(
    pdspl_samples,
    mc_results,
    fit_type="linear",
    save_path=None,
    show_fit_eqn_label=True,
    custom_colors_dict=None,
    custom_markers_dict=None,
    plot_dissimilarity_range=(0, 0.1),
    legend_fontsize=9,
):
    """
    Plot scatter of Δβ_E/β_E vs dissimilarity (left) and the MC-averaged
    scatter-vs-dissimilarity curve with fit (right).

    The fit type controls both the displayed equation and — crucially — what
    coefficients are stored back into ``pdspl_samples`` for downstream use.
    Use ``eval_scatter_model`` everywhere you need to evaluate those coefficients.

    Parameters
    ----------
    pdspl_samples : dict
        Keys are sample names; values contain at minimum:
        ``'pairs_analysis'``, ``'name'``, ``'color'``.
    mc_results : dict
        Output of the MC loop: ``binned_dissim``, ``binned_scatter``, etc.
    fit_type : str
        ``'linear'`` | ``'quadrature'`` | ``'power_law'``.
        Default is ``'quadrature'``.
    save_path : str, optional
    show_fit_eqn_label : bool
    custom_colors_dict : dict, optional
    custom_markers_dict : dict, optional
    plot_dissimilarity_range : tuple

    Returns
    -------
    matplotlib.figure.Figure
        The function also writes fit results into
        ``pdspl_samples[key]['pairs_analysis']``:
        * ``scatter_vs_dissimilarity_fit_coeffs`` — coefficients for ``eval_scatter_model``
        * ``scatter_vs_dissimilarity_fit_cov``    — covariance matrix
        * ``scatter_vs_dissimilarity_fit_type``   — fit_type string
    """
    _default_alpha = {
        "lsst_y10": 0.007, "lsst_y1": 0.01,
        "lsst_4most_spec-z": 0.03, "lsst_4most_spec-z_sigma_v": 0.04,
        "lsst_y10_photo_z": 0.007, "lsst_y10_spec_z": 0.01,
    }
    _default_markers = {
        "lsst_y10": "o", "lsst_y1": "^",
        "lsst_4most_spec-z": "s", "lsst_4most_spec-z_sigma_v": "D",
        "lsst_y10_photo_z": "o", "lsst_y10_spec_z": "s",
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
        ax.tick_params(axis="both", which="major", width=frame_width, length=6)
        ax.tick_params(axis="both", which="minor", width=frame_width, length=4)

    for sample_key, s in pdspl_samples.items():
        if "pairs_analysis" not in s:
            continue

        color  = (custom_colors_dict or {}).get(sample_key, s.get("color", "black"))
        marker = (custom_markers_dict or {}).get(sample_key, _default_markers.get(sample_key, "o"))
        alpha  = _default_alpha.get(sample_key, 0.01)

        # -------------------------------------------------------
        # LEFT PANEL: Single Realization Scatter & Distributions
        # -------------------------------------------------------
        tbl_err = s["pairs_analysis"]["pairs_table_with_errors"]
        tbl     = s["pairs_analysis"]["pairs_table"]

        dissim       = tbl_err["dissimilarity"]
        delta_beta_E = 1 - tbl["beta_E_pseudo"] / tbl["beta_E_DSPL"]

        mask         = (dissim < plot_dissimilarity_range[1]) & np.isfinite(delta_beta_E)
        dissim_clean = dissim[mask]
        delta_clean  = delta_beta_E[mask]

        # ── left panel: scatter ──────────────────────────────────────────
        ax_scatter.scatter(dissim_clean, delta_clean, color=color, alpha=alpha, s=10)

        try:
            x_grid = np.linspace(0, plot_dissimilarity_range[1], 200)
            y_grid = np.linspace(-0.4, 0.4, 200)
            ax_histx.plot(x_grid, gaussian_kde(dissim_clean)(x_grid), color=color, lw=1.5)
            ax_histy.plot(gaussian_kde(delta_clean)(y_grid), y_grid, color=color, lw=1.5)
        except np.linalg.LinAlgError:
            pass # Handle edge cases where KDE fails

        # 3. Binned Error Bars (Last Realization)
        percentiles = np.percentile(dissim_clean, np.arange(0, 101, 10))
        digitized   = np.digitize(dissim_clean, percentiles)
        bx, by, bye = [], [], []
        for i in range(1, len(percentiles)):
            bm = digitized == i
            if np.any(bm):
                bx.append(np.median(dissim_clean[bm]))
                by.append(np.nanmean(delta_clean[bm]))
                bye.append(np.nanstd(delta_clean[bm]))

        ax_scatter.errorbar(
            bx, by, yerr=bye, fmt=marker, color=color,
            markersize=8, label=s["name"], capsize=5, elinewidth=3,
        )

        # ── right panel: MC statistics + fit ─────────────────────────────
        all_dissim  = np.array(mc_results[sample_key]["binned_dissim"])
        all_scatter = np.array(mc_results[sample_key]["binned_scatter"])

        mean_dissim  = np.nanmean(all_dissim, axis=0)
        mean_scatter = np.nanmean(all_scatter, axis=0)
        std_scatter  = np.nanstd(all_scatter, axis=0)

        valid = np.isfinite(mean_dissim) & np.isfinite(mean_scatter) & (mean_scatter > 0)
        x_data, y_data, y_err = mean_dissim[valid], mean_scatter[valid], std_scatter[valid]

        ax_fit.scatter(x_data, y_data, marker=marker, color=color, alpha=0.8)

        if len(x_data) > 2:
            weights = np.where(y_err > 0, 1.0 / y_err, 0.0)

            if fit_type == "linear":
                coeffs, cov = np.polyfit(x_data, y_data, 1, w=weights, cov=True)
                y_fit = np.polyval(coeffs, x_data)
                slope, intercept = coeffs[0], coeffs[1]
                eq_latex = (
                    r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}} = (%.2f \pm %.2f) \mathcal{D} + (%.3f \pm %.3f)$"
                    % (slope, np.sqrt(cov[0, 0]), intercept, np.sqrt(cov[1, 1])) + r" $\rightarrow$ " + f"{s['name']}"
                )

            elif fit_type == "power_law":
                log_x = np.log10(x_data)
                log_y = np.log10(y_data)
                sigma_log_y = y_err / (y_data * np.log(10))
                w = np.where(sigma_log_y > 0, 1.0 / sigma_log_y, 0.0)
                coeffs, cov = np.polyfit(log_x, log_y, 1, w=w, cov=True)
                y_fit = 10 ** np.polyval(coeffs, log_x)
                eq_latex = (
                    r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}} = 10^{%.2f \pm %.2f} \mathcal{D}^{%.2f \pm %.2f}$"
                    % (coeffs[1], np.sqrt(cov[1, 1]), coeffs[0], np.sqrt(cov[0, 0])) + r" $\rightarrow$ " + f"{s['name']}"
                )

            elif fit_type == "quadrature":
                # Model: σ = sqrt(c0² + (c1·D)²)
                def quad_model(x, c0, c1):
                    return np.sqrt(c0 ** 2 + (c1 * x) ** 2)

                popt, pcov = curve_fit(
                    quad_model, x_data, y_data,
                    sigma=y_err, absolute_sigma=True, p0=[0.1, 1.0],
                )
                c0, c1 = popt
                err_c0, err_c1 = np.sqrt(np.diag(pcov))
                y_fit = quad_model(x_data, c0, c1)

                # Store as [c1_slope, c0_floor] so eval_scatter_model can unpack:
                #   coeffs[0] = c1_slope,  coeffs[1] = c0_floor
                coeffs = np.array([c1, c0])
                cov    = pcov  # 2×2 covariance in (c0, c1) order from curve_fit

                eq_latex = (
                    r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}"
                    r"= \sqrt{(%.3f \pm %.3f)^2 + [(%.2f \pm %.2f)\mathcal{D}]^2}$"
                    % (c0, err_c0, c1, err_c1) + r" $\rightarrow$ " + f"{s['name']}"
                )

            else:
                raise ValueError(f"Unknown fit_type '{fit_type}'.")

            ax_fit.plot(x_data, y_fit, linestyle="--", color=color, label=eq_latex)

            # Write fit results back into the sample dict for downstream use
            s["pairs_analysis"]["scatter_vs_dissimilarity_fit_coeffs"] = coeffs
            s["pairs_analysis"]["scatter_vs_dissimilarity_fit_cov"]    = cov
            s["pairs_analysis"]["scatter_vs_dissimilarity_fit_type"]   = fit_type

        ax_fit.fill_between(x_data, y_data - y_err, y_data + y_err, color=color, alpha=0.2)

    # ── axis formatting ──────────────────────────────────────────────────
    ax_scatter.set_xlabel(r"$\mathcal{D}_{\rm deflector}$", fontsize=18)
    ax_scatter.set_ylabel(
        r"$\Delta \beta_{E} / \beta_{E} = 1 - \beta_{\rm E, pseudo}/\beta_{\rm E, DSPL}$",
        fontsize=18,
    )
    ax_scatter.tick_params(axis="both", which="major", labelsize=14)
    ax_scatter.set_xlim(plot_dissimilarity_range)
    ax_scatter.set_ylim(-0.43, 0.43)
    ax_scatter.legend(frameon=True, fontsize=12)

    ax_histx.axis("off")
    ax_histy.axis("off")

    ax_fit.set_xlabel(r"$\mathcal{D}_{\rm deflector}$", fontsize=18)
    ax_fit.set_ylabel(
        r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}} = \sigma(\Delta \beta_{E} / \beta_{E})$",
        fontsize=18,
    )
    ax_fit.tick_params(axis="both", which="major", labelsize=14)
    ax_fit.set_xlim(plot_dissimilarity_range)
    ax_fit.set_ylim(0, 0.26)

    if show_fit_eqn_label:
        ax_fit.legend(
            frameon=True, fontsize=legend_fontsize,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            borderaxespad=0.0,
        )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_dataset_corner(
    pdspl_samples,
    samples_to_plot,
    key_list,
    key_latex_labels,
    plot_ranges,
    save_path=None,
    custom_colors_dict=None,
):
    """
    Corner plot of the raw lens-population properties for multiple samples.
    """
    fig_corner_ref = None

    for sample_key in samples_to_plot:
        color = (custom_colors_dict or {}).get(
            sample_key, pdspl_samples[sample_key].get("color", "black")
        )
        data = np.array([pdspl_samples[sample_key]["table"][k] for k in key_list]).T

        corner_kwargs = dict(
            labels=[key_latex_labels[k] for k in key_list],
            range=plot_ranges,
            hist_kwargs={"density": True},
            color=color,
            smooth=1,
            plot_datapoints=False,
            fill_contours=True,
            levels=(0.68, 0.95),
        )

        if fig_corner_ref is None:
            fig_corner_ref = corner.corner(data, **corner_kwargs)
        else:
            corner.corner(data, fig=fig_corner_ref, **corner_kwargs)

    # Legend
    handles = [
        mpatches.Patch(
            color=(custom_colors_dict or {}).get(k, pdspl_samples[k].get("color", "black")),
            alpha=0.8,
            label=pdspl_samples[k]["name"],
        )
        for k in samples_to_plot
    ]
    fig_corner_ref.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=18,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
    )

    for ax in fig_corner_ref.get_axes():
        ax.tick_params(axis="both", which="major", labelsize=17, pad=8)
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
        ax.title.set_size(22)

    if save_path:
        fig_corner_ref.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig_corner_ref


def plot_reldiff_corner(
    pdspl_samples,
    samples_to_plot,
    key_list,
    key_latex_labels,
    figsize=(14, 14),
    show_multiple_titles=True,
    error_type="asymmetric",
    title_y_spacing=0.15,
    custom_ranges=None,
    save_path=None,
    custom_colors_dict=None,
    label_fontsize=22,
    tick_fontsize=16,
    legend_fontsize=20,
):
    """
    Corner plot of relative-difference properties for paired deflectors.
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
        color = (custom_colors_dict or {}).get(
            sample_key, pdspl_samples[sample_key].get("color", "#333333")
        )
        table = pdspl_samples[sample_key]["pairs_analysis"]["pairs_table_with_errors"]
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
            label_kwargs={"fontsize": label_fontsize},
        )

    if show_multiple_titles and valid_samples:
        ndim = len(key_list)
        axes = np.array(fig.axes).reshape((ndim, ndim))

        for col_idx, param_key in enumerate(key_list):
            ax = axes[col_idx, col_idx]
            param_math = plot_labels[col_idx].replace("$", "")

            for row_idx, sample_key in enumerate(valid_samples):
                color = (custom_colors_dict or {}).get(
                    sample_key, pdspl_samples[sample_key].get("color", "#333333")
                )
                table = pdspl_samples[sample_key]["pairs_analysis"]["pairs_table_with_errors"]
                vals = np.array(table[param_key])
                vals = vals[~np.isnan(vals)]

                if error_type == "symmetric":
                    mean_val = np.mean(vals)
                    std_val  = np.std(vals)
                    title_str = fr"${param_math} = {mean_val:.3f} \pm {std_val:.3f}$"
                else:
                    q16, q50, q84 = np.percentile(vals, [16, 50, 84])
                    title_str = (
                        fr"${param_math} = {q50:.3f}"
                        fr"^{{+{q84-q50:.3f}}}_{{-{q50-q16:.3f}}}$"
                    )

                y_offset = 1.05 + title_y_spacing * row_idx
                ax.text(
                    0.5, y_offset, title_str,
                    transform=ax.transAxes, ha="center", va="bottom",
                    color=color, fontsize=label_fontsize - 4,
                )

    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", labelsize=tick_fontsize, direction="in")
        if ax.get_xticklabels():
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    handles = [
        mlines.Line2D(
            [], [],
            color=(custom_colors_dict or {}).get(k, pdspl_samples[k].get("color", "#333333")),
            linewidth=3,
            label=pdspl_samples[k].get("name", k),
        )
        for k in valid_samples
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=fig.transFigure,
        fontsize=legend_fontsize,
        frameon=False,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def generate_latex_summary_table(pdspl_samples, keys_to_plot):
    """
    Print a LaTeX summary table of pairing statistics and scatter-model fit.

    Reports mean ± std for ``# Pairs`` across MC realizations (stored in
    ``mc_results`` via ``mean_num_pairs`` / ``scatter_in_beta_E``).

    Also note: a lens can appear in more than one pair because the kD-Tree
    matches each lens to its nearest neighbour independently, so the pairing
    is not one-to-one.
    """
    # Determine fit type from first valid sample for the header
    header_fit = "Fit"
    for key in keys_to_plot:
        s = pdspl_samples.get(key)
        if s and "pairs_analysis" in s:
            ft = s["pairs_analysis"].get("scatter_vs_dissimilarity_fit_type")
            if ft == "linear":
                header_fit = "Linear Fit"
            elif ft == "quadrature":
                header_fit = "Quadrature Fit"
            elif ft == "power_law":
                header_fit = "Power-Law Fit"
            break

    print(r"\begin{tabular}{l | c c c | c c}")
    print(r"\hline")
    print(
        fr"\multirow{{2}}{{*}}{{\textbf{{Sample}}}} "
        fr"& \multicolumn{{3}}{{c|}}{{\textbf{{Pairing at 20K deg$^2$}}}} "
        fr"& \multicolumn{{2}}{{c}}{{\textbf{{{header_fit}}}}} \\"
    )
    print(
        r" & \textbf{\# Lenses} & \textbf{Mean \# Pairs}"
        r" & \textbf{$\overline{\Delta\beta_{\rm E}/\beta_{\rm E}}$}"
        r" & \textbf{$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(0)}$}"
        r" & \textbf{$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(1)}$} \\"
    )
    print(r"\hline")

    for key in keys_to_plot:
        s = pdspl_samples.get(key)
        if not s or "pairs_analysis" not in s:
            continue

        pa       = s["pairs_analysis"]
        fit_type = pa.get("scatter_vs_dissimilarity_fit_type", "linear")
        coeffs   = pa.get("scatter_vs_dissimilarity_fit_coeffs", [0, 0])
        fit_covs = pa.get("scatter_vs_dissimilarity_fit_cov", np.zeros((2, 2)))

        # For quadrature: coeffs = [c1_slope, c0_floor]; curve_fit cov is in (c0,c1) order
        # For linear:     coeffs = [slope, intercept]; polyfit cov is in (slope, intercept) order
        if fit_type == "quadrature":
            c1, c0     = coeffs[0], coeffs[1]
            # curve_fit pcov is (c0, c1) → diag indices 0→c0, 1→c1
            dc0 = np.sqrt(fit_covs[0, 0])
            dc1 = np.sqrt(fit_covs[1, 1])
        else:
            # linear: coeffs = [slope, intercept]
            c1, c0 = coeffs[0], coeffs[1]
            dc1 = np.sqrt(fit_covs[0, 0])
            dc0 = np.sqrt(fit_covs[1, 1])

        mean_pairs  = pa.get("mean_num_pairs",        pa.get("num_pairs", 0))
        mean_scatter = pa.get("mean_scatter_in_beta_E", pa.get("scatter_in_beta_E", 0.0))

        print(
            f"{s['name']} & "
            f"{pa['num_lenses']} & "
            f"{mean_pairs:.0f} & "
            f"{mean_scatter:.3f} & "
            f"{c0:.3f} $\\pm$ {dc0:.3f} & "
            f"{c1:.2f} $\\pm$ {dc1:.2f} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print()
    print(
        r"% Note: a single lens may appear in more than one pair because the kD-Tree"
        "\n"
        r"% assigns each lens its nearest neighbour independently (not a one-to-one matching)."
    )


def plot_pairing_scatter(
    pdspl_samples,
    samples_to_plot,
    pairing_param_keys,
    pairing_param_labels,
    pairing_hist_labels=None,
    custom_colors_dict=None,
    custom_markers_dict=None,
    n_scatter_points=2000,
    save_path=None,
    figsize=None,
    reldiff_range=(-0.4, 0.4),
):
    """
    Layout per pairing parameter showing the quality of deflector pairing.
    Plots two parameters per row. Scatter and Histogram heights are strictly locked.
    """
    n_params = len(pairing_param_keys)
    params_per_row = 2
    n_rows = int(np.ceil(n_params / params_per_row))
    
    if figsize is None:
        # Adjusted height so the constrained aspect ratios fit nicely without excess white space
        figsize = (18, 3.8 * n_rows)
 
    _default_markers = {
        "lsst_y1":                   "^",
        "lsst_y10":                  "o",
        "lsst_4most_spec-z":         "s",
        "lsst_4most_spec-z_sigma_v": "D",
        "lsst_y10_photo_z":          "o",
        "lsst_y10_spec_z":           "s",
    }
 
    valid = [
        k for k in samples_to_plot
        if k in pdspl_samples and "pairs_analysis" in pdspl_samples[k]
    ]
    
    # Explicitly define width ratios to calculate exact aspect constraints later
    w_sc = 1.0     # Scatter plot width
    w_hist = 2.2   # Histogram width (made wider per request)
    w_space = 0.4  # Invisible spacer column width
    
    fig, axes = plt.subplots(
        n_rows, 5,
        figsize=figsize,
        gridspec_kw={
            "width_ratios": [w_sc, w_hist, w_space, w_sc, w_hist], 
            "wspace": 0.10, 
            "hspace": 0.10
        },
    )
    
    if n_rows == 1:
        axes = axes[np.newaxis, :] 
        
    # Hide the spacer column completely
    for r in range(n_rows):
        axes[r, 2].set_visible(False)
        axes[r, 2].axis("off")
        
    pairing_hist_labels = pairing_hist_labels or {}
    global_legend_handles = []
    legend_names_added = set()
 
    for idx, key in enumerate(pairing_param_keys):
        row = idx // params_per_row
        col_base = (idx % params_per_row) * 3 
        
        ax_sc   = axes[row, col_base]   
        ax_hist = axes[row, col_base + 1]   
 
        col_1   = f"{key}_1"
        col_2   = f"{key}_2"
        rel_key = f"rel_diff_{key}"
        
        label_sc = pairing_param_labels.get(key, key)
        fallback_hist_label = r"$\Delta$" + label_sc.replace("$", "").replace(r"\,", "") + r" / mean"
        label_hist = pairing_hist_labels.get(key, fallback_hist_label)
 
        all_vals = []
 
        for sample_key in valid:
            s  = pdspl_samples[sample_key]
            pt = s["pairs_analysis"]["pairs_table_with_errors"]
 
            color  = (custom_colors_dict or {}).get(sample_key, s.get("color", "black"))
            marker = (custom_markers_dict or {}).get(sample_key, _default_markers.get(sample_key, "o"))
            name   = s.get("name", sample_key)
 
            if col_1 not in pt.colnames or col_2 not in pt.colnames:
                continue
 
            v1 = np.array(pt[col_1], dtype=float)
            v2 = np.array(pt[col_2], dtype=float)
            ok = np.isfinite(v1) & np.isfinite(v2)
            v1, v2 = v1[ok], v2[ok]
 
            all_vals.extend(v1)
            all_vals.extend(v2)
 
            # ── scatter (down-sampled) ──────────────────────────────
            n   = min(n_scatter_points, len(v1))
            rnd_idx = np.random.choice(len(v1), size=n, replace=False)
            ax_sc.scatter(
                v1[rnd_idx], v2[rnd_idx],
                color=color, alpha=0.30, s=6,
                marker=marker, rasterized=True, linewidths=0,
            )
 
            # ── relative-difference smoothed KDE ────────────────────
            if rel_key in pt.colnames:
                rd = np.array(pt[rel_key], dtype=float)
            else:
                rd = 2.0 * (v2 - v1) / (v2 + v1)
            
            rd = rd[np.isfinite(rd)]
            
            if len(rd) > 1:
                kde = gaussian_kde(rd)
                x_eval = np.linspace(reldiff_range[0], reldiff_range[1], 200)
                y_eval = kde(x_eval)
                
                ax_hist.plot(x_eval, y_eval, color=color, linewidth=1.8)
 
            # Collect for figure-level legend
            if name not in legend_names_added:
                global_legend_handles.append(
                    mlines.Line2D(
                        [], [], color=color, linewidth=3,
                        marker=marker, markersize=6, label=name,
                    )
                )
                legend_names_added.add(name)
 
        # ── formatting ─────────────────────────
        if all_vals:
            lo  = np.nanpercentile(all_vals, 0.5)
            hi  = np.nanpercentile(all_vals, 99.5)
            pad = (hi - lo) * 0.05
            lim = (lo - pad, hi + pad)
            ax_sc.plot(lim, lim, "--", color="#888888", linewidth=1.0, zorder=3)
            ax_sc.set_xlim(lim)
            ax_sc.set_ylim(lim)
        
        # Force scatter to be perfectly square (1:1)
        ax_sc.set_box_aspect(1)
        
        # Force the histogram's height to exactly match the square scatter plot's height
        ax_hist.set_box_aspect(w_sc / w_hist)
        
        ax_sc.set_xlabel(label_sc + r"$_{\,1}$",  fontsize=19)
        ax_sc.set_ylabel(label_sc + r"$_{\,2}$",  fontsize=19)
        ax_sc.tick_params(labelsize=15)
        ax_sc.xaxis.set_major_locator(MaxNLocator(4))
        ax_sc.yaxis.set_major_locator(MaxNLocator(4))
        for sp in ax_sc.spines.values():
            sp.set_linewidth(1.0)
 
        ax_hist.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--", zorder=3)
        ax_hist.set_xlabel(label_hist, fontsize=19)
        ax_hist.set_ylabel("density", fontsize=19)
        ax_hist.set_xlim(reldiff_range)
        ax_hist.tick_params(labelsize=15)
        ax_hist.yaxis.set_major_locator(MaxNLocator(4))
        for sp in ax_hist.spines.values():
            sp.set_linewidth(1.0)
            
    # Add perfect match line to legend handles
    global_legend_handles.append(
        mlines.Line2D([], [], color="#888888", linewidth=1.2,
                      linestyle="--", label=r"$y = x$ (perfect match)")
    )
            
    # Hide any unused trailing subplots if n_params is odd
    for blank_idx in range(n_params, n_rows * params_per_row):
        row = blank_idx // params_per_row
        col_base = (blank_idx % params_per_row) * 3
        axes[row, col_base].set_visible(False)
        axes[row, col_base + 1].set_visible(False)
 
    # Align labels securely by columns
    fig.align_ylabels(axes[:, 0]) # Left block scatters
    fig.align_ylabels(axes[:, 1]) # Left block histograms
    fig.align_ylabels(axes[:, 3]) # Right block scatters
    fig.align_ylabels(axes[:, 4]) # Right block histograms

    # hide density y-axis for histograms to reduce clutter
    for row in range(n_rows):
        axes[row, 1].set_yticklabels([])
        axes[row, 4].set_yticklabels([])
        axes[row, 1].set_ylabel("")
        axes[row, 4].set_ylabel("")
        axes[row, 1].set_yticks([])
        axes[row, 4].set_yticks([])
    
    # Master Figure Legend
    fig.legend(
        handles=global_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(global_legend_handles),
        fontsize=18,
        frameon=False,
    )
 
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
 
    return fig