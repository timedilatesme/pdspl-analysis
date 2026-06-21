# pdspl_utils/plotting.py

import numpy as np
import matplotlib.pyplot as plt
import corner
import matplotlib.lines as mlines
import matplotlib.colors as mcolors


def plot_dspl_corner(
    scenarios,
    truth,
    fixed_params,
    scenarios_to_plot=None,
    custom_ranges=None,
    latex_labels=None,
    figsize=(16, 16),
    show_multiple_titles=True,
    error_type="asymmetric",
    title_y_spacing=0.18,
    samples_key="samples",
    custom_linestyles_dict=None,
    custom_linewidths_dict=None,
    custom_colors_dict=None,
    custom_fill_contours_dict=None,
    custom_alphas_dict=None,
    save_path=None,
    titles_fontsize=13,
):
    """
    Publication-ready corner plot for multiple MCMC scenarios.

    Parameters
    ----------
    scenarios : dict
        Keys are scenario names; each value must contain at minimum:
        ``'samples'`` (numpy array, shape N×D), ``'name'``, ``'color'``.
    truth : dict
        Parameter name → true value.  Parameters in ``fixed_params`` are
        excluded from the axes automatically.
    fixed_params : dict
        Parameters that were held fixed during inference (excluded from axes).
    scenarios_to_plot : list of str, optional
        Ordered subset of scenario keys to draw.  Defaults to all.
    custom_ranges : list, optional
        Per-parameter axis ranges, same length as free parameters.
    latex_labels : dict, optional
        Parameter name → LaTeX string.
    figsize : tuple
    show_multiple_titles : bool
        If True, draw per-scenario median ± 1σ titles above each diagonal panel.
    error_type : str  ``'asymmetric'`` | ``'symmetric'``
    title_y_spacing : float
        Vertical spacing between stacked titles.
    samples_key : str
        Key in each scenario dict that holds the MCMC samples array.
    custom_linestyles_dict : dict, optional  scenario key → linestyle
    custom_linewidths_dict : dict, optional  scenario key → linewidth
    custom_colors_dict : dict, optional      scenario key → colour
    custom_fill_contours_dict : dict, optional  scenario key → bool
    custom_alphas_dict : dict, optional      scenario key → alpha
    save_path : str, optional
    titles_fontsize : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    if scenarios_to_plot is None:
        scenarios_to_plot = list(scenarios.keys())

    sampled_keys  = [k for k in truth.keys() if k not in fixed_params]
    truth_values  = [truth[k] for k in sampled_keys]

    if latex_labels is None:
        latex_labels = {k: k for k in sampled_keys}
    plot_labels = [latex_labels.get(k, k) for k in sampled_keys]

    # ── default dicts ────────────────────────────────────────────────────
    custom_linestyles_dict     = custom_linestyles_dict     or {}
    custom_linewidths_dict     = custom_linewidths_dict     or {}
    custom_fill_contours_dict  = custom_fill_contours_dict  or {}
    custom_alphas_dict         = custom_alphas_dict         or {}

    valid_scenarios = [
        k for k in scenarios_to_plot
        if k in scenarios and samples_key in scenarios[k]
    ]

    def _color(key):
        return (custom_colors_dict or {}).get(key, scenarios[key].get("color", "black"))

    colors      = [_color(k)                                          for k in valid_scenarios]
    linestyles  = [custom_linestyles_dict.get(k,    "-")              for k in valid_scenarios]
    linewidths  = [custom_linewidths_dict.get(k,    2.5)              for k in valid_scenarios]
    fill_flags  = [custom_fill_contours_dict.get(k, True)             for k in valid_scenarios]
    alphas      = [custom_alphas_dict.get(k,        1.0)              for k in valid_scenarios]

    fig = plt.figure(figsize=figsize)

    for idx, key in enumerate(valid_scenarios):
        sc            = scenarios[key]
        current_color = colors[idx]
        current_ls    = linestyles[idx]
        current_lw    = linewidths[idx]
        current_fill  = fill_flags[idx]
        current_alpha = alphas[idx]

        base_rgb   = mcolors.to_rgb(current_color)
        rgba_bg    = (*base_rgb, 0.0)
        rgba_outer = (*base_rgb, 0.3 * current_alpha)
        rgba_inner = (*base_rgb, 1.0 * current_alpha)
        c_kwargs   = {"colors": [rgba_bg, rgba_outer, rgba_inner]} if current_fill else {}

        fig = corner.corner(
            sc[samples_key],
            fig=fig,
            labels=plot_labels,
            range=custom_ranges,
            color=current_color,
            weights=np.ones(len(sc[samples_key])) / len(sc[samples_key]),
            smooth=2,
            smooth1d=2,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=current_fill,
            show_titles=False,
            contourf_kwargs=c_kwargs,
            levels=[0.68, 0.95],
            hist_kwargs={"linewidth": current_lw, "linestyle": current_ls},
            contour_kwargs={
                "linewidths": current_lw,
                "linestyles": current_ls,
                "alpha":      current_alpha,
            },
            truths=truth_values,
            truth_color="#444444",
            label_kwargs={"fontsize": 18},
            max_n_ticks=4,
        )

    # ── axis styling ─────────────────────────────────────────────────────
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.tick_params(
            axis="both", which="major",
            direction="in", width=1.5, length=6, labelsize=11, pad=1,
        )
        if ax.get_xlabel():
            ax.xaxis.labelpad = 35
        if ax.get_ylabel():
            ax.yaxis.labelpad = 35

    # ── stacked coloured titles ───────────────────────────────────────────
    if show_multiple_titles and valid_scenarios:
        ndim = len(sampled_keys)
        axes = np.array(fig.axes).reshape((ndim, ndim))

        for col_idx, param_key in enumerate(sampled_keys):
            ax             = axes[col_idx, col_idx]
            param_math_txt = plot_labels[col_idx].replace("$", "")

            for row_idx, sc_key in enumerate(valid_scenarios):
                sc         = scenarios[sc_key]
                samples_1d = sc[samples_key][:, col_idx]

                if error_type == "symmetric":
                    m, s       = np.mean(samples_1d), np.std(samples_1d)
                    title_str  = fr"${param_math_txt} = {m:.2f} \pm {s:.2f}$"
                else:
                    q16, q50, q84 = np.percentile(samples_1d, [16, 50, 84])
                    lo, hi        = q50 - q16, q84 - q50
                    if param_key == "beta_c0":
                        title_str = (
                            fr"${param_math_txt} = {q50:.3f}"
                            fr"^{{+{hi:.3f}}}_{{-{lo:.3f}}}$"
                        )
                    else:
                        title_str = (
                            fr"${param_math_txt} = {q50:.2f}"
                            fr"^{{+{hi:.2f}}}_{{-{lo:.2f}}}$"
                        )

                y_offset = 1.05 + title_y_spacing * row_idx
                ax.text(
                    0.5, y_offset, title_str,
                    transform=ax.transAxes, ha="center", va="bottom",
                    color=colors[valid_scenarios.index(sc_key)],
                    fontsize=titles_fontsize, fontweight="bold",
                )

    # ── legend ───────────────────────────────────────────────────────────
    handles = [
        mlines.Line2D(
            [], [],
            color=colors[valid_scenarios.index(k)],
            linewidth=linewidths[valid_scenarios.index(k)],
            linestyle=custom_linestyles_dict.get(k, "-"),
            label=scenarios[k]["name"],
        )
        for k in valid_scenarios
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=fig.transFigure,
        fontsize=16,
        frameon=False,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig