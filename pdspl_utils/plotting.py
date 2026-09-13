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
            # smooth=1,
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


# def plot_w0wa(
#     scenarios,
#     truth,
#     fixed_params,
#     scenarios_to_plot=("lsst_y1", "lsst_y10"),
#     params=("w0", "wa"),
#     colors=None,
#     linestyles=None,
#     linewidths=None,
#     fill_contours=None,
#     alphas=None,
#     ranges=None,
#     labels=(r"$w_0$", r"$w_a$"),
#     figsize=(9, 9),
#     samples_key="samples",
# ):
#     colors       = colors or {}
#     linestyles   = linestyles or {}
#     linewidths   = linewidths or {}
#     fill_contours = fill_contours or {}
#     alphas       = alphas or {}

#     # Column ordering used by the original samples.
#     sampled_keys = [key for key in truth if key not in fixed_params]
#     column_ids = [sampled_keys.index(param) for param in params]

#     valid_scenarios = [
#         key for key in scenarios_to_plot
#         if key in scenarios and samples_key in scenarios[key]
#     ]

#     fig = plt.figure(figsize=figsize)
#     handles = []

#     for key in valid_scenarios:
#         samples = scenarios[key][samples_key][:, column_ids]

#         color = colors.get(key, scenarios[key].get("color", "black"))
#         linestyle = linestyles.get(key, "-")
#         linewidth = linewidths.get(key, 2.5)
#         alpha = alphas.get(key, 1.0)
#         fill = fill_contours.get(key, True)

#         rgb = mcolors.to_rgb(color)
#         contourf_kwargs = {
#             "colors": [
#                 (*rgb, 0.0),
#                 (*rgb, 0.30 * alpha),
#                 (*rgb, 0.90 * alpha),
#             ]
#         }

#         fig = corner.corner(
#             samples,
#             fig=fig,
#             labels=labels,
#             range=ranges,
#             color=color,
#             smooth=2,
#             smooth1d=2,
#             plot_datapoints=False,
#             plot_density=False,
#             fill_contours=fill,
#             contourf_kwargs=contourf_kwargs if fill else {},
#             levels=(0.68, 0.95),
#             hist_kwargs={
#                 "linewidth": linewidth,
#                 "linestyle": linestyle,
#             },
#             contour_kwargs={
#                 "linewidths": linewidth,
#                 "linestyles": linestyle,
#                 "alpha": alpha,
#             },
#             truths=[truth[param] for param in params],
#             truth_color="#444444",
#             label_kwargs={"fontsize": 20},
#             max_n_ticks=4,
#             show_titles=False,
#         )

#         handles.append(
#             mlines.Line2D(
#                 [], [],
#                 color=color,
#                 linestyle=linestyle,
#                 linewidth=linewidth,
#                 label=scenarios[key]["name"],
#             )
#         )

#     for ax in fig.axes:
#         ax.tick_params(
#             axis="both",
#             direction="in",
#             width=1.5,
#             length=6,
#             labelsize=13,
#         )
#         for spine in ax.spines.values():
#             spine.set_linewidth(1.5)

#     fig.legend(
#         handles=handles,
#         loc="upper right",
#         bbox_to_anchor=(0.96, 0.96),
#         frameon=False,
#         fontsize=15,
#     )

#     return fig



def plot_w0wa(
    scenarios,
    truth,
    fixed_params,
    scenarios_to_plot=("lsst_y1", "lsst_y10"),
    params=("w0", "wa"),
    ranges=None,
    labels=(r"$w_0$", r"$w_a$"),
    figsize=(8, 8),
    samples_key="samples",
    colors=None,
    linestyles=None,
    linewidths=None,
    fill_contours=None,
    alphas=None,
    hide_1d=True,
    labelpad=6,
    tick_labelsize=13,
    label_fontsize=22,
    legend_fontsize=15,
):
    colors = colors or {}
    linestyles = linestyles or {}
    linewidths = linewidths or {}
    fill_contours = fill_contours or {}
    alphas = alphas or {}

    # infer sample column positions from the original parameter ordering
    sampled_keys = [k for k in truth.keys() if k not in fixed_params]
    col_ids = [sampled_keys.index(p) for p in params]
    truth_vals = [truth[p] for p in params]

    valid_scenarios = [
        k for k in scenarios_to_plot
        if k in scenarios and samples_key in scenarios[k]
    ]

    fig = plt.figure(figsize=figsize)

    handles = []

    for key in valid_scenarios:
        s = scenarios[key][samples_key][:, col_ids]

        color = colors.get(key, scenarios[key].get("color", "black"))
        ls = linestyles.get(key, "-")
        lw = linewidths.get(key, 2.5)
        fill = fill_contours.get(key, True)
        alpha = alphas.get(key, 1.0)

        rgb = mcolors.to_rgb(color)

        fig = corner.corner(
            s,
            fig=fig,
            labels=labels,
            range=ranges,
            color=color,
            weights=np.ones(len(s)) / len(s),
            smooth=1,
            smooth1d=2,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=fill,
            show_titles=False,
            # contourf_kwargs=contourf_kwargs,
            levels=[0.68, 0.95],
            hist_kwargs={"linewidth": lw, "linestyle": ls},
            contour_kwargs={
                "linewidths": lw,
                "linestyles": ls,
                "alpha": alpha,
            },
            truths=truth_vals,
            truth_color="#444444",
            label_kwargs={"fontsize": label_fontsize},
            max_n_ticks=4,
        )

        handles.append(
            mlines.Line2D(
                [], [],
                color=color,
                linewidth=lw,
                linestyle=ls,
                label=scenarios[key]["name"],
            )
        )

    # corner gives a 2x2 axis grid for two parameters
    axes = np.array(fig.axes).reshape((2, 2))

    ax2d = axes[1, 0]

    # Hide the two 1D panels and the unused upper-right panel
    if hide_1d:
        axes[0, 0].set_visible(False)
        axes[0, 1].set_visible(False)
        axes[1, 1].set_visible(False)

        # [left, bottom, width, height]
        ax2d.set_position([0.14, 0.13, 0.82, 0.80])

        # Restyle only the visible 2D axis
        for spine in ax2d.spines.values():
            spine.set_linewidth(1.5)

    ax2d.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=1.5,
        length=6,
        labelsize=tick_labelsize,
        pad=4,
    )

    

    # corner rotates some tick labels; reset them
    for tick in ax2d.get_xticklabels():
        tick.set_rotation(0)
        tick.set_horizontalalignment("center")

    for tick in ax2d.get_yticklabels():
        tick.set_rotation(0)
        tick.set_verticalalignment("center")

    # Reset labels after corner has positioned them
    ax2d.set_xlabel(labels[0], fontsize=label_fontsize)
    ax2d.set_ylabel(labels[1], fontsize=label_fontsize)

    # Explicit axis-coordinate placement:
    # less negative values move labels closer to the axes
    ax2d.xaxis.set_label_coords(0.5, -0.075)
    ax2d.yaxis.set_label_coords(-0.085, 0.5)

    # activate minor ticks on the 2D panel
    ax2d.minorticks_on()

    # make minor ticks point inward and be slightly shorter than major ticks
    ax2d.tick_params(
        axis="both",
        which="minor",
        direction="in",
        width=1.5,
        length=4,
    )

    # Remove any figure-level legend
    for legend in list(fig.legends):
        legend.remove()

    # Place the legend relative to the actual 2D panel
    legend = ax2d.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=len(handles),
        fontsize=legend_fontsize,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.5,
        borderaxespad=0.0,
    )

    return fig, ax2d