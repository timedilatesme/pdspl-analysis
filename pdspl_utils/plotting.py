import numpy as np
import matplotlib.pyplot as plt
import corner
import matplotlib.lines as mlines
import matplotlib.colors as mcolors

def plot_dspl_corner(scenarios, truth, fixed_params, scenarios_to_plot=None, 
                     custom_ranges=None, latex_labels=None, figsize=(16, 16),
                     show_multiple_titles=True, error_type='asymmetric',
                     title_y_spacing=0.18, samples_key='samples', custom_linestyles_dict=None,
                     custom_linewidths_dict=None,
                     custom_colors_dict=None, 
                     custom_fill_contours_dict=None,
                     custom_alphas_dict=None,
                     save_path=None, titles_fontsize=13):
    """
    Generates a combined corner plot for multiple MCMC scenarios with custom stacked titles.
    Optimized for publication-ready styling and high-dimensional grids.
    """
    # 1. Setup Data and Labels
    if scenarios_to_plot is None:
        scenarios_to_plot = list(scenarios.keys())

    sampled_keys = [k for k in truth.keys() if k not in fixed_params]
    truth_values = [truth[k] for k in sampled_keys]
    
    if latex_labels is None:
        latex_labels = {k: k for k in sampled_keys}
    plot_labels = [latex_labels.get(k, k) for k in sampled_keys]

    fig = plt.figure(figsize=figsize)

    # Filter out scenarios that haven't been run yet
    valid_scenarios = [key for key in scenarios_to_plot if key in scenarios and 'samples' in scenarios[key]]
    colors_for_valid_scenarios = [scenarios[key]['color'] if custom_colors_dict is None else custom_colors_dict.get(key, scenarios[key]['color']) for key in valid_scenarios]
    if custom_linestyles_dict is None:
        custom_linestyles_dict = {}
    if custom_linewidths_dict is None:
        custom_linewidths_dict = {}
    if custom_fill_contours_dict is None:
        custom_fill_contours_dict = {}
    if custom_alphas_dict is None:
        custom_alphas_dict = {}
    linestyles_for_valid_scenarios = [custom_linestyles_dict.get(key, '-') for key in valid_scenarios]
    linewidths_for_valid_scenarios = [custom_linewidths_dict.get(key, 2.5) for key in valid_scenarios]
    contour_fill_options_for_valid_scenarios = [custom_fill_contours_dict.get(key, True) for key in valid_scenarios]
    alphas_for_valid_scenarios = [custom_alphas_dict.get(key, 1.0) for key in valid_scenarios]

    # 2. Draw Contours
    for key in valid_scenarios:
        sc = scenarios[key]

        idx = valid_scenarios.index(key)
        
        current_color = colors_for_valid_scenarios[idx]
        current_ls = linestyles_for_valid_scenarios[idx]
        current_lw = linewidths_for_valid_scenarios[idx]
        current_fill_contours = contour_fill_options_for_valid_scenarios[idx]
        current_alpha = alphas_for_valid_scenarios[idx]

        # Extract the RGB values of the current color
        base_rgb = mcolors.to_rgb(current_color)
        
        # Scale ONLY the fill opacities by the user's custom overall alpha
        rgba_bg    = (*base_rgb, 0.0)  
        rgba_outer = (*base_rgb, 0.3 * current_alpha)  
        rgba_inner = (*base_rgb, 1.0 * current_alpha)  
        
        c_kwargs = {"colors": [rgba_bg, rgba_outer, rgba_inner]} if current_fill_contours else {}


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
            fill_contours=current_fill_contours,  
            show_titles=False,      
            contourf_kwargs=c_kwargs,
            
            levels=[0.68, 0.95],    
            hist_kwargs={"linewidth": current_lw, "linestyle": current_ls},
            contour_kwargs={"linewidths": current_lw, "linestyles": current_ls, "alpha": current_alpha},
            
            truths=truth_values, 
            truth_color="#444444",
            label_kwargs={"fontsize": 18},
            max_n_ticks=4 
        )

    # Apply thicker borders (spines), inward ticks, and label padding to all generated axes
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        ax.tick_params(axis='both', which='major', direction='in', width=1.5, length=6, labelsize=11, pad=1)
        
        if ax.get_xlabel():
            ax.xaxis.labelpad = 35
        if ax.get_ylabel():
            ax.yaxis.labelpad = 35

    # 3. Stacked Colored Titles
    if show_multiple_titles and valid_scenarios:
        ndim = len(sampled_keys)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        
        for col_idx, param_key in enumerate(sampled_keys):
            ax = axes[col_idx, col_idx] 
            
            param_label_raw = plot_labels[col_idx]
            param_math_text = param_label_raw.replace('$', '')
            
            for row_idx, sc_key in enumerate(valid_scenarios):
                sc = scenarios[sc_key]
                samples_1d = sc['samples'][:, col_idx]
                
                if error_type == 'symmetric':
                    mean_val = np.mean(samples_1d)
                    std_val = np.std(samples_1d)
                    title_str = fr"${param_math_text} = {mean_val:.2f} \pm {std_val:.2f}$"
                else:
                    q16, q50, q84 = np.percentile(samples_1d, [16, 50, 84])
                    lower_err = q50 - q16
                    upper_err = q84 - q50
                    if param_key == 'beta_c0':
                        title_str = fr"${param_math_text} = {q50:.3f}^{{+{upper_err:.3f}}}_{{-{lower_err:.3f}}}$"
                    else:
                        title_str = fr"${param_math_text} = {q50:.2f}^{{+{upper_err:.2f}}}_{{-{lower_err:.2f}}}$"
                
                y_offset = 1.05 + (title_y_spacing * row_idx) 
                
                ax.text(0.5, y_offset, title_str, transform=ax.transAxes, 
                        ha='center', va='bottom', 
                        color=colors_for_valid_scenarios[valid_scenarios.index(sc_key)], 
                        fontsize=titles_fontsize, fontweight='bold')

    # 4. Custom Legend
    legend_handles = []
    for key in valid_scenarios:
        sc = scenarios[key]
        line = mlines.Line2D([], [], color=colors_for_valid_scenarios[valid_scenarios.index(key)], linewidth=linewidths_for_valid_scenarios[valid_scenarios.index(key)], 
                             label=f"{sc['name']}", linestyle=custom_linestyles_dict.get(key, '-'))
        legend_handles.append(line)

    fig.legend(
        handles=legend_handles,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98), 
        bbox_transform=fig.transFigure,
        fontsize=16,
        frameon=False
    )

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300) 

    return fig