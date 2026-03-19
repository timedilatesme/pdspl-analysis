import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from IPython.display import display, Latex, Markdown


def plot_custom_corner(results, scenario_keys, params_to_plot,
                       custom_colors=None, 
                       custom_legend_labels=None, param_ranges=None, param_truths=None, 
                       custom_param_labels=None, filename=None, ):
    """
    Generates a stylized corner plot for a specific subset of parameters and scenarios.
    """
    print(f"\nGenerating plot for: {' vs '.join(scenario_keys)}")
    
    if custom_param_labels is None:
        custom_param_labels = {}

    # 1. Default Style Dictionaries
    default_ranges = {
        r"$\Omega_m$": (0, 1), 
        r"$w_0$": (-2, 0),
        r"$w_a$": (-3.0, 3.0),
        r"$\bar{\lambda}_{\rm int}$": (0.8, 1.2),
        r"$\sigma({\lambda}_{\rm int})$": (0.0, 0.12),
        r"$\bar{\gamma}_{\rm pl}$": (1.9, 2.3),
        r"$\sigma({\gamma}_{\rm pl})$": (0.0, 0.28),
        r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(0)}$": (-0.05, 0.25),
        r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(1)}$": (-0.05, 0.25), 
        r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$": (-0.05, 0.25) 
    }
    
    default_truths = {
        r"$\Omega_m$": 0.3, r"$w_0$": -1.0, r"$w_a$": 0.0,
        r"$\bar{\lambda}_{\rm int}$": 1.0, r"$\sigma({\lambda}_{\rm int})$": 0.05,
        r"$\bar{\gamma}_{\rm pl}$": 2.078, r"$\sigma({\gamma}_{\rm pl})$": 0.16,
        r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(0)}$": 0.0,
        r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(1)}$": 0.10, 
        r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$": 0.10 
    }

    if param_ranges is not None: default_ranges.update(param_ranges)
    if param_truths is not None: default_truths.update(param_truths)

    ranges_list = [default_ranges.get(p, None) for p in params_to_plot]
    display_labels = [custom_param_labels.get(p, p) for p in params_to_plot]
    custom_colors = {key: res['color'] for idx, key in enumerate(scenario_keys) for res in [results[key]]} if custom_colors is None else custom_colors

    fig = None
    for idx, key in enumerate(scenario_keys):
        res = results[key]
        
        # 2. Extract strictly the requested columns using original names
        col_indices = []
        for p in params_to_plot:
            if p == r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$":
                if r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(1)}$" in res['labels']:
                    col_indices.append(res['labels'].index(r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(1)}$"))
                elif r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$" in res['labels']:
                    col_indices.append(res['labels'].index(r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$"))
                elif r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(0)}$" in res['labels']:
                    col_indices.append(res['labels'].index(r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(0)}$"))
                else:
                    raise ValueError(f"No valid scatter parameter found for '{key}'.")
            elif p in res['labels']:
                col_indices.append(res['labels'].index(p))
            else:
                raise ValueError(f"Parameter '{p}' not found in labels for '{key}'.")
                
        samples_subset = res['samples'][:, col_indices]
        
        # Override the scatter truth line if the specific run has a known true scatter
        truths_list = []
        for p in params_to_plot:
            if p == r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$" and 'true_scatter' in res:
                truths_list.append(res['true_scatter'])
            else:
                truths_list.append(default_truths.get(p, None))
                
        truths_list = truths_list if idx == 0 else None
        
        # 3. Generate Corner Plot
        fig = corner.corner(
            samples_subset, fig=fig, labels=display_labels, range=ranges_list,
            truths=truths_list, truth_color="#444444", color=custom_colors[key],smooth=1.5,
            plot_datapoints=False, plot_density=False, fill_contours=True,
            levels=[0.68, 0.95],
            hist_kwargs={"density": True, "linewidth": 2.5, "histtype": "step"},
            contour_kwargs={"linewidths": 2.0},
            label_kwargs={"fontsize": 22}, show_titles=False
        )

    # 4. Stacked Multi-colored Titles
    ndim = len(params_to_plot)
    axes = np.array(fig.axes).reshape((ndim, ndim))

    for i in range(ndim):
        ax = axes[i, i]
        for row_idx, key in enumerate(scenario_keys):
            res = results[key]
            p = params_to_plot[i]
            
            if p == r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$":
                if r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(1)}$" in res['labels']: col_idx = res['labels'].index(r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(1)}$")
                elif r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$" in res['labels']: col_idx = res['labels'].index(r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$")
                elif r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(0)}$" in res['labels']: col_idx = res['labels'].index(r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(0)}$")
            elif p in res['labels']: 
                col_idx = res['labels'].index(p)
                
            samples_1d = res['samples'][:, col_idx]
            q16, q50, q84 = np.percentile(samples_1d, [16, 50, 84])
            
            clean_label = display_labels[i].replace('$', '')
            title_str = fr"${clean_label} = {q50:.3f}^{{+{q84-q50:.3f}}}_{{-{q50-q16:.3f}}}$"
            
            y_offset = 1.05 + (0.18 * row_idx) 
            ax.text(0.5, y_offset, title_str, transform=ax.transAxes, 
                    ha='center', va='bottom', color=custom_colors[key], fontsize=16)

    # 5. Rich Legend
    legend_handles = []
    for k in scenario_keys:
        res = results[k]
        label_str = custom_legend_labels[k] if custom_legend_labels and k in custom_legend_labels else res.get('label', k)
        legend_handles.append(mlines.Line2D([], [], color=custom_colors[k], linewidth=5, label=label_str))

    fig.legend(
        handles=legend_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
        bbox_transform=fig.transFigure, fontsize=15, frameon=True,
        facecolor='white', framealpha=1.0, edgecolor='black', labelspacing=1.2
    )

    # 6. Aesthetic Formatting & Coordinate Override
    for ax in fig.get_axes():
        # --- THE FIX: Added direction='in' here ---
        ax.tick_params(axis='both', which='major', labelsize=14, pad=6, width=1.5, length=6, direction='in')
        
        for spine in ax.spines.values(): spine.set_linewidth(1.5)
            
        # Hard-code the label coordinates so they don't crash into the ticks
        if ax.get_xlabel():
            ax.xaxis.set_label_coords(0.5, -0.4) 
        if ax.get_ylabel():
            ax.yaxis.set_label_coords(-0.35, 0.5)

    fig.subplots_adjust(top=0.82) 
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def display_latex_constraints_table(results_dict, scenario_keys, params_to_list, table_title="Parameter Constraints"):
    """
    Renders a KaTeX-safe table in Jupyter AND returns a proper LaTeX tabular string for Overleaf.
    """
    # ==========================================
    # 1. Initialize both strings
    # ==========================================
    # Jupyter Visual String (Math mode array, no multicolumn)
    jup_str = r"$$" + "\n"
    jup_str += r"\def\arraystretch{1.5}" + "\n"
    jup_str += r"\begin{array}{l | " + "c " * len(params_to_list) + r"}" + "\n"
    jup_str += r"\hline \hline" + "\n"
    
    # Paper Export String (Text mode tabular, standard LaTeX)
    pap_str = r"\begin{table*}[htbp]" + "\n"
    pap_str += r"\centering" + "\n"
    pap_str += r"\renewcommand{\arraystretch}{1.5}" + "\n"
    pap_str += r"\begin{tabular}{l | " + "c " * len(params_to_list) + r"}" + "\n"
    pap_str += r"\hline \hline" + "\n"

    # ==========================================
    # 2. Build the Header
    # ==========================================
    header = [r"\textbf{Scenario}"] + params_to_list
    header_row = " & ".join(header) + r" \\" + "\n"
    
    jup_str += header_row + r"\hline" + "\n"
    pap_str += header_row + r"\hline" + "\n"

    # ==========================================
    # 3. Build the Data Rows
    # ==========================================
    for key in scenario_keys:
        res = results_dict[key]
        clean_name = key.replace(' + Pan+', ' (Pan+)').replace('_', r'\_')
        
        jup_row = [fr"\text{{{clean_name}}}"]
        pap_row = [clean_name]

        for p in params_to_list:
            # --- Robust Fallback Logic for Scatter Alias ---
            col_idx = None
            if p == r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$":
                for alias in [r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(1)}$", r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$", r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(0)}$"]:
                    if alias in res['labels']:
                        col_idx = res['labels'].index(alias)
                        break
            elif p in res['labels']:
                col_idx = res['labels'].index(p)

            # --- Calculate Margins ---
            if col_idx is not None:
                samples_1d = res['samples'][:, col_idx]
                q16, q50, q84 = np.percentile(samples_1d, [16, 50, 84])
                upper_err = q84 - q50
                lower_err = q50 - q16
                
                # Format exactly as needed for each environment
                val_str = fr"{q50:.3f}^{{+{upper_err:.3f}}}_{{-{lower_err:.3f}}}"
                jup_row.append(val_str)
                pap_row.append(fr"${val_str}$") # Text-mode tabular needs math wrappers
            else:
                jup_row.append(r"\text{-}")
                pap_row.append(r"-")

        jup_str += " & ".join(jup_row) + r" \\" + "\n"
        pap_str += " & ".join(pap_row) + r" \\" + "\n"

    # ==========================================
    # 4. Close the strings
    # ==========================================
    jup_str += r"\hline \hline" + "\n"
    jup_str += r"\end{array}" + "\n"
    jup_str += r"$$"
    
    pap_str += r"\hline \hline" + "\n"
    pap_str += r"\end{tabular}" + "\n"
    pap_str += fr"\caption{{{table_title}}}" + "\n"
    pap_str += r"\label{tab:constraints}" + "\n"
    pap_str += r"\end{table*}" + "\n"

    # ==========================================
    # 5. Display and Return
    # ==========================================
    display(Markdown(f"### {table_title}"))
    display(Latex(jup_str))
    
    return pap_str