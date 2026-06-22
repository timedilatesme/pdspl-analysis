# pdspl_utils/__init__.py

from .data_utils import preprocess_ggl_table
from .pairing import (
    normalize_data,
    get_kdtree_pairs,
    build_generic_pairs_table,
    get_pairs_table_PDSPL,
    inject_observational_errors,
    compute_dissimilarity,
    eval_scatter_model,
    plot_beta_E_vs_D_MC,
    plot_dataset_corner,
    plot_reldiff_corner,
    plot_pairing_scatter,
    generate_latex_summary_table,
)
from .inference import (
    beta_double_source_plane,
    beta2theta_e_ratio,
    get_beta_z_derivatives,
    draw_lens_from_given_zs,
    DSPLLikelihood,
    run_dspl_inference,
    check_mcmc_convergence,
)
from .plotting import plot_dspl_corner

__all__ = [
    # data
    "preprocess_ggl_table",
    # pairing
    "normalize_data",
    "get_kdtree_pairs",
    "build_generic_pairs_table",
    "get_pairs_table_PDSPL",
    "inject_observational_errors",
    "compute_dissimilarity",
    "eval_scatter_model",
    "plot_beta_E_vs_D_MC",
    "plot_dataset_corner",
    "plot_reldiff_corner",
    "plot_pairing_scatter",
    "generate_latex_summary_table",
    # inference
    "beta_double_source_plane",
    "beta2theta_e_ratio",
    "get_beta_z_derivatives",
    "draw_lens_from_given_zs",
    "DSPLLikelihood",
    "run_dspl_inference",
    "check_mcmc_convergence",
    # plotting
    "plot_dspl_corner",
]