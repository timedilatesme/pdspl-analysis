"""
Unified toolkit for Double Source Plane Lens (DSPL) / Pseudo-DSPL (PDSPL) cosmology.
"""

import os
from multiprocessing import Pool

import emcee
import numpy as np
from astropy.cosmology import Flatw0waCDM


# ---------------------------------------------------------------------------
# 1. PHYSICS FUNCTIONS
# ---------------------------------------------------------------------------

def beta_double_source_plane(
    z_lens,
    z_source_1,
    z_source_2,
    cosmo,
    *,
    return_z_derivatives=False,
    n_grid=1000,
):
    r"""Evaluate the DSPL geometric ratio at the supplied flat cosmology.

    β = (D_{ds1}/D_{s1}) · (D_{s2}/D_{ds2})

    The calculation uses

    .. math::

        \beta = \frac{1-\chi_{\rm l}/\chi_{\rm s1}}
                      {1-\chi_{\rm l}/\chi_{\rm s2}},

    where :math:`\chi` is the line-of-sight comoving distance. For catalogs
    with at least 50 systems, the distances are interpolated from a common
    grid to accelerate repeated likelihood evaluations.

    When ``return_z_derivatives`` is true, the function also returns the
    analytic derivatives with respect to all three redshifts. Consequently,
    beta and its derivatives are always evaluated from the same distances and
    at the same cosmological parameter point.

    Parameters
    ----------
    z_lens, z_source_1, z_source_2 : float or array-like
        Deflector, nearer-source, and farther-source redshifts.
    cosmo : astropy.cosmology instance
        Flat cosmology at which to evaluate the geometry.
    return_z_derivatives : bool, optional
        If true, also return ``dβ/dz_l``, ``dβ/dz_s1``, and ``dβ/dz_s2``.
    n_grid : int, optional
        Size of the interpolation grid used for large catalogs.

    Returns
    -------
    beta : numpy.ndarray
        Geometric DSPL ratio.
    dbeta_dzl, dbeta_dzs1, dbeta_dzs2 : numpy.ndarray, optional
        Returned only when ``return_z_derivatives=True``.
    """
    z_lens, z_source_1, z_source_2 = np.broadcast_arrays(
        np.atleast_1d(z_lens),
        np.atleast_1d(z_source_1),
        np.atleast_1d(z_source_2),
    )

    if not cosmo.is_flat:
        raise ValueError(
            "beta_double_source_plane requires a flat cosmology."
        )

    if z_lens.size < 50:
        chi_l = cosmo.comoving_distance(z_lens).value
        chi_s1 = cosmo.comoving_distance(z_source_1).value
        chi_s2 = cosmo.comoving_distance(z_source_2).value
    else:
        max_z = max(
            np.max(z_lens),
            np.max(z_source_1),
            np.max(z_source_2),
        )
        z_grid = np.linspace(0.0, max_z + 0.1, n_grid)
        chi_grid = cosmo.comoving_distance(z_grid).value
        chi_l = np.interp(z_lens, z_grid, chi_grid)
        chi_s1 = np.interp(z_source_1, z_grid, chi_grid)
        chi_s2 = np.interp(z_source_2, z_grid, chi_grid)

    numerator = 1.0 - chi_l / chi_s1
    denominator = 1.0 - chi_l / chi_s2
    beta_geo = numerator / denominator

    if not return_z_derivatives:
        return beta_geo

    # Partial derivatives with respect to the three comoving distances.
    dbeta_dchi_l = (
        -denominator / chi_s1 + numerator / chi_s2
    ) / denominator**2
    dbeta_dchi_s1 = chi_l / (chi_s1**2 * denominator)
    dbeta_dchi_s2 = -numerator * chi_l / (
        chi_s2**2 * denominator**2
    )

    # dchi/dz = c/H(z) = D_H/E(z).
    hubble_distance = cosmo.hubble_distance.value
    dchi_dzl = hubble_distance / cosmo.efunc(z_lens)
    dchi_dzs1 = hubble_distance / cosmo.efunc(z_source_1)
    dchi_dzs2 = hubble_distance / cosmo.efunc(z_source_2)

    dbeta_dzl = dbeta_dchi_l * dchi_dzl
    dbeta_dzs1 = dbeta_dchi_s1 * dchi_dzs1
    dbeta_dzs2 = dbeta_dchi_s2 * dchi_dzs2

    return beta_geo, dbeta_dzl, dbeta_dzs1, dbeta_dzs2


def beta2theta_e_ratio(beta_dsp, gamma_pl=2.0, lambda_mst=1.0):
    """
    Convert geometric β into the observable Einstein-radius ratio.

    For a power-law mass profile with MST parameter λ:
        μ = [β − (1−λ)(1−β)]^{1/(γ−1)}

    Parameters
    ----------
    beta_dsp : float or array-like
        The geometric scaling factor.
    gamma_pl : float, optional
        The power-law slope of the mass density profile (default is 2.0, isothermal).
    lambda_mst : float, optional
        The internal mass sheet transform parameter (default is 1.0, no mass sheet).

    Returns
    -------
    mu : float or array-like
        The predicted ratio of Einstein radii (theta_E1 / theta_E2). Returns np.nan
        for unphysical geometries where the argument drops below zero.
    """
    base_term = beta_dsp - (1 - lambda_mst) * (1 - beta_dsp)
    if np.isscalar(base_term):
        if base_term <= 0:
            return np.nan
    else:
        base_term = np.where(base_term <= 0, np.nan, base_term)
    return base_term ** (1.0 / (gamma_pl - 1.0))


def theta_e_ratio_derivatives(beta_geo, gamma_pl, lambda_mst, model_mu):
    """Return derivatives of the Einstein-radius ratio model.

    Returns derivatives with respect to the geometric ratio, the MST
    parameter, and the power-law slope, in that order.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        transformed_beta = beta_geo - (1 - lambda_mst) * (1 - beta_geo)
        common_factor = model_mu / ((gamma_pl - 1.0) * transformed_beta)

        dmu_dbeta = common_factor * (2.0 - lambda_mst)
        dmu_dlambda = common_factor * (1.0 - beta_geo)
        dmu_dgamma = (
            -model_mu
            * np.log(transformed_beta)
            / (gamma_pl - 1.0) ** 2
        )

    return dmu_dbeta, dmu_dlambda, dmu_dgamma


def draw_lens_from_given_zs(
    z_lens, z1, z2,
    lambda_mst_mean, lambda_mst_sigma,
    gamma_pl_mean, gamma_pl_sigma,
    sigma_beta_intrinsic,
    sigma_meas_rel,
    dissimilarity=0.0,
    cosmo=None,
    down_sampling=1,
    with_noise=True,
    redshift_error_rel=0.0,
):
    """
    Generate a single mock DSPL / PDSPL data dictionary.

    ``sigma_beta_intrinsic`` should be the output of
    ``eval_scatter_model(coeffs, fit_type, dissimilarity)`` — i.e. the
    dissimilarity-dependent scatter evaluated at the given pair's dissimilarity.

    Parameters
    ----------
    z_lens, z1, z2 : float
        Redshifts of deflector, nearer source, farther source.
    lambda_mst_mean, lambda_mst_sigma : float
        Population mean and scatter of the MST parameter.
    gamma_pl_mean, gamma_pl_sigma : float
        Population mean and scatter of the power-law slope.
    sigma_beta_intrinsic : float
        Fractional scatter on β_E from deflector dissimilarity (PDSPL) or
        0 for ideal DSPL.  Pass ``eval_scatter_model(coeffs, fit_type, D)``
        here — do NOT pass np.polyval output.
    sigma_meas_rel : float
        Combined fractional measurement + LoS scatter on β_E.
    dissimilarity : float
        Metric characterizing lens complexity for the heteroscedastic model.
    cosmo : astropy.cosmology instance
    down_sampling : float
        Statistical weight factor (DSF = N_total / N_subset).
    with_noise : bool
        True → add stochastic noise (noisy mock).
        False → return Asimov expectation (recommended for forecasts).
    redshift_error_rel : float or list/array of 3 floats
        Relative photo-z error σ_z/(1+z) for [z_lens, z1, z2].
        Scalar → same value for all three.

    Returns
    -------
    dict
        Mock-system inputs and observables required by ``DSPLLikelihood``.
    """
    # Photometric redshift error propagation
    if np.isscalar(redshift_error_rel):
        err_zl, err_zs1, err_zs2 = (
            redshift_error_rel * (1.0 + z_lens),
            redshift_error_rel * (1.0 + z1),
            redshift_error_rel * (1.0 + z2),
        )
    else:
        err_zl  = redshift_error_rel[0] * (1.0 + z_lens)
        err_zs1 = redshift_error_rel[1] * (1.0 + z1)
        err_zs2 = redshift_error_rel[2] * (1.0 + z2)

    has_redshift_errors = err_zl > 0 or err_zs1 > 0 or err_zs2 > 0
    if has_redshift_errors:
        beta_geo, dbeta_dzl, dbeta_dzs1, dbeta_dzs2 = (
            beta_double_source_plane(
                z_lens,
                z1,
                z2,
                cosmo,
                return_z_derivatives=True,
            )
        )
    else:
        beta_geo = beta_double_source_plane(z_lens, z1, z2, cosmo=cosmo)

    mu_model = beta2theta_e_ratio(
        beta_dsp=beta_geo,
        gamma_pl=gamma_pl_mean,
        lambda_mst=lambda_mst_mean,
    )

    dmu_dbeta, dmu_dlambda, dmu_dgamma = theta_e_ratio_derivatives(
        beta_geo,
        gamma_pl_mean,
        lambda_mst_mean,
        mu_model,
    )

    if has_redshift_errors:
        sigma_photoz_true_sq = (
            (dmu_dbeta * dbeta_dzl * err_zl) ** 2
            + (dmu_dbeta * dbeta_dzs1 * err_zs1) ** 2
            + (dmu_dbeta * dbeta_dzs2 * err_zs2) ** 2
        )
    else:
        sigma_photoz_true_sq = 0.0

    sigma_pop_sq = (
        (dmu_dlambda * lambda_mst_sigma) ** 2
        + (dmu_dgamma * gamma_pl_sigma) ** 2
        + (sigma_beta_intrinsic * mu_model) ** 2
    )
    sigma_meas_sq = (sigma_meas_rel * mu_model) ** 2
    sigma_pop_true_sq = sigma_pop_sq + sigma_photoz_true_sq

    if with_noise:
        sigma_tot = np.sqrt((sigma_meas_sq + sigma_pop_true_sq) / down_sampling)
        beta_measured = np.random.normal(loc=mu_model, scale=sigma_tot)
    else:
        beta_measured = mu_model

    return {
        "z_lens":    z_lens, "z_source": z1, "z_source2": z2,
        "beta_dspl": beta_measured,
        "sigma_beta_dspl":  np.sqrt(sigma_meas_sq),
        "sigma_pop_true_sq": sigma_pop_true_sq,
        "with_noise":     with_noise,
        "dissimilarity":  dissimilarity,
        "err_zl":  err_zl,  "err_zs1": err_zs1,  "err_zs2": err_zs2,
        "down_sampling": down_sampling,
    }


# ---------------------------------------------------------------------------
# 2. INFERENCE FRAMEWORK
# ---------------------------------------------------------------------------

ALL_PARAM_NAMES = [
    "h0", "om", "w0", "wa",
    "lambda_int", "lambda_sigma",
    "gamma_pl", "gamma_sigma",
    "beta_c0", "beta_c1", "beta_c2",
]


class DSPLLikelihood:
    """
    Log-likelihood for hierarchical DSPL / PDSPL inference.

    Scatter model (quadrature form, consistent with pairing.py quadrature fit):
        σ_{β_E, D}(D) = sqrt(c0² + (c1·D)²)

    where c0 = beta_c0 and c1 = beta_c1 are sampled or fixed parameters.
    beta_c2 is kept for legacy compatibility but set to 0 in the quadrature
    framework (it was the quadratic coefficient of the old linear model).
    """

    def __init__(
        self,
        kwargs_likelihood_list,
        sampled_params,
        fixed_params,
        priors=None,
        scatter_model_type="linear",
    ):
        """Unpack the mock catalog and configure the likelihood."""
        self.sampled_params = sampled_params
        self.fixed_params = fixed_params
        self.scatter_model_type = scatter_model_type

        def unpack(key, default=None):
            if default is None:
                values = [item[key] for item in kwargs_likelihood_list]
            else:
                values = [item.get(key, default) for item in kwargs_likelihood_list]
            return np.asarray(values).ravel()

        self.down_sampling = unpack("down_sampling", 1.0)
        self.z_l = unpack("z_lens")
        self.z_s1 = unpack("z_source")
        self.z_s2 = unpack("z_source2")
        self.beta_obs = unpack("beta_dspl")
        self.sigma_meas = unpack("sigma_beta_dspl")
        self.sigma_pop_true_sq = unpack("sigma_pop_true_sq", 0.0)
        self.dissimilarity = unpack("dissimilarity", 0.0)
        self.err_zl = unpack("err_zl", 0.0)
        self.err_zs1 = unpack("err_zs1", 0.0)
        self.err_zs2 = unpack("err_zs2", 0.0)
        self.has_redshift_errors = (
            np.any(self.err_zl > 0)
            or np.any(self.err_zs1 > 0)
            or np.any(self.err_zs2 > 0)
        )

        self.is_asimov = not any(
            item.get("with_noise", True) for item in kwargs_likelihood_list
        )

        # Default priors
        self.priors = {
            "h0":           ("uniform", 50.0, 90.0),
            "om":           ("uniform",  0.0,  1.0),
            "w0":           ("uniform", -3.0,  0.0),
            "wa":           ("uniform", -5.0,  5.0),
            "lambda_int":   ("uniform",  0.8,  1.2),
            "lambda_sigma": ("uniform",  0.0,  0.2),
            "gamma_pl":     ("uniform",  1.0,  3.0),
            "gamma_sigma":  ("uniform",  0.0,  0.5),
            "beta_c0":      ("uniform", -1.0,  1.0),
            "beta_c1":      ("uniform", -5.0,  5.0),
            "beta_c2":      ("uniform", -5.0,  5.0),
        }
        if priors is not None:
            self.priors.update(priors)

    def _get_full_params(self, theta):
        params = self.fixed_params.copy()
        for name, val in zip(self.sampled_params, theta):
            params[name] = val
        return params

    def log_prior(self, theta):
        """Evaluates the unnormalized prior probability of the parameter set."""
        lp = 0.0
        for i, name in enumerate(self.sampled_params):
            val = theta[i]
            prior_type, p1, p2 = self.priors[name]
            if prior_type == "uniform":
                if not (p1 < val < p2):
                    return -np.inf
            elif prior_type == "gaussian":
                lp += -0.5 * ((val - p1) ** 2 / p2 ** 2) - np.log(p2)
        return lp

    def log_likelihood(self, theta):
        """Calculates the log-likelihood of the observed data given the model parameters."""
        p = self._get_full_params(theta)

        try:
            cosmo_step = Flatw0waCDM(H0=p["h0"], Om0=p["om"], w0=p["w0"], wa=p["wa"])
            if self.has_redshift_errors:
                beta_geo, dbeta_dzl, dbeta_dzs1, dbeta_dzs2 = (
                    beta_double_source_plane(
                        self.z_l,
                        self.z_s1,
                        self.z_s2,
                        cosmo_step,
                        return_z_derivatives=True,
                    )
                )
            else:
                beta_geo = beta_double_source_plane(
                    self.z_l,
                    self.z_s1,
                    self.z_s2,
                    cosmo_step,
                )
        except Exception:
            return -np.inf

        model_mu = beta2theta_e_ratio(
            beta_geo,
            gamma_pl=p["gamma_pl"],
            lambda_mst=p["lambda_int"],
        )
        if not np.all(np.isfinite(model_mu)):
            return -np.inf

        dmu_dbeta, dmu_dlambda, dmu_dgamma = theta_e_ratio_derivatives(
            beta_geo,
            p["gamma_pl"],
            p["lambda_int"],
            model_mu,
        )

        c0 = p.get("beta_c0", 0.0)   # Floor or intercept
        c1 = p.get("beta_c1", 0.0)   # Slope
        c2 = p.get("beta_c2", 0.0)   # Quadratic extension (0 for standard linear)

        if self.scatter_model_type == "linear":
            sigma_beta_int_lens = c0 + c1 * self.dissimilarity + c2 * (self.dissimilarity**2)
        elif self.scatter_model_type == "quadrature":
            sigma_beta_int_lens = np.sqrt(c0 ** 2 + (c1 * self.dissimilarity) ** 2)
        elif self.scatter_model_type == "power_law":
            sigma_beta_int_lens = (10 ** c0) * (self.dissimilarity ** c1)
        else:
            raise ValueError(f"Unknown scatter_model_type: '{self.scatter_model_type}'")

        # Catch unphysical negative scatters in the linear model during MCMC exploration
        if np.any(sigma_beta_int_lens < 0):
            return -np.inf

        sigma_pop_sq = (
            (dmu_dlambda * p.get("lambda_sigma", 0.0)) ** 2
            + (dmu_dgamma * p.get("gamma_sigma", 0.0)) ** 2
            + (sigma_beta_int_lens * model_mu) ** 2
        )

        # Photo-z variance
        sigma_photoz_sq = 0.0
        if self.has_redshift_errors:
            sigma_photoz_sq = (
                (dmu_dbeta * dbeta_dzl  * self.err_zl)  ** 2
                + (dmu_dbeta * dbeta_dzs1 * self.err_zs1) ** 2
                + (dmu_dbeta * dbeta_dzs2 * self.err_zs2) ** 2
            )

        sigma_tot_model_sq = self.sigma_meas ** 2 + sigma_pop_sq + sigma_photoz_sq

        if self.is_asimov:
            sigma_tot_true_sq      = self.sigma_meas ** 2 + self.sigma_pop_true_sq
            expected_residuals_sq  = sigma_tot_true_sq + (self.beta_obs - model_mu) ** 2
            log_prob = -0.5 * np.sum(
                self.down_sampling
                * (expected_residuals_sq / sigma_tot_model_sq + np.log(sigma_tot_model_sq))
            )
        else:
            sigma_tot_binned_sq = sigma_tot_model_sq / self.down_sampling
            residuals_sq        = (self.beta_obs - model_mu) ** 2
            log_prob = -0.5 * np.sum(
                residuals_sq / sigma_tot_binned_sq + np.log(sigma_tot_binned_sq)
            )

        return -np.inf if np.isnan(log_prob) else log_prob

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)


def run_dspl_inference(
    kwargs_dspl_list,
    n_walkers=32,
    n_steps=1000,
    n_burn=200,
    initial_guess=None,
    initial_scatter=None,
    priors=None,
    fixed_params=None,
    backend_path=None,
    num_cpus=None,
    scatter_model_type="linear"
):
    """
    Run the emcee MCMC sampler for DSPL / PDSPL hierarchical inference.

    Parameters
    ----------
    kwargs_dspl_list : list of dict
        Mock data dicts from ``draw_lens_from_given_zs``.
    n_walkers, n_steps, n_burn : int
    initial_guess : dict, optional   parameter name → starting value
    initial_scatter : dict, optional  parameter name → initialisation spread
    priors : dict, optional           overrides for ``DSPLLikelihood.priors``
    fixed_params : dict, optional     parameters held fixed during sampling
    backend_path : str, optional      HDF5 file path for emcee backend
    num_cpus : int, optional

    Returns
    -------
    flat_samples : numpy.ndarray  shape (N_samples, N_params)
    labels : list of str           LaTeX labels for sampled parameters
    """
    if fixed_params is None:
        fixed_params = {}
    sampled_params = [p for p in ALL_PARAM_NAMES if p not in fixed_params]

    print("--- Starting DSPL Inference ---")
    print(f"Fixed Params  : {list(fixed_params.keys())}")
    print(f"Sampled Params: {sampled_params}")

    default_start = {
        "h0": 70.0, "om": 0.3, "w0": -1.0, "wa": 0.0,
        "lambda_int": 1.0, "lambda_sigma": 0.05,
        "gamma_pl": 2.0, "gamma_sigma": 0.1,
        "beta_c0": 0.15, "beta_c1": 0.9, "beta_c2": 0.0,
    }
    default_scatter = {k: 1e-3 for k in ALL_PARAM_NAMES}

    if initial_guess:
        default_start.update(initial_guess)
    if initial_scatter:
        default_scatter.update(initial_scatter)

    like = DSPLLikelihood(
        kwargs_dspl_list,
        sampled_params,
        fixed_params,
        priors=priors,
        scatter_model_type=scatter_model_type,
    )

    ndim = len(sampled_params)
    p0   = np.zeros((n_walkers, ndim))
    for w in range(n_walkers):
        valid = False
        while not valid:
            pos = np.zeros(ndim)
            for i, name in enumerate(sampled_params):
                mu  = default_start[name]
                sig = default_scatter[name]
                pos[i] = mu + sig * np.random.randn()
                if "sigma" in name or name == "beta_c0":
                    pos[i] = np.abs(pos[i])
            if np.isfinite(like.log_probability(pos)):
                p0[w] = pos
                valid  = True

    if backend_path is not None:
        backend = emcee.backends.HDFBackend(backend_path)
        backend.reset(n_walkers, ndim)
    else:
        backend = None

    if num_cpus is None:
        num_cpus = min(os.cpu_count() - 2 if os.cpu_count() else 1, 8)

    with Pool(processes=num_cpus) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, like.log_probability, pool=pool, backend=backend
        )
        sampler.run_mcmc(p0, n_steps, progress=True)

    flat_samples = sampler.get_chain(discard=n_burn, thin=1, flat=True)

    latex_labels = {
        "h0":           r"$H_0$",
        "om":           r"$\Omega_m$",
        "w0":           r"$w_0$",
        "wa":           r"$w_a$",
        "lambda_int":   r"$\bar{\lambda}_{\rm int}$",
        "lambda_sigma": r"$\sigma({\lambda}_{\rm int})$",
        "gamma_pl":     r"$\bar{\gamma}_{\rm pl}$",
        "gamma_sigma":  r"$\sigma({\gamma}_{\rm pl})$",
        "beta_c0":      r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(0)}$",
        "beta_c1":      r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(1)}$",
        "beta_c2":      r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}^{(2)}$",
    }
    # If only beta_c0 is sampled (no c1/c2), use a cleaner label
    if "beta_c1" not in sampled_params and "beta_c2" not in sampled_params:
        latex_labels["beta_c0"] = r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$"

    labels = [latex_labels[n] for n in sampled_params]
    return flat_samples, labels


def check_mcmc_convergence(scenarios_dict, param_labels, n_walkers=64):
    """
    Loops through a dictionary of forecast scenarios, unflattening the emcee 
    samples to calculate and print the integrated autocorrelation time (tau) 
    for convergence checks.

    Parameters
    ----------
    scenarios_dict : dict
        The dictionary containing scenario configurations and MCMC 'samples'.
    param_labels : list of str
        The list of LaTeX or text labels corresponding to the sampled parameters.
    n_walkers : int, optional
        The number of walkers used in the MCMC run. Default is 64.
    """
    print("=========================================================")
    print("               MCMC CONVERGENCE REPORT                   ")
    print("=========================================================")

    for key, sc in scenarios_dict.items():
        if "samples" not in sc or sc["samples"] is None:
            continue

        flat_samples = sc["samples"]
        ndim = flat_samples.shape[1]
        n_steps_kept = len(flat_samples) // n_walkers
        
        # Reshape back to [steps, walkers, parameters] for emcee
        chain = flat_samples.reshape((n_steps_kept, n_walkers, ndim))

        print(f"\nScenario: {sc['name']}")
        print(f"Post-burn-in steps per walker: {n_steps_kept}")
        print(f"{'Parameter':<25} | {'tau (steps)':<12} | {'Min Required':<15} | {'Status'}")
        print("-" * 75)

        try:
            # tol=0 forces emcee to return an estimate even if the chain is short
            tau = emcee.autocorr.integrated_time(chain, tol=0)
            max_tau = np.max(tau)

            for i, label in enumerate(param_labels):
                # Clean up LaTeX formatting for clean console logging
                clean_label = (label.replace('$', '')
                                    .replace('\\', '')
                                    .replace('rm ', '')
                                    .replace('{', '')
                                    .replace('}', ''))
                min_steps = int(50 * tau[i])
                status = "✓ OK" if n_steps_kept > min_steps else "! SHORT"
                print(f"{clean_label:<25} | {tau[i]:<12.1f} | {min_steps:<15} | {status}")

            if n_steps_kept > 50 * max_tau:
                print(f"--> OVERALL STATUS: ✓ CONVERGED")
            else:
                print(
                    "--> OVERALL STATUS: ! WARNING: Unconverged "
                    f"(Needs {int(50 * max_tau)} steps, has {n_steps_kept})"
                )
                
        except Exception as e:
            print(f"--> OVERALL STATUS: ERROR calculating tau: {e}")
