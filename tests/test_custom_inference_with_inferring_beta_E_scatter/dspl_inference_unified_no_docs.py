"""
dspl_inference.py
A complete, unified toolkit for Double Source Plane Lens (DSPL) cosmology.
Handles constant scatters, heteroscedastic scatters, and arbitrary fixed parameters seamlessly.
"""

import numpy as np
import emcee
from astropy.cosmology import Flatw0waCDM
from multiprocessing import Pool

# --- 1. PHYSICS & SIMULATION FUNCTIONS ---

def beta_double_source_plane(z_lens, z_source_1, z_source_2, cosmo):
    """
    Model prediction of ratio of scaled deflection angles.
    """
    z_lens = np.atleast_1d(z_lens)
    z_source_1 = np.atleast_1d(z_source_1)
    z_source_2 = np.atleast_1d(z_source_2)

    ds1 = cosmo.angular_diameter_distance(z_source_1).value
    ds2 = cosmo.angular_diameter_distance(z_source_2).value
    dds1 = cosmo.angular_diameter_distance_z1z2(z_lens, z_source_1).value
    dds2 = cosmo.angular_diameter_distance_z1z2(z_lens, z_source_2).value
    
    beta = (dds1 / ds1) * (ds2 / dds2)
    return beta

def beta2theta_e_ratio(beta_dsp, gamma_pl=2.0, lambda_mst=1.0):
    """
    Calculates Einstein radii ratio for a power-law + MST profile.
    """
    base_term = beta_dsp - (1 - lambda_mst) * (1 - beta_dsp)
    if np.isscalar(base_term):
        if base_term <= 0: return np.nan
    else:
        base_term[base_term <= 0] = np.nan
    return base_term ** (1.0 / (gamma_pl - 1.0))

def get_beta_z_derivatives(z_l, z_s1, z_s2, cosmo, dz=0.001):
    """
    Calculates d(beta)/dz using a numerical central finite difference.
    dz is a small step size for the numerical derivative.
    """
    # z_lens derivative
    beta_zl_up = beta_double_source_plane(z_l + dz, z_s1, z_s2, cosmo)
    beta_zl_dn = beta_double_source_plane(z_l - dz, z_s1, z_s2, cosmo)
    dbeta_dzl = (beta_zl_up - beta_zl_dn) / (2.0 * dz)
    
    # z_source_1 derivative
    beta_zs1_up = beta_double_source_plane(z_l, z_s1 + dz, z_s2, cosmo)
    beta_zs1_dn = beta_double_source_plane(z_l, z_s1 - dz, z_s2, cosmo)
    dbeta_dzs1 = (beta_zs1_up - beta_zs1_dn) / (2.0 * dz)
    
    # z_source_2 derivative
    beta_zs2_up = beta_double_source_plane(z_l, z_s1, z_s2 + dz, cosmo)
    beta_zs2_dn = beta_double_source_plane(z_l, z_s1, z_s2 - dz, cosmo)
    dbeta_dzs2 = (beta_zs2_up - beta_zs2_dn) / (2.0 * dz)
    
    return dbeta_dzl, dbeta_dzs1, dbeta_dzs2

def draw_lens_from_given_zs(z_lens, z1, z2, 
                            lambda_mst_mean, lambda_mst_sigma, 
                            gamma_pl_mean, gamma_pl_sigma, 
                            sigma_beta_intrinsic, sigma_meas_rel,
                            dissimilarity=0.0, 
                            cosmo=None, down_sampling=1, with_noise=True,
                            redshift_error_rel=0.0):
    
    beta_geo = beta_double_source_plane(z_lens, z1, z2, cosmo=cosmo)
    mu_model = beta2theta_e_ratio(beta_dsp=beta_geo, gamma_pl=gamma_pl_mean, lambda_mst=lambda_mst_mean)
    
    U = beta_geo - (1 - lambda_mst_mean) * (1 - beta_geo)
    dy_dlambda = mu_model * (1 - beta_geo) / ((gamma_pl_mean - 1.0) * U)
    dy_dgamma = - mu_model * np.log(U) / ((gamma_pl_mean - 1.0)**2)
    
    err_zl = redshift_error_rel * (1.0 + z_lens)
    err_zs1 = redshift_error_rel * (1.0 + z1)
    err_zs2 = redshift_error_rel * (1.0 + z2)
    
    if redshift_error_rel > 0:
        dbeta_dzl, dbeta_dzs1, dbeta_dzs2 = get_beta_z_derivatives(z_lens, z1, z2, cosmo)
        dmu_dbeta_fid = (mu_model / ((gamma_pl_mean - 1.0) * U)) * (2.0 - lambda_mst_mean)
        sigma_photoz_fid_sq = (dmu_dbeta_fid * dbeta_dzl * err_zl)**2 + \
                              (dmu_dbeta_fid * dbeta_dzs1 * err_zs1)**2 + \
                              (dmu_dbeta_fid * dbeta_dzs2 * err_zs2)**2
    else:
        dbeta_dzl, dbeta_dzs1, dbeta_dzs2 = 0.0, 0.0, 0.0
        sigma_photoz_fid_sq = 0.0
    
    # Calculate absolute variances combining ALL scatters (True Universe)
    sigma_pop_sq = (dy_dlambda * lambda_mst_sigma)**2 + \
                   (dy_dgamma * gamma_pl_sigma)**2 + \
                   (sigma_beta_intrinsic * mu_model)**2
                   
    sigma_meas_sq = (sigma_meas_rel * mu_model)**2 
    sigma_pop_true_sq = sigma_pop_sq + sigma_photoz_fid_sq
    
    if with_noise:
        sigma_tot_binned = np.sqrt((sigma_meas_sq + sigma_pop_true_sq) / down_sampling)
        beta_measured = np.random.normal(loc=mu_model, scale=sigma_tot_binned)
    else:
        beta_measured = mu_model

    return {
        "z_lens": z_lens, "z_source": z1, "z_source2": z2,
        "beta_dspl": beta_measured,
        "sigma_beta_dspl": np.sqrt(sigma_meas_sq), 
        "sigma_pop_true_sq": sigma_pop_true_sq,     
        "with_noise": with_noise,
        "dissimilarity": dissimilarity,               
        "err_zl": err_zl, "err_zs1": err_zs1, "err_zs2": err_zs2,
        "dbeta_dzl": dbeta_dzl, "dbeta_dzs1": dbeta_dzs1, "dbeta_dzs2": dbeta_dzs2
    }

# --- 2. INFERENCE FRAMEWORK ---

# This master list dictates what can potentially be sampled
ALL_PARAM_NAMES = [
    'h0', 'om', 'w0', 'wa', 
    'lambda_int', 'lambda_sigma', 
    'gamma_pl', 'gamma_sigma',
    'beta_c0', 'beta_c1', 'beta_c2' 
]

class DSPLLikelihood:
    def __init__(self, kwargs_likelihood_list, sampled_params, fixed_params, down_sampling=1, priors=None):
        self.sampled_params = sampled_params
        self.fixed_params = fixed_params
        self.down_sampling = down_sampling
        
        self.z_l = np.array([d['z_lens'] for d in kwargs_likelihood_list]).flatten()
        self.z_s1 = np.array([d['z_source'] for d in kwargs_likelihood_list]).flatten()
        self.z_s2 = np.array([d['z_source2'] for d in kwargs_likelihood_list]).flatten()
        self.beta_obs = np.array([d['beta_dspl'] for d in kwargs_likelihood_list]).flatten()
        
        self.sigma_meas = np.array([d['sigma_beta_dspl'] for d in kwargs_likelihood_list]).flatten()
        self.sigma_pop_true_sq = np.array([d.get('sigma_pop_true_sq', 0.0) for d in kwargs_likelihood_list]).flatten()
        self.dissimilarity = np.array([d.get('dissimilarity', 0.0) for d in kwargs_likelihood_list]).flatten()
        
        self.err_zl = np.array([d.get('err_zl', 0.0) for d in kwargs_likelihood_list]).flatten()
        self.err_zs1 = np.array([d.get('err_zs1', 0.0) for d in kwargs_likelihood_list]).flatten()
        self.err_zs2 = np.array([d.get('err_zs2', 0.0) for d in kwargs_likelihood_list]).flatten()
        self.dbeta_dzl = np.array([d.get('dbeta_dzl', 0.0) for d in kwargs_likelihood_list]).flatten()
        self.dbeta_dzs1 = np.array([d.get('dbeta_dzs1', 0.0) for d in kwargs_likelihood_list]).flatten()
        self.dbeta_dzs2 = np.array([d.get('dbeta_dzs2', 0.0) for d in kwargs_likelihood_list]).flatten()
        
        with_noise_flags = [d.get('with_noise', True) for d in kwargs_likelihood_list]
        self.is_asimov = not any(with_noise_flags)
        
        self.priors = {
            'h0': ('uniform', 50.0, 90.0),
            'om': ('uniform', 0.0, 1.0),
            'w0': ('uniform', -3.0, 0.0),
            'wa': ('uniform', -5.0, 5.0),
            'lambda_int': ('uniform', 0.8, 1.2),
            'lambda_sigma': ('uniform', 0.0, 0.2),
            'gamma_pl': ('uniform', 1.0, 3.0),
            'gamma_sigma': ('uniform', 0.0, 0.5),
            'beta_c0': ('uniform', -1.0, 1.0),
            'beta_c1': ('uniform', -5.0, 5.0), 
            'beta_c2': ('uniform', -5.0, 5.0)  
        }
        if priors is not None:
            self.priors.update(priors)

    def _get_full_params(self, theta):
        params = self.fixed_params.copy()
        for name, val in zip(self.sampled_params, theta):
            params[name] = val
        return params

    def get_derivatives(self, beta_geo, lambda_int, gamma_pl, model_val):
        with np.errstate(invalid='ignore', divide='ignore'):
            U = beta_geo - (1 - lambda_int) * (1 - beta_geo)
            if np.any(U <= 0): return np.nan, np.nan
            dy_dlambda = model_val * (1 - beta_geo) / ((gamma_pl - 1.0) * U)
            dy_dgamma = - model_val * np.log(U) / ((gamma_pl - 1.0)**2)
        return dy_dlambda, dy_dgamma

    def log_prior(self, theta):
        lp = 0.0
        for i, name in enumerate(self.sampled_params):
            val = theta[i]
            prior_type, p1, p2 = self.priors[name]
            if prior_type == 'uniform':
                if not (p1 < val < p2): return -np.inf
            elif prior_type == 'gaussian':
                lp += -0.5 * ((val - p1)**2 / p2**2) - np.log(p2)
        return lp

    def log_likelihood(self, theta):
        p = self._get_full_params(theta)
        
        try:
            cosmo_step = Flatw0waCDM(H0=p['h0'], Om0=p['om'], w0=p['w0'], wa=p['wa'])
            beta_geo = beta_double_source_plane(self.z_l, self.z_s1, self.z_s2, cosmo_step)
        except Exception:
            return -np.inf 

        model_mu = beta2theta_e_ratio(beta_geo, gamma_pl=p['gamma_pl'], lambda_mst=p['lambda_int'])
        if np.any(np.isnan(model_mu)): return -np.inf

        # 1. Unified Variance Components
        dy_dlam, dy_dgam = self.get_derivatives(beta_geo, p['lambda_int'], p['gamma_pl'], model_mu)
        
        # Build the intrinsic scatter per-lens 
        # Safely defaults to 0.0 if not sampled AND not fixed
        c0 = p.get('beta_c0', 0.0)
        c1 = p.get('beta_c1', 0.0)
        c2 = p.get('beta_c2', 0.0)
        sigma_beta_int_lens = c0 + c1 * self.dissimilarity + c2 * (self.dissimilarity**2)
        
        # Uses whichever values are provided (sampled or fixed), defaults to 0.0 otherwise
        sigma_pop_sq = (dy_dlam * p.get('lambda_sigma', 0.0))**2 + \
                       (dy_dgam * p.get('gamma_sigma', 0.0))**2 + \
                       (sigma_beta_int_lens * model_mu)**2
            
        # 2. Photo-Z Variance
        sigma_photoz_sq = 0
        if np.any(self.err_zl > 0):
            U = beta_geo - (1 - p['lambda_int']) * (1 - beta_geo)
            dmu_dbeta = (model_mu / ((p['gamma_pl'] - 1.0) * U)) * (2.0 - p['lambda_int'])
            
            sigma_photoz_sq = (dmu_dbeta * self.dbeta_dzl * self.err_zl)**2 + \
                              (dmu_dbeta * self.dbeta_dzs1 * self.err_zs1)**2 + \
                              (dmu_dbeta * self.dbeta_dzs2 * self.err_zs2)**2
            
        sigma_tot_model_sq = self.sigma_meas**2 + sigma_pop_sq + sigma_photoz_sq

        if self.is_asimov:
            sigma_tot_true_sq = self.sigma_meas**2 + self.sigma_pop_true_sq
            expected_residuals_sq = sigma_tot_true_sq + (self.beta_obs - model_mu)**2
            log_prob = -0.5 * self.down_sampling * np.sum( (expected_residuals_sq / sigma_tot_model_sq) + np.log(sigma_tot_model_sq) )
        else:
            sigma_tot_binned_sq = sigma_tot_model_sq / self.down_sampling
            residuals_sq = (self.beta_obs - model_mu)**2
            log_prob = -0.5 * np.sum( (residuals_sq / sigma_tot_binned_sq) + np.log(sigma_tot_binned_sq) )
        
        if np.isnan(log_prob): return -np.inf
        return log_prob

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        ll = self.log_likelihood(theta)
        return lp + ll


def run_dspl_inference(kwargs_dspl_list, down_sampling=1, n_walkers=32, n_steps=1000, n_burn=200, 
                       initial_guess=None, initial_scatter=None, priors=None, 
                       fixed_params=None, backend_path=None):
    
    if fixed_params is None: fixed_params = {}
    sampled_params = [p for p in ALL_PARAM_NAMES if p not in fixed_params]
    
    print(f"--- Starting DSPL Inference ---")
    print(f"Fixed Params  : {list(fixed_params.keys())}")
    print(f"Sampled Params: {sampled_params}")

    default_start = {
        'h0': 70.0, 'om': 0.3, 'w0': -1.0, 'wa': 0.0,
        'lambda_int': 1.0, 'lambda_sigma': 0.05,
        'gamma_pl': 2.0, 'gamma_sigma': 0.1,
        'beta_c0': 0.02, 'beta_c1': 0.0, 'beta_c2': 0.0
    }
    default_scatter = {k: 1e-3 for k in ALL_PARAM_NAMES}

    if initial_guess: default_start.update(initial_guess)
    if initial_scatter: default_scatter.update(initial_scatter)

    like = DSPLLikelihood(kwargs_dspl_list, sampled_params, fixed_params, down_sampling=down_sampling, priors=priors)

    ndim = len(sampled_params)
    p0 = np.zeros((n_walkers, ndim))
    for w in range(n_walkers):
        valid = False
        while not valid:
            pos = np.zeros(ndim)
            for i, name in enumerate(sampled_params):
                mu = default_start[name]
                sig = default_scatter[name]
                pos[i] = mu + sig * np.random.randn()
                if 'sigma' in name: pos[i] = np.abs(pos[i])
            if np.isfinite(like.log_probability(pos)):
                p0[w] = pos
                valid = True

    if backend_path is not None:
        backend = emcee.backends.HDFBackend(backend_path)
        backend.reset(n_walkers, ndim)
    else:
        backend = None

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(n_walkers, ndim, like.log_probability, pool=pool, backend=backend)
        sampler.run_mcmc(p0, n_steps, progress=True)

    flat_samples = sampler.get_chain(discard=n_burn, thin=1, flat=True)
    
    latex_labels = {
        'h0': r"$H_0$", 'om': r"$\Omega_m$", 'w0': r"$w_0$", 'wa': r"$w_a$",
        'lambda_int': r"$\bar{\lambda}_{\rm int}$", 'lambda_sigma': r"$\sigma({\lambda}_{\rm int})$",
        'gamma_pl': r"$\bar{\gamma}_{\rm pl}$", 'gamma_sigma': r"$\sigma({\gamma}_{\rm pl})$",
        'beta_c0': r"$c_0$", 'beta_c1': r"$c_1$", 'beta_c2': r"$c_2$"
    }

    # Dynamic Label Logic: If we are ONLY inferring c0, switch the label to the generalized sigma
    if 'beta_c1' not in sampled_params and 'beta_c2' not in sampled_params:
        latex_labels['beta_c0'] = r"$\sigma_{\beta_{\rm E},\rm \mathcal{D}}$"

    labels = [latex_labels[n] for n in sampled_params]
    
    return flat_samples, labels