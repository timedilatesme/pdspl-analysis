import numpy as np
from astropy.cosmology import Flatw0waCDM
from astropy.cosmology import Cosmology
import astropy.units as u
import emcee
from multiprocessing import Pool, cpu_count


def beta_double_source_plane(z_lens, z_source_1, z_source_2, cosmo: Cosmology):
    """
    Model prediction of ratio of scaled deflection angles.

    :param z_lens: lens redshift
    :param z_source_1: source_1 redshift
    :param z_source_2: source_2 redshift
    :param cosmo: ~astropy.cosmology instance
    :return: beta
    """
    # Ensure z_lens < z_source_1 and z_lens < z_source_2
    if np.any(z_lens >= z_source_1) or np.any(z_lens >= z_source_2):
        raise ValueError("z_lens must be less than both z_source_1 and z_source_2")
    
    D_ds1 = cosmo.angular_diameter_distance_z1z2(z_lens, z_source_1)
    D_s1 = cosmo.angular_diameter_distance(z_source_1)
    D_ds2 = cosmo.angular_diameter_distance_z1z2(z_lens, z_source_2)
    D_s2 = cosmo.angular_diameter_distance(z_source_2)

    alpha1 = (D_ds1 / D_s1).to_value(u.dimensionless_unscaled)
    alpha2 = (D_ds2 / D_s2).to_value(u.dimensionless_unscaled)

    beta = alpha1 / alpha2

    return beta

def beta2theta_e_ratio(beta_dsp, gamma_pl=2, lambda_mst=1):
    """Convert scaled deflection angles to Einstein radius ratio.

    :param beta_dsp: scaled deflection angles alpha_1 / alpha_2 as ratio between
        z_source and z_source2 source planes
    :param gamma_pl: power-law density slope of main deflector (=2 being isothermal)
    :param lambda_mst: mass-sheet transform at the main deflector
    :return: theta_E_ratio
    """

    theta_E_ratio = (beta_dsp * (2 - lambda_mst) - (1 - lambda_mst)) ** (1 / (gamma_pl - 1))
    return theta_E_ratio

def log_likelihood(theta, zd_arr, zs1_arr, zs2_arr, beta_E_obs_arr, beta_E_obs_err_arr, H0=70.0, 
                   normalized=True):
    """
    Calculates the vectorized log-likelihood for the entire dataset.
    theta: Model parameters [Omega_m, w0, wa, lambda_MST, gamma_pl]
    zd_arr, zs1_arr, zs2_arr: Arrays of redshifts for deflector, source1, source2
    beta_E_obs_arr: Array of observed beta_E values
    beta_E_obs_err_arr: Array of errors on observed beta_E values
    H0: Hubble constant, default is 70.0 km/s/Mpc
    """

    Omega_m, w0, wa, lambda_MST, gamma_pl = theta

    # check gamma_pl != 1 to avoid division by zero
    if np.isclose(gamma_pl, 1):
        return -np.inf

    try:
        # 1. Create cosmology object ONCE per theta evaluation
        cosmo = Flatw0waCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Omega_m, w0=w0, wa=wa)

        # 2. beta_dspl calculation
        beta_arr = beta_double_source_plane(zd_arr, zs1_arr, zs2_arr, cosmo)
        
        # 3. model einstein radius ratio
        beta_E_model_arr = beta2theta_e_ratio(beta_arr, gamma_pl=gamma_pl, lambda_mst=lambda_MST)

        # 4. Likelihood calculation
        log_l = -0.5 * ((beta_E_model_arr - beta_E_obs_arr) / beta_E_obs_err_arr) ** 2
        if normalized:
            log_l -= 1 / 2.0 * np.log(2 * np.pi * beta_E_obs_err_arr**2)
        
        # 5. sum log-likelihood for all data points
        log_l = np.sum(log_l)

        return log_l
    except Exception as e:
        # Catch astropy errors for extreme cosmological parameters or other issues
        # print(f"Warning: Log-likelihood calculation error for theta={theta}: {e}") # For debugging
        return -np.inf

def log_prior(theta):
    """
    Calculates the log-prior for the parameters.
    theta: Model parameters [Omega_m, w0, wa, lambda_MST, gamma_pl]
    """
    Omega_m, w0, wa, lambda_MST, gamma_pl = theta

    # Define flat priors (log prior = 0 if in range, -inf otherwise)
    if (0 < Omega_m < 1.0 and           # Omega_m
        -2.0 < w0 < 0 and               # w0 (present-day dark energy equation of state)
        -3.0 < wa < 3.0 and             # wa (time evolution of dark energy equation of state)
        0.0 < lambda_MST < 2.0 and      # lambda_MST (model specific)
        1 < gamma_pl < 3):              # gamma_pl (model specific, >1 to avoid issues)
        return 0.0
    return -np.inf


def log_probability(theta, zd_arr, zs1_arr, zs2_arr, beta_E_obs_arr, beta_E_err_arr):
    """
    Calculates the total log-posterior probability.
    theta: Model parameters [Omega_m, w0, wa, lambda_MST, gamma_pl]
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    # H0 is fixed here, but could be part of theta if fitting for it.
    ll = log_likelihood(theta, zd_arr, zs1_arr, zs2_arr, beta_E_obs_arr, beta_E_err_arr, H0=70.0)
    
    # ll will be -np.inf if model calculation fails or likelihood is invalid
    if not np.isfinite(ll):
        return -np.inf
        
    return lp + ll

def run_mcmc_analysis(z_lens_arr, z1_arr, z2_arr, beta_E_obs_arr, beta_E_obs_err_arr,
                      kwargs_means=None, kwargs_spreads=None, nwalkers=400, nsteps=1000):
    """Run MCMC analysis on the data.
    
    :param z_lens_arr: Array of lens redshifts
    :param z1_arr: Array of source1 redshifts
    :param z2_arr: Array of source2 redshifts
    :param beta_E_obs_arr: Array of observed beta_E values
    :param beta_E_obs_err_arr: Array of errors on observed beta_E values
    :param kwargs_means: Dictionary of means for initial walker positions. Must contain keys:
        'Omega_m', 'w0', 'wa', 'lambda_MST', 'gamma_pl'
    :param kwargs_spreads: Dictionary of spreads for initial walker positions. Must contain keys:
        'Omega_m', 'w0', 'wa', 'lambda_MST', 'gamma_pl'
    :param nwalkers: Number of walkers (default is 400)
    :param nsteps: Number of steps to run (default is 1000)
    :return: MCMC sampler object
    """
    
    # Initialize Walkers [PARAMS: Omega_m, w0, wa, lambda_MST, gamma_pl]
    # initial_guess_means = np.array([0.3, -1.0, 0.0, 1.0, 2.0])
    # initial_spreads = np.array([0.05, 0.2, 0.3, 0.01, 0.01])
    if kwargs_means is None:
        initial_guess_means = np.array([0.3, -1.0, 0.0, 1.0, 2.0])
    else:
        initial_guess_means = np.array([kwargs_means['Omega_m'], kwargs_means['w0'], 
                                        kwargs_means['wa'], kwargs_means['lambda_MST'], 
                                        kwargs_means['gamma_pl']])
    if kwargs_spreads is None:
        initial_spreads = np.array([0.05, 0.2, 0.3, 0.01, 0.01])
    else:
        initial_spreads = np.array([kwargs_spreads['Omega_m'], kwargs_spreads['w0'], 
                                    kwargs_spreads['wa'], kwargs_spreads['lambda_MST'], 
                                    kwargs_spreads['gamma_pl']])
    
    ndim = len(initial_guess_means)  # Number of parameters

    pos_initial = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        while True:
            p = initial_guess_means + initial_spreads * np.random.randn(ndim)
            if np.isfinite(log_prior(p)):
                if np.all(np.isfinite(log_likelihood(p, z_lens_arr, z1_arr, z2_arr, beta_E_obs_arr, beta_E_obs_err_arr))):
                    pos_initial[i,:] = p
                    break
    
    # Run MCMC
    sampler_args = (z_lens_arr, z1_arr, z2_arr, beta_E_obs_arr, beta_E_obs_err_arr)
    
    with Pool(processes=cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=sampler_args,
            pool=pool
        )

        print(f"Starting MCMC run: {nwalkers} walkers, {nsteps} steps, {cpu_count()} cores...")
        print(f"Parameters: Omega_m, w0, wa, lambda_MST, gamma_pl")
        sampler.run_mcmc(pos_initial, nsteps, progress=True)
        print("MCMC run finished!")
    
    return sampler