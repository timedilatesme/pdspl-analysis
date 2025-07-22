import numpy as np
from astropy.cosmology import FlatwCDM
from astropy.cosmology import Cosmology
import astropy.units as u


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
    theta: Model parameters [Omega_m, w, lambda_MST, gamma_pl]
    zd_arr, zs1_arr, zs2_arr: Arrays of redshifts for deflector, source1, source2
    beta_E_obs_arr: Array of observed beta_E values
    beta_E_obs_err_arr: Array of errors on observed beta_E values
    H0: Hubble constant, default is 70.0 km/s/Mpc
    """

    Omega_m, w, lambda_MST, gamma_pl = theta

    # check gamma_pl != 1 to avoid division by zero
    if np.isclose(gamma_pl, 1):
        return -np.inf

    try:
        # 1. Create cosmology object ONCE per theta evaluation
        cosmo = FlatwCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Omega_m, w0=w)

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
    theta: Model parameters [Omega_m, w, lambda_MST, gamma_pl]
    """
    Omega_m, w, lambda_MST, gamma_pl = theta

    # Define flat priors (log prior = 0 if in range, -inf otherwise)
    if (0 < Omega_m < 1.0 and   # Omega_m
        -2.0 < w < 0 and        # w (dark energy equation of state)
        0.0 < lambda_MST < 2.0 and # lambda_MST (model specific)
        1 < gamma_pl < 3):    # gamma_pl (model specific, >1 to avoid issues)
        return 0.0
    return -np.inf


def log_probability(theta, zd_arr, zs1_arr, zs2_arr, beta_E_obs_arr, beta_E_err_arr):
    """
    Calculates the total log-posterior probability.
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