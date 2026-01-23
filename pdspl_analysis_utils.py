from astropy.table import Table
import numpy as np
from tqdm import tqdm
from itertools import combinations
# from hierarc.Likelihood.LensLikelihood.double_source_plane import beta2theta_e_ratio, beta_double_source_plane
# from lenstronomy.LensModel.lens_model import LensModel
from scipy import spatial

#############################################################################
# PDSPL UTILITIES
#############################################################################
def beta_double_source_plane(z_lens, z_source_1, z_source_2, cosmo):
    """Model prediction of ratio of scaled deflection angles.

    :param z_lens: lens redshift
    :param z_source_1: source_1 redshift
    :param z_source_2: source_2 redshift
    :param cosmo: ~astropy.cosmology instance
    :return: beta
    """
    ds1 = cosmo.angular_diameter_distance(z_source_1).value
    dds1 = cosmo.angular_diameter_distance_z1z2(z_lens, z_source_1).value
    ds2 = cosmo.angular_diameter_distance(z_source_2).value
    dds2 = cosmo.angular_diameter_distance_z1z2(z_lens, z_source_2).value
    beta = dds1 / ds1 * ds2 / dds2
    return beta


def beta2theta_e_ratio(beta_dsp, gamma_pl=2, lambda_mst=1):
    """Calculates Einstein radii ratio for a power-law + MST profile with given
    parameters.

    :param beta_dsp: scaled deflection angles alpha_1 / alpha_2 as ratio between
        z_source and z_source2 source planes
    :param gamma_pl: power-law density slope of main deflector (=2 being isothermal)
    :param lambda_mst: mass-sheet transform at the main deflector
    :return: theta_E1 / theta_E2
    """
    return (beta_dsp - (1 - lambda_mst) * (1 - beta_dsp)) ** (1 / (gamma_pl - 1))

def draw_lens_from_given_zs(z_lens, z1, z2, 
                            lambda_mst_mean, lambda_mst_sigma, gamma_pl_mean, gamma_pl_sigma, 
                            sigma_beta, cosmo,
                            down_sampling=1, with_noise=False):
    """
    draw the likelihood object of a double source plane lens

    :param z_lens: redshift of the lens
    :param z1: redshift of the first source plane
    :param z2: redshift of the second source plane
    :param lambda_mst_mean: mean value of the mass-sheet transformation parameter
    :param lambda_mst_sigma: standard deviation of the mass-sheet transformation parameter
    :param gamma_pl_mean: mean value of the power-law slope of the lensing potential
    :param gamma_pl_sigma: standard deviation of the power-law slope of the lensing potential
    :param sigma_beta: relative precision on Einstein radius, used to compute the noise on the measured beta
    :param down_sampling: downsampling factor, noise will be reduced by sqrt(down_sampling)
    :param with_noise: if True, add noise to the measured beta 
    """
    beta = beta_double_source_plane(z_lens, z1, z2, cosmo=cosmo)
    
    beta_e_list = []
    beta_e_mean = beta2theta_e_ratio(beta_dsp=beta, gamma_pl=gamma_pl_mean, lambda_mst=lambda_mst_mean)
    for i in range(100):
        lambda_mst = np.random.normal(lambda_mst_mean, lambda_mst_sigma)
        gamma_pl = np.random.normal(gamma_pl_mean, gamma_pl_sigma)
        beta_e_ = beta2theta_e_ratio(beta_dsp=beta, gamma_pl=gamma_pl, lambda_mst=lambda_mst)
        beta_e_list.append(beta_e_)
    beta_e_list = np.array(beta_e_list)
    beta_e_mean_ = np.mean(beta_e_list)
    beta_e_sigma = np.sqrt(np.std(beta_e_list)**2 + (sigma_beta * beta_e_mean)**2) / np.sqrt(down_sampling)

    if with_noise:
        beta_measured = beta_e_mean + np.random.normal(loc=0, scale=beta_e_sigma)
    else:
        beta_measured = beta_e_mean

    kwargs_likelihood = {
        "z_lens": z_lens,
        "z_source": z1,
        "z_source2": z2,
        "beta_dspl": beta_measured,
        "sigma_beta_dspl": beta_e_sigma,
        "likelihood_type": "DSPL",
    }
    return kwargs_likelihood
#############################################################################

#############################################################################
# PLANE FITTING AND SCATTER
#############################################################################
# Fit plane to the data
def fit_plane(x, y, z):
    """Fit a plane to the data points (x, y, z). The plane is z = ax + by + c."""

    # don't use nans or infs
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]

    A = np.c_[x, y, np.ones_like(x)]
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs

# Find the scatter of the data points from the fitted plane
def find_scatter(x, y, z, coeffs, return_fit=False):
    """Find the scatter of the data points from the fitted plane. Return ``z - z_fit``."""

    # don't use nans or infs
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]

    z_fit = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    scatter = z - z_fit
    if return_fit:
        return scatter, z_fit
    else:
        return scatter
############################################################################

############################################################################
# PAIRING SIMULATION
############################################################################
def normalize_data(data, type='minmax', data_min=None, data_max=None):
    """Normalize the data to the range [0, 1].
    Parameters
    ----------
    data : array-like
        The input data to be normalized.
    type : str, optional
        The type of normalization to be applied. Default is 'minmax'.
        Options are:
        - 'minmax': Min-Max normalization to the range [0, 1].
        - 'zscore': Z-score normalization (mean=0, std=1).
    data_min : float, optional
        The minimum value of the data for 'minmax' normalization. If None, it is
        computed from the data.
    data_max : float, optional
        The maximum value of the data for 'minmax' normalization. If None, it is
        computed from the data.
    Returns
    -------
    normalized_data : array-like
        The normalized data.
    """
    data = np.array(data)
    if type == 'minmax':
        if data_min is None:
            data_min = np.nanmin(data)
        if data_max is None:
            data_max = np.nanmax(data)
        normalized_data = (data - data_min) / (data_max - data_min)
    elif type == 'zscore':
        mean = np.nanmean(data)
        std = np.nanstd(data)
        normalized_data = (data - mean) / std
    else:
        raise ValueError("Unsupported normalization type. Use 'minmax' or 'zscore'.")
    return normalized_data


def kdtree_matching_n_dim(points, n_neighbors=2, unique_pairs=True):
    """Find the nearest neighbors in n-dimensional space using a KDTree.

    Parameters
    ----------
    points : array-like, shape (n_samples, n_features)
        The input points to build the KDTree.
    n_neighbors : int, optional
        The number of nearest neighbors to find. Default is 2. Note that if it is set to 1, the self-match will be returned.
    unique_pairs : bool, optional
        If True, only unique pairs are returned. Default is True.

    Returns
    -------
    indices : array, shape (n_samples, n_neighbors)
        Indices of the nearest neighbors for each point.
    distances : array, shape (n_samples, n_neighbors)
        Distances to the nearest neighbors for each point.
    """
    tree = spatial.KDTree(points)
    distances, indices = tree.query(points, k=n_neighbors)

    distances = distances[:,1] # exclude self-distance (0th column)

    if unique_pairs:
        # Sort each pair to ensure (i, j) and (j, i) are treated the same
        for i in range(indices.shape[0]):
            if indices[i][0] > indices[i][1]:
                indices[i] = indices[i][::-1] # reverse the order
        
        # only return unique pairs
        indices, unique_pairs_idxs = np.unique(indices, axis=0, return_index=True)
        distances = distances[unique_pairs_idxs]

    return indices, distances

def get_pairs_table_PDSPL(data_table, pair_indices, cosmo, progress_bar=True):
    """Get the pairs table for the PDSPL model.

    Parameters
    ----------
    data_table : astropy.table.Table
        The input data table containing the lens and source properties.
    pair_indices : array, shape (n_pairs, 2)
        Indices of the paired lenses.
    cosmo : astropy.cosmology.Cosmology
        The cosmology to use for lens distance calculations.
    """
    pairs_table = {
        "index_1": [],
        "index_2": [],
        "z_D1": [],
        "z_D2": [],
        "z_D": [],
        "z_S1": [],
        "z_S2": [],
        "theta_E1": [],
        "theta_E2": [],
        "beta_E_DSPL": [],
        "beta_E_pseudo": [],
        "sigma_v_D1": [],
        "sigma_v_D2": [],
        "R_e_kpc_D1": [],
        "R_e_kpc_D2": [],
        "R_e_arcsec_D1": [],
        "R_e_arcsec_D2": [],
        "Sigma_half_Msun/pc2_D1": [],
        "Sigma_half_Msun/pc2_D2": [],
        "gamma_pl_1": [],
        "gamma_pl_2": [],
        "color_D_gr_1": [],
        "color_D_gr_2": [],
        "color_D_ri_1": [],
        "color_D_ri_2": [],
        "mag_D_i_1": [],
        "mag_D_i_2": [],
    }

    if "err_z_D" in data_table.colnames:
        pairs_table["err_z_D1"] = []
        pairs_table["err_z_D2"] = []
    if "err_sigma_v_D" in data_table.colnames:
        pairs_table["err_sigma_v_D1"] = []
        pairs_table["err_sigma_v_D2"] = []
    if "err_R_e_arcsec" in data_table.colnames:
        pairs_table["err_R_e_arcsec_D1"] = []
        pairs_table["err_R_e_arcsec_D2"] = []

    if progress_bar:
        iterator = tqdm(enumerate(pair_indices), total=len(pair_indices), desc="Processing pairs")
    else:
        iterator = enumerate(pair_indices)

    for i, (idx1, idx2) in iterator:

        # z_S1 should be less than z_S2
        if data_table[idx1]["z_S"] > data_table[idx2]["z_S"]:
            idx1, idx2 = idx2, idx1

        pairs_table["index_1"].append(idx1)
        pairs_table["index_2"].append(idx2)
        pairs_table["z_D1"].append(data_table[idx1]["z_D"])
        pairs_table["z_D2"].append(data_table[idx2]["z_D"])
        pairs_table["z_D"].append(0.5 * (data_table[idx1]["z_D"] + data_table[idx2]["z_D"]))
        pairs_table["z_S1"].append(data_table[idx1]["z_S"])
        pairs_table["z_S2"].append(data_table[idx2]["z_S"])
        pairs_table["theta_E1"].append(data_table[idx1]["theta_E"])
        pairs_table["theta_E2"].append(data_table[idx2]["theta_E"])
        pairs_table["sigma_v_D1"].append(data_table[idx1]["sigma_v_D"])
        pairs_table["sigma_v_D2"].append(data_table[idx2]["sigma_v_D"])
        pairs_table["R_e_kpc_D1"].append(data_table[idx1]["R_e_kpc"])
        pairs_table["R_e_kpc_D2"].append(data_table[idx2]["R_e_kpc"])
        pairs_table["R_e_arcsec_D1"].append(data_table[idx1]["R_e_arcsec"])
        pairs_table["R_e_arcsec_D2"].append(data_table[idx2]["R_e_arcsec"])
        pairs_table["Sigma_half_Msun/pc2_D1"].append(data_table[idx1]["Sigma_half_Msun/pc2"])
        pairs_table["Sigma_half_Msun/pc2_D2"].append(data_table[idx2]["Sigma_half_Msun/pc2"])
        pairs_table["beta_E_pseudo"].append(data_table[idx1]["theta_E"] / data_table[idx2]["theta_E"])
        pairs_table["gamma_pl_1"].append(data_table[idx1]["gamma_pl"])
        pairs_table["gamma_pl_2"].append(data_table[idx2]["gamma_pl"])

        # errors on z_D, sigma_v_D, R_e_arcsec, mags
        if "err_z_D" in data_table.colnames:
            pairs_table["err_z_D1"].append(data_table[idx1]["err_z_D"])
            pairs_table["err_z_D2"].append(data_table[idx2]["err_z_D"])
        if "err_sigma_v_D" in data_table.colnames:
            pairs_table["err_sigma_v_D1"].append(data_table[idx1]["err_sigma_v_D"])
            pairs_table["err_sigma_v_D2"].append(data_table[idx2]["err_sigma_v_D"])
        if "err_R_e_arcsec" in data_table.colnames:
            pairs_table["err_R_e_arcsec_D1"].append(data_table[idx1]["err_R_e_arcsec"])
            pairs_table["err_R_e_arcsec_D2"].append(data_table[idx2]["err_R_e_arcsec"])

        # calculate beta_E_DSPL
        _beta_E_DSPL_D1 = beta_double_source_plane(
            z_lens = data_table[idx1]["z_D"],
            z_source_1 = data_table[idx1]["z_S"],
            z_source_2 = data_table[idx2]["z_S"],
            cosmo = cosmo,
        )
        _beta_E_DSPL_D2 = beta_double_source_plane(
            z_lens = data_table[idx2]["z_D"],
            z_source_1 = data_table[idx1]["z_S"],
            z_source_2 = data_table[idx2]["z_S"],
            cosmo = cosmo,
        )
        beta_E_DSPL_D1 = beta2theta_e_ratio(_beta_E_DSPL_D1, gamma_pl=data_table[idx1]["gamma_pl"], lambda_mst=1)
        beta_E_DSPL_D2 = beta2theta_e_ratio(_beta_E_DSPL_D2, gamma_pl=data_table[idx2]["gamma_pl"], lambda_mst=1)
        pairs_table["beta_E_DSPL"].append(0.5 * (beta_E_DSPL_D1 + beta_E_DSPL_D2))

        # add color_D_gr and color_D_ri
        color_D_gr_1 = data_table[idx1]['mag_D_g'] - data_table[idx1]['mag_D_r']
        color_D_gr_2 = data_table[idx2]['mag_D_g'] - data_table[idx2]['mag_D_r']
        color_D_ri_1 = data_table[idx1]['mag_D_r'] - data_table[idx1]['mag_D_i']
        color_D_ri_2 = data_table[idx2]['mag_D_r'] - data_table[idx2]['mag_D_i']
        pairs_table["color_D_gr_1"].append(color_D_gr_1)
        pairs_table["color_D_gr_2"].append(color_D_gr_2)
        pairs_table["color_D_ri_1"].append(color_D_ri_1)
        pairs_table["color_D_ri_2"].append(color_D_ri_2)

        pairs_table["mag_D_i_1"].append(data_table[idx1]['mag_D_i'])
        pairs_table["mag_D_i_2"].append(data_table[idx2]['mag_D_i'])

    # make it an astropy table
    pairs_table = Table(pairs_table)

    # add a column for the fractional difference between beta_E_DSPL and beta_E_pseudo
    pairs_table["rel_diff_beta_E"] = (
        1 - pairs_table['beta_E_pseudo'] / pairs_table['beta_E_DSPL']
    )

    # add a column for the rel difference between sigma_v of the two lenses
    pairs_table["rel_diff_sigma_v_D"] = 2*(pairs_table["sigma_v_D2"] - pairs_table["sigma_v_D1"]) / (pairs_table["sigma_v_D2"] + pairs_table["sigma_v_D1"])

    # add a column for the rel difference between R_e of the two lenses
    pairs_table["rel_diff_R_e_kpc"] = 2*(pairs_table["R_e_kpc_D2"] - pairs_table["R_e_kpc_D1"]) / (pairs_table["R_e_kpc_D2"] + pairs_table["R_e_kpc_D1"])
    pairs_table["rel_diff_R_e_arcsec"] = 2*(pairs_table["R_e_arcsec_D2"] - pairs_table["R_e_arcsec_D1"]) / (pairs_table["R_e_arcsec_D2"] + pairs_table["R_e_arcsec_D1"])

    # add a column for the rel difference between Sigma_half of the two lenses
    pairs_table["rel_diff_Sigma_half"] = 2*(pairs_table["Sigma_half_Msun/pc2_D2"] - pairs_table["Sigma_half_Msun/pc2_D1"]) / (pairs_table["Sigma_half_Msun/pc2_D2"] + pairs_table["Sigma_half_Msun/pc2_D1"])

    # add a column for i band magnitude difference between the two lenses
    pairs_table['rel_diff_mag_D_i'] = 2*(pairs_table['mag_D_i_2'] - pairs_table['mag_D_i_1']) / (pairs_table['mag_D_i_2'] + pairs_table['mag_D_i_1'])

    # relative difference in colors between the two deflectors
    pairs_table['rel_diff_color_D_gr'] = 2*(pairs_table['color_D_gr_2'] - pairs_table['color_D_gr_1']) / (pairs_table['color_D_gr_2'] + pairs_table['color_D_gr_1'])
    pairs_table['rel_diff_color_D_ri'] = 2*(pairs_table['color_D_ri_2'] - pairs_table['color_D_ri_1']) / (pairs_table['color_D_ri_2'] + pairs_table['color_D_ri_1'])

    # add a column for the rel difference between z_D of the two lenses
    pairs_table["rel_diff_z_D"] = 2*(pairs_table["z_D2"] - pairs_table["z_D1"]) / (pairs_table["z_D2"] + pairs_table["z_D1"])

    # add a column for the rel difference between gamma_pl of the two lenses
    pairs_table["rel_diff_gamma_pl"] = 2*(pairs_table["gamma_pl_2"] - pairs_table["gamma_pl_1"]) / (pairs_table["gamma_pl_2"] + pairs_table["gamma_pl_1"])

    return pairs_table
#############################################################################



#############################################################################
# PRIORS
#############################################################################
# class CustomPrior(object):
#     def __init__(self, log_scatter=False, anisotropy='const'):
#         """Customized prior distribution

#         Args:
#             log_scatter (bool, optional): _description_. Defaults to False.
#             anisotropy (str, optional): _description_. Defaults to 'const'.
#         """
#         self._log_scatter = log_scatter
#         # we use flat priors on constant anisotropy, and 1/a_ani prior for Osipkov-Merrit anisotropy
#         if anisotropy == 'const': 
#             self._ani_log = False
#         else:
#             self._ani_log = True


#     def __call__(self, kwargs_cosmo, kwargs_lens, kwargs_kin, kwargs_source, kwargs_los):
#         return self.log_likelihood(kwargs_cosmo, kwargs_lens, kwargs_kin, kwargs_source, kwargs_los)

#     def log_likelihood(self, kwargs_cosmo, kwargs_lens, kwargs_kin, kwargs_source, kwargs_los):

#         logL = 0

#         if self._log_scatter is True:
#             lambda_mst_sigma = kwargs_lens.get('lambda_mst_sigma', 1)
#             logL += np.log(1/lambda_mst_sigma)
#             a_ani_sigma = kwargs_kin.get('a_ani_sigma', 1)
#             logL += np.log(1/a_ani_sigma)
#             sigma_v_sys_error = kwargs_kin.get('sigma_v_sys_error', 1)
#             logL += np.log(1/sigma_v_sys_error)
#         if self._ani_log is True:
#             a_ani = kwargs_kin.get('a_ani', 1)
#             logL += np.log(1/a_ani)
#         return logL

class OmegaMPrior(object):
    def __init__(self, mean, sigma):
        """
        Gaussian prior on Omega_m.
        
        Args:
            mean (float): Target mean for Omega_m (e.g., 0.3)
            sigma (float): Gaussian width (e.g., 0.05)
        """
        self._mean = mean
        self._sigma = sigma

    def __call__(self, kwargs_cosmo, kwargs_lens, kwargs_kin, kwargs_source, kwargs_los):
        """
        This method is called by CosmoLikelihood inside the MCMC loop.
        """
        # Extract the current value of Omega_m being sampled
        om = kwargs_cosmo['om']
        
        # Calculate the Gaussian Log-Likelihood
        # We ignore the normalization constant as it doesn't affect the MCMC sampler
        logL = -0.5 * ((om - self._mean) / self._sigma)**2
        
        return logL