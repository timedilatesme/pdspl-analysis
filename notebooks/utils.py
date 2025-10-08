from astropy.table import Table
import numpy as np
from itertools import combinations
from hierarc.Likelihood.LensLikelihood.double_source_plane import beta2theta_e_ratio, beta_double_source_plane
# from lenstronomy.LensModel.lens_model import LensModel

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
# def generate_fake_catalog(source_table, num_samples):
#     """
#     Generates a fake galaxy catalog by resampling rows from a source table
#     and computes the number of pairs on the fundamental plane.

#     Args:
#         source_table (astropy.table.Table): The original data table to resample from.
#         num_samples (int): The number of galaxies to generate for the catalog.

#     Returns:
#         astropy.table.Table: The generated fake galaxy catalog.
#     """
#     indices = np.random.choice(len(source_table), size=num_samples, replace=True)
#     fake_data = source_table[indices]

#     # remove the 'lens_id' column if it exists
#     if 'lens_id' in fake_data.colnames:
#         fake_data.remove_column('lens_id')

#     # for each lens perturb the parameters a little bit by 1% of std of each parameter
#     for col in fake_data.colnames:
#         col_std = np.std(source_table[col])
#         if not col_std == 0:
#             fake_data[col] += np.random.normal(0, col_std, num_samples)*0.01

#     return fake_data

# def generate_fake_catalog(source_table, num_samples):
#     """
#     Generates a fake galaxy catalog by sampling points around the MFP of a source table.

#     Args:
#         source_table (astropy.table.Table): The original data table to resample from.
#         num_samples (int): The number of galaxies to generate for the catalog.

#     Returns:
#         astropy.table.Table: The generated fake galaxy catalog.
#     """
#     log_sigma_v_D = np.log10(source_table['sigma_v_D'])
#     log_R_e_kpc = np.log10(source_table['R_e_kpc'])
#     log_Sigma_half = np.log10(source_table['Sigma_half_Msun/pc2'])
#     coeffs_MFP = fit_plane(log_R_e_kpc, log_Sigma_half, log_sigma_v_D)
#     scatter = find_scatter(log_R_e_kpc, log_Sigma_half, log_sigma_v_D, coeffs_MFP)
#     scatter_std = np.std(scatter)

#     return fake_data

def run_pairing_simulation_from_data(data, threshold_rel_delta_z=0.01, 
                                     verbose=True):
    """
    Computes the number of pairs on the fundamental plane.

    Args:
        data (astropy.table.Table): The original data table to resample from.
        mag_limit (float): The magnitude limit to apply to the catalog.
        num_bins (int): The number of bins to use for pairing in the R_e-Sigma plane.
        threshold_rel_delta_z (float): The relative redshift difference threshold for a valid pair.

    Returns:
        int: The number of valid pairs found.
        astropy.table.Table: Table of valid pairs with columns 'lens_id_1' and 'lens_id_2'.

    """
    if verbose:
        print(f"\n--- Running simulation for {len(data)} lenses ---")


    # Step 1: Use the data
    data = data.copy()
    if len(data) < 2:
        return 0

    # Step 4: Bin the data and find pairs
    x = np.log10(data["R_e_kpc"])
    y = np.log10(data["Sigma_half_Msun/pc2"])

    # remove infinite and NaN values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    data = data[mask]

    # determine the bin_widths based on the data's stddev
    x_std = np.nanstd(x)
    y_std = np.nanstd(y)
    x_width_bin = 3.5 * x_std / (len(x)**(1/(2+2))) # Scott's rule
    y_width_bin = 3.5 * y_std / (len(y)**(1/(2+2))) # Scott's rule
    num_bins_x = int((np.nanmax(x) - np.nanmin(x)) / x_width_bin) + 1
    num_bins_y = int((np.nanmax(y) - np.nanmin(y)) / y_width_bin) + 1

    if verbose:
        print(f"x_std: {x_std}, y_std: {y_std}")
        print(f"Using bin widths of {x_width_bin:.2f} in x and {y_width_bin:.2f} in y for pairing.")
        print(f"Using {num_bins_x} bins in x and {num_bins_y} bins in y for pairing.")

    x_bins = np.linspace(np.nanmin(x), np.nanmax(x), num_bins_x)
    y_bins = np.linspace(np.nanmin(y), np.nanmax(y), num_bins_y)

    total_pairs = 0
    valid_pairs_idx1 = []
    valid_pairs_idx2 = []
    
    # Iterate through each bin on the plane
    if verbose:
        print("Finding pairs in the binned data...")
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            mask = (
                (x >= x_bins[i]) & (x < x_bins[i + 1]) &
                (y >= y_bins[j]) & (y < y_bins[j + 1])
            )
            
            # Need at least 2 points to form a pair
            if np.sum(mask) < 2:
                continue
            
            data_points_in_bin = data[mask]
            
            for lens1, lens2 in combinations(data_points_in_bin, 2):
                # check z_lens < z_source
                if lens1['z_D'] >= lens2['z_S'] or lens2['z_D'] >= lens1['z_S']:
                    continue
                if 2 * np.abs(lens1['z_D'] - lens2['z_D']) / (lens1['z_D'] + lens2['z_D']) <= threshold_rel_delta_z:
                    total_pairs += 1
                    valid_pairs_idx1.append(lens1['lens_id'])
                    valid_pairs_idx2.append(lens2['lens_id'])
    if verbose:
        print(f"Found {total_pairs} pairs.")
    
    # astropy table of valid pairs
    valid_pairs_table = Table()
    valid_pairs_table['lens_id_1'] = valid_pairs_idx1
    valid_pairs_table['lens_id_2'] = valid_pairs_idx2

    return total_pairs, valid_pairs_table

def pdspl_pairing_table_from_valid_pairs_table(valid_pairs_table, full_data_table, cosmo_true):
    """
    Given a table of valid pairs (with columns 'lens_id_1' and 'lens_id_2'),
    return a full pairing table with all properties of the lenses.

    Args:
        valid_pairs_table (astropy.table.Table): Table of valid pairs with columns 'lens_id_1' and 'lens_id_2'.
        full_data_table (astropy.table.Table): The full data table containing all lens properties.

    Returns:
        astropy.table.Table: Full pairing table with properties of both lenses in each pair.
    """
    individual_pair_data = {
        'beta_E_pseudo': [],
        'beta_E_DSPL': [],
        'z_D1': [],
        'z_D2': [],
        'z_S1': [],
        'z_S2': [],
        'sigma_v_D1': [],
        'sigma_v_D2': [],
        'lens1_id': [],
        'lens2_id': [],
        }

    for lens1_id, lens2_id in zip(valid_pairs_table['lens_id_1'], valid_pairs_table['lens_id_2']):
        lens1 = full_data_table[full_data_table['lens_id'] == lens1_id][0]
        lens2 = full_data_table[full_data_table['lens_id'] == lens2_id][0]
        beta_E_pseudo = lens1['theta_E'] / lens2['theta_E']

        # check z_lens < z_source
        if lens1['z_D'] >= lens2['z_S'] or lens2['z_D'] >= lens1['z_S']:
            continue

        beta_DSPL = beta_double_source_plane(
            z_lens = np.mean([lens1['z_D'], lens2['z_D']]),
            z_source_1= lens1['z_S'],
            z_source_2= lens2['z_S'],
            cosmo = cosmo_true,
        )
        beta_E_DSPL = beta2theta_e_ratio(
            beta_dsp= beta_DSPL,
            gamma_pl= 2, #TODO: what if gamma_pl != 2?
            lambda_mst= 1
        )

        individual_pair_data['beta_E_pseudo'].append(beta_E_pseudo)
        individual_pair_data['beta_E_DSPL'].append(beta_E_DSPL)
        individual_pair_data['z_D1'].append(lens1['z_D'])
        individual_pair_data['z_D2'].append(lens2['z_D'])
        individual_pair_data['z_S1'].append(lens1['z_S'])
        individual_pair_data['z_S2'].append(lens2['z_S'])
        individual_pair_data['sigma_v_D1'].append(lens1['sigma_v_D'])
        individual_pair_data['sigma_v_D2'].append(lens2['sigma_v_D'])
        individual_pair_data['lens1_id'].append(lens1['lens_id'])
        individual_pair_data['lens2_id'].append(lens2['lens_id'])
    
    pairing_table = Table(individual_pair_data)
    return pairing_table

# def run_pairing_simulation(num_samples, source_table, mag_limit=22, threshold_rel_delta_z=0.01, 
#                            verbose=True, is_source_cut=False):
#     # Generates a fake galaxy catalog by resampling rows from a source table
#     fake_data = generate_fake_catalog(
#         source_table=source_table,
#         num_samples=num_samples,
#     )

#     # Step 2: Run the pairing simulation on the fake data
#     return run_pairing_simulation_from_data(
#         data=fake_data,
#         mag_limit=mag_limit,
#         threshold_rel_delta_z=threshold_rel_delta_z,
#         verbose=verbose,
#         is_source_cut=is_source_cut
#     )
#############################################################################