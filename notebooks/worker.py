# This entire cell will be saved as 'worker.py'

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
import warnings

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

cosmo_true = FlatLambdaCDM(H0=70, Om0=0.3)

def extract_lens_properties(args):
    """
    Worker function that extracts parameters from a lens object and returns them as a dictionary.
    This function is designed to be used in parallel processing.
    """
    lens_id, lens = args

    # --- Objects ---
    deflector = lens.deflector
    source = lens.source(index=0)

    # --- System Properties ---
    z_D = lens.deflector_redshift
    z_S = lens.source_redshift_list[0]
    theta_E = lens.einstein_radius[0]
    num_images = lens.image_number[0]

    # --- Source Properties (Geometry) ---
    # needed for Collett 2015 like cuts
    xs, ys = lens.source(0).extended_source_position
    radial_dist_S = np.sqrt(xs**2 + ys**2)
    size_S = source.angular_size 

    # --- Photometry (Source & Deflector) ---
    bands = ['g', 'r', 'i', 'z', 'y']
    mags = {}

    for b in bands:
        # 1. Unlensed Source
        mags[f'mag_S_{b}'] = source.extended_source_magnitude(b)

        # 2. Lensed Source
        mags[f'mag_S_{b}_lensed'] = lens.extended_source_magnitude(band=b, lensed=True)[0]

        # 3. Deflector
        mags[f'mag_D_{b}'] = deflector.magnitude(b)

    # --- Magnification ---
    es_magnification = lens.extended_source_magnification[0]

    # --- Deflector Mass & Geometry ---
    sigma_v_D = deflector.velocity_dispersion()
    stellar_mass_D = deflector.stellar_mass
    e1_mass_D, e2_mass_D = deflector.mass_ellipticity
    e_mass_D = np.sqrt(e1_mass_D**2 + e2_mass_D**2)
    gamma_pl = deflector.halo_properties.get('gamma_pl', 2.0)

    # Light Size
    size_D = deflector.angular_size_light # Half-light radius in arcsec

    # --- Advanced Physics (Kappa & Surface Brightness) ---
    lenstronomy_kwargs = lens.lenstronomy_kwargs()
    lens_model_lenstronomy = LensModel(lens_model_list=lenstronomy_kwargs[0]["lens_model_list"])
    lenstronomy_kwargs_lens = lenstronomy_kwargs[1]["kwargs_lens"]

    deflector_center = deflector.deflector_center
    grid = np.linspace(-size_D, size_D, 500)
    grid_x, grid_y = np.meshgrid(grid + deflector_center[0], grid + deflector_center[1])

    # Kappa Calculation
    kappa_map = lens_model_lenstronomy.kappa(grid_x, grid_y, kwargs=lenstronomy_kwargs_lens)
    mask = np.sqrt((grid_x - deflector_center[0])**2 + (grid_y - deflector_center[1])**2) < size_D / 2
    kappa_within_half_light_radii = np.nanmean(kappa_map[mask])

    D_s = cosmo_true.angular_diameter_distance(z_S)
    D_d = cosmo_true.angular_diameter_distance(z_D)
    D_ds = cosmo_true.angular_diameter_distance_z1z2(z_D, z_S)

    sigma_crit = (const.c**2 / (4 * np.pi * const.G)) * (D_s / (D_d * D_ds))
    sigma_crit = sigma_crit.to(u.Msun / u.pc**2).value
    surface_density = sigma_crit * kappa_within_half_light_radii

    # Surface Brightness Calculation (g-band)
    surface_brightness_map = deflector.surface_brightness(grid_x, grid_y, band="g")
    mask_sb = np.sqrt((grid_x - deflector_center[0])**2 + (grid_y - deflector_center[1])**2) < size_D
    mean_surface_brightness = np.nanmean(surface_brightness_map[mask_sb])

    # Physical Radius
    R_e_kpc_val = (cosmo_true.kpc_proper_per_arcmin(z_D) * \
                    ((size_D * u.arcsec).to(u.arcmin))).to(u.kpc).value

    # --- Contrast Ratios ---
    contrasts = {}
    for b in bands:
        cr_raw = lens.contrast_ratio(band=b, source_index=0)
        cr_padded = np.array(list(cr_raw) + [np.nan] * (4 - len(cr_raw)))
        contrasts[f'contrast_ratio_{b}'] = cr_padded

    return {
        "lens_id": lens_id, 
        "z_D": z_D, 
        "z_S": z_S, 
        "theta_E": theta_E, 
        "num_images": num_images,

        # Source (Union of geometry + photometry)
        "radial_dist_S": radial_dist_S,    # = sqrt(x_S^2 + y_S^2)
        "size_S": size_S,                  # Angular Size of Source (arcsec)
        **mags,                            # Contains mag_S_i AND mag_S_i_lensed AND mag_D_i for i in g,r,i,z,y bands
        "es_magnification": es_magnification,

        # Deflector Light
        "size_D": size_D,                  # Same as R_e_arcsec
        "surf_bri_mag/arcsec2": mean_surface_brightness,

        # Deflector Mass
        "sigma_v_D": sigma_v_D,
        "stellar_mass_D": stellar_mass_D,
        "e1_mass_D": e1_mass_D, 
        "e2_mass_D": e2_mass_D, 
        "e_mass_D": e_mass_D,
        "gamma_pl": gamma_pl, 

        # Physics
        "R_e_kpc": R_e_kpc_val, 
        "Sigma_half_Msun/pc2": surface_density,

        # Observables
        **contrasts
    }
