import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
import warnings

warnings.filterwarnings("ignore")

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
all_bands = [
    # 'g', 'r', 'i', 'z', 'y',
    'F062', 'F106', 'F129', 'F158', 'F184'
]


def extract_lens_properties(args):
    lens_id, lens = args
    
    deflector = lens.deflector
    source = lens.source(index=0)
    z_D = lens.deflector_redshift
    z_S = lens.source_redshift_list[0]
    theta_E = lens.einstein_radius[0]
    num_images = lens.image_number[0]

    xs, ys = lens.source(0).extended_source_position
    radial_dist_S = np.sqrt(xs**2 + ys**2)
    size_S = source.angular_size 

    mags = {}
    snrs = {}
    contrasts = {}
    
    fov_arcsec = np.max([theta_E * 4, 1])
    
    for b in all_bands:
        # Map to the correct observatory for SNR
        obs = 'LSST' if b in ['g', 'r', 'i', 'z', 'y'] else 'Roman'

        mags[f'mag_S_{b}'] = source.extended_source_magnitude(b)
        mags[f'mag_S_{b}_lensed'] = lens.extended_source_magnitude(band=b, lensed=True)[0]
        mags[f'mag_D_{b}'] = deflector.magnitude(b)
        
        # Calculate SNR
        snrs[f'snr_{b}'] = lens.snr(
            band=b,
            fov_arcsec=fov_arcsec,
            observatory=obs, 
            snr_per_pixel_threshold=1,
            exposure_time=642,  # seconds for Roman HLWAS Medium
        )
        
        # Calculate Contrast Ratio
        cr_raw = lens.contrast_ratio(band=b, source_index=0)
        cr_padded = np.array(list(cr_raw) + [np.nan] * (4 - len(cr_raw)))
        contrasts[f'contrast_ratio_{b}'] = cr_padded

    es_magnification = lens.extended_source_magnification[0]

    sigma_v_D = deflector.velocity_dispersion()
    stellar_mass_D = deflector.stellar_mass
    e1_mass_D, e2_mass_D = deflector.mass_ellipticity
    e_mass_D = np.sqrt(e1_mass_D**2 + e2_mass_D**2)
    gamma_pl = deflector.halo_properties.get('gamma_pl', 2.0)
    size_D = deflector.angular_size_light

    lenstronomy_kwargs = lens.lenstronomy_kwargs()
    lens_model_lenstronomy = LensModel(lens_model_list=lenstronomy_kwargs[0]["lens_model_list"])
    lenstronomy_kwargs_lens = lenstronomy_kwargs[1]["kwargs_lens"]
    
    deflector_center = deflector.deflector_center
    grid = np.linspace(-size_D, size_D, 500)
    grid_x, grid_y = np.meshgrid(grid + deflector_center[0], grid + deflector_center[1])
    
    kappa_map = lens_model_lenstronomy.kappa(grid_x, grid_y, kwargs=lenstronomy_kwargs_lens)
    mask = np.sqrt((grid_x - deflector_center[0])**2 + (grid_y - deflector_center[1])**2) < size_D / 2
    kappa_within_half_light_radii = np.nanmean(kappa_map[mask])

    D_s = cosmo.angular_diameter_distance(z_S)
    D_d = cosmo.angular_diameter_distance(z_D)
    D_ds = cosmo.angular_diameter_distance_z1z2(z_D, z_S)
    
    sigma_crit = ((const.c**2 / (4 * np.pi * const.G)) * (D_s / (D_d * D_ds))).to(u.Msun / u.pc**2).value
    surface_density = sigma_crit * kappa_within_half_light_radii

    surface_brightness_map = deflector.surface_brightness(grid_x, grid_y, band="g")
    mask_sb = np.sqrt((grid_x - deflector_center[0])**2 + (grid_y - deflector_center[1])**2) < size_D
    mean_surface_brightness = np.nanmean(surface_brightness_map[mask_sb])

    R_e_kpc_val = (cosmo.kpc_proper_per_arcmin(z_D) * ((size_D * u.arcsec).to(u.arcmin))).to(u.kpc).value

    return {
        "lens_id": lens_id, 
        "z_D": z_D, "z_S": z_S, "theta_E": theta_E, "num_images": num_images,
        "radial_dist_S": radial_dist_S, "size_S": size_S,
        **mags,
        **snrs,
        "es_magnification": es_magnification,
        "R_e_arcsec": size_D, "surf_bri_mag/arcsec2": mean_surface_brightness,
        "sigma_v_D": sigma_v_D, "stellar_mass_D": stellar_mass_D,
        "e1_mass_D": e1_mass_D, "e2_mass_D": e2_mass_D, "e_mass_D": e_mass_D,
        "gamma_pl": gamma_pl, 
        "R_e_kpc": R_e_kpc_val, "Sigma_half_Msun/pc2": surface_density,
        **contrasts
    }
