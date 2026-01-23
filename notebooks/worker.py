# This entire cell will be saved as 'worker.py'

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel

cosmo_true = FlatLambdaCDM(H0=70, Om0=0.3)

# --- THE WORKER FUNCTION ---
def extract_lens_properties(args):
    """
    Worker function that takes a tuple (id, lens_object), extracts all
    properties, and returns them as a single dictionary.
    """
    lens_id, lens = args

    # try:
    deflector = lens.deflector
    source = lens.source(index=0)

    z_D = lens.deflector_redshift
    z_S = lens.source_redshift_list[0]
    theta_E = lens.einstein_radius[0]
    sigma_v_D = deflector.velocity_dispersion()
    e1_mass_D, e2_mass_D = deflector.mass_ellipticity

    half_light_radii_arcsec = deflector.angular_size_light

    lenstronomy_kwargs = lens.lenstronomy_kwargs()
    lens_model_lenstronomy = LensModel(lens_model_list=lenstronomy_kwargs[0]["lens_model_list"])
    lenstronomy_kwargs_lens = lenstronomy_kwargs[1]["kwargs_lens"]

    deflector_center = deflector.deflector_center
    grid = np.linspace(-half_light_radii_arcsec, half_light_radii_arcsec, 500)
    xs, ys = np.meshgrid(grid + deflector_center[0], grid + deflector_center[1])

    kappa_map = lens_model_lenstronomy.kappa(xs, ys, kwargs=lenstronomy_kwargs_lens)
    mask = np.sqrt((xs - deflector_center[0])**2 + (ys - deflector_center[1])**2) < half_light_radii_arcsec / 2
    kappa_within_half_light_radii = np.nanmean(kappa_map[mask])

    D_s = cosmo_true.angular_diameter_distance(lens.source_redshift_list[0])
    D_d = cosmo_true.angular_diameter_distance(lens.deflector_redshift)
    D_ds = cosmo_true.angular_diameter_distance_z1z2(lens.deflector_redshift, lens.source_redshift_list[0])

    sigma_crit = (const.c**2 / (4 * np.pi * const.G)) * (D_s / (D_d * D_ds))
    sigma_crit = sigma_crit.to(u.Msun / u.pc**2).value
    surface_density = sigma_crit * kappa_within_half_light_radii

    surface_brightness_map = deflector.surface_brightness(xs, ys, band="g")
    mask_sb = np.sqrt((xs - deflector_center[0])**2 + (ys - deflector_center[1])**2) < half_light_radii_arcsec
    mean_surface_brightness = np.nanmean(surface_brightness_map[mask_sb])

    R_e_kpc_val = (cosmo_true.kpc_proper_per_arcmin(lens.deflector_redshift) * \
                    ((half_light_radii_arcsec * u.arcsec).to(u.arcmin))).to(u.kpc).value

    ### contrast ratio of images in mag difference
    contrast_ratio_i = lens.contrast_ratio(band="i", source_index = 0)
    contrast_ratio_r = lens.contrast_ratio(band="r", source_index = 0)
    contrast_ratio_g = lens.contrast_ratio(band="g", source_index = 0)
    contrast_ratio_z = lens.contrast_ratio(band="z", source_index = 0)
    contrast_ratio_y = lens.contrast_ratio(band="y", source_index = 0)

    # make all to have len of 4
    contrast_ratio_i = np.array(list(contrast_ratio_i) + [np.nan] * (4 - len(contrast_ratio_i)))
    contrast_ratio_r = np.array(list(contrast_ratio_r) + [np.nan] * (4 - len(contrast_ratio_r)))
    contrast_ratio_g = np.array(list(contrast_ratio_g) + [np.nan] * (4 - len(contrast_ratio_g)))
    contrast_ratio_z = np.array(list(contrast_ratio_z) + [np.nan] * (4 - len(contrast_ratio_z)))
    contrast_ratio_y = np.array(list(contrast_ratio_y) + [np.nan] * (4 - len(contrast_ratio_y)))

    ## magnifications of point images and extended images
    magnification_point = lens.point_source_magnification()[0]
    # convert to array of length 4
    magnification_point = np.array(list(magnification_point) + [np.nan] * (4 - len(magnification_point)))

    magnification_extended = lens.extended_source_magnification[0] # not an array, just a number

    return {
        "lens_id": lens_id, "z_D": z_D, "z_S": z_S, "theta_E": theta_E, "sigma_v_D": sigma_v_D,
        "stellar_mass_D": deflector.stellar_mass, "mag_S_i": source.extended_source_magnitude("i"),
        "mag_S_r": source.extended_source_magnitude("r"), "mag_S_g": source.extended_source_magnitude("g"),
        "mag_S_z": source.extended_source_magnitude("z"), "mag_S_y": source.extended_source_magnitude("y"),
        "mag_D_i": deflector.magnitude("i"), "mag_D_r": deflector.magnitude("r"), "mag_D_g": deflector.magnitude("g"),
        "mag_D_z": deflector.magnitude("z"), "mag_D_y": deflector.magnitude("y"), "size_D": deflector.angular_size_light,
        "e1_mass_D": e1_mass_D, "e2_mass_D": e2_mass_D, "e_mass_D": np.sqrt(e1_mass_D**2 + e2_mass_D**2),
        "gamma_pl": deflector.halo_properties['gamma_pl'], "R_e_kpc": R_e_kpc_val, "R_e_arcsec": half_light_radii_arcsec,
        "Sigma_half_Msun/pc2": surface_density, "surf_bri_mag/arcsec2": mean_surface_brightness,
        "num_images": lens.image_number[0],
        "contrast_ratio_i": contrast_ratio_i,
        "contrast_ratio_r": contrast_ratio_r,
        "contrast_ratio_g": contrast_ratio_g,
        "contrast_ratio_z": contrast_ratio_z,
        "contrast_ratio_y": contrast_ratio_y,
        "ps_magnification": magnification_point,
        "es_magnification": magnification_extended,
    }
    # except Exception as e:
    #     print(f"Error processing lens ID {lens_id}: {e}")
    #     return None
