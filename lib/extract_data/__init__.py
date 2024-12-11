import numpy as np


def convert_age_to_depth(age):
    """From paleobathymetry-workflow/traditional_workflow/1_convert_age_to_basement_depth.py"""
    basement_depth = np.zeros_like(age)
    # If age is nan, make it a nan for basement depth
    mask_isnan = np.isnan(age)
    basement_depth[mask_isnan] = np.nan
    
    # This should NOT be needed, but adding just in case...
    # If the agegrid has negative ages, set them to -2600 (MOR depth)
    mask_age_0 = age < 0
    basement_depth[mask_age_0] = -2600
    
    # For crust between 0 and 20 Myr
    mask_age_0_20 = (age <= 20) & (age >= 0)
    basement_depth[mask_age_0_20] = -1.0 * (2600 + 365 * (age[mask_age_0_20] ** 0.5))
    
    # For crust older than 20 Myr
    mask_age_20_ = age > 20
    basement_depth[mask_age_20_] = -1.0 * (5651 - 2473 * np.exp(-0.0278 * age[mask_age_20_]))

    return basement_depth
