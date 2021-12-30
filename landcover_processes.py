# -*- coding: utf-8 -*-
'''
FILENAME:
    landcover_processes.py

DESCRIPTION:
    The model class contains global options and spatial parameters attribute
    dictionaries as well as the model processes processes
    (vertical water balance and river routing).

AUTHOR:
    Tobias Stacke

Copyright (C):
    2021 Helmholtz-Zentrum Hereon
    2020 Helmholtz-Zentrum Geesthacht

LICENSE:
    This program is free software: you can redistribute it and/or modify it under the
    terms of the GNU General Public License as published by the Free Software Foundation,
    either version 3 of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program.
    If not, see http://www.gnu.org/licenses/.
'''

# Module info
__author__ = 'Tobias Stacke'
__copyright__ = 'Copyright (C) 2021 Helmholtz-Zentrum Hereon, 2020 Helmholtz-Zentrum Geesthacht'
__license__ = 'GPLv3'

# Load modules
import numpy as np
import utility_routines as utr
from termcolor import colored
import pdb


# ======================================================================================================
# Landcover processes
# ======================================================================================================

def get_daily_cover(model, fcover, ctypes, fluxes, states, date):
    '''compute daily land cover type fractions'''
    # This routine interpolates the daily value from the monthly climatology
    # for all 3D cover fields, but considers 2D cover to be constant.
    # LCT priority follows the order of entrys in ctypes. If no residual area remains
    # the subsequent LCTs go hungry.
    # Note: HydroPy considers the land (non-ocean) fraction only and thus all CTLs are
    # treated as being relative to the LSM, e.g. range between 0-1

    # Set residual field to 1 and identify residual land cover type
    resi_name = ctypes[-1]
    resi_field = np.where(model.param['lsm'].values > 0, 1, 0).astype(np.float64)

    for ct in ctypes[:-1]:
        if ct not in model.param.data_vars:
            raise LookupError("Cover type", ct,
                              "not found in parameter data")

        # Check bounds for specific cover type
        if model.param[ct].min() < 0.0 or model.param[ct].max() > 1.0:
            raise ValueError("get_daily_cover: ERROR --> cover type", ct,
                             "outside of 0-1 bounds")

        # Interpolate between month if climatology or else use it as constant cover
        if 'time' in model.param[ct].dims:
            fcover[ct] = (utr.monthly_interpol(
                field=model.param[ct], fdate=date)).to_masked_array()
        elif model.param[ct].dims == ('lat', 'lon'):
            fcover[ct] = model.param[ct].to_masked_array()
        else:
            raise LookupError('Unexpected dimensions for parameter field',
                              ct, model.param[ct].dims)

        # Apply conditions for specific land cover types
        if ct == 'flake':
            # Check bounds for wetlands, too
            if model.param['fwetl'].min() < 0.0 or model.param['fwetl'].max() > 1.0:
                raise ValueError("get_daily_cover: ERROR --> cover type fwetl",
                                 "outside of 0-1 bounds")
            # Merge Lake and wetland fraction and provide minimum fraction as surface
            # water reservoir (surface runoff sink)
            flake = np.ma.maximum(fcover['flake'], model.param['fwetl'].to_masked_array())
            fcover[ct] = np.ma.minimum(np.ma.maximum(flake, 1.0e-10), 1)

        # Reduce cover fraction in case its already claims by higher priority class
        fcover[ct] = np.ma.minimum(fcover[ct], resi_field)

        # Reduce residum fraction fraction accordingly
        resi_field -= fcover[ct]

    if resi_field.min() < 0.0 or resi_field.max() > 1.0:
        raise ValueError(resi_name,
                         'residual cover fraction out of bounds')

    fcover[resi_name] = resi_field

    # Write fractions to log
    if 'log' in vars(model).keys():
        for ct in ctypes:
            model.log.add_value(fcover[ct], 'lcf_'+ct, ct+' cover fraction', unit='/', )

# ======================================================================================================
# Hydrological processes for snow cover
# ======================================================================================================
def diag_frozen_ground(model, states, fcover):
    '''diagnose frozen ground'''
    fcover['frozen'] = np.ma.where(states['tsurf'] < model.opt['melt_crit'], True, False)
