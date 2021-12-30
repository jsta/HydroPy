# -*- coding: utf-8 -*-
'''
FILENAME:
  utility_routines.py

DESCRIPTION:
  In this file utility routines for HydroPy are collected:
  - check_coords:     Check spatial coordinates
  - check_vector:     Compares vectors from different sources (usually latitude and longitude data)
  - check_lsm:        Checks whether all land points contain data and removes data from non-land points
  - monthly_interpol: Interpolate from monthly means to daily value 
  - linear_cascade:   Linear reservoir cascaded used for different storages

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

# Load module
import xarray as xr
import numpy as np
import calendar as cal
import datetime as dt
import warnings as warn
import dateutil.relativedelta as drd
import pdb


# ======================================================================================================
def dataset2double(xdata):
    '''returns new dataset with all fields converted to float64 type'''
    # Convert all datafields to float64
    dbldata = xr.Dataset()
    for var in xdata.data_vars:
        varattrs = xdata[var].attrs
        if xdata[var].dtype == 'float32':
            dbldata[var] = xdata[var].astype('float64').copy(deep=True)
        else:
            dbldata[var] = xdata[var].copy(deep=True)
        dbldata[var].attrs = varattrs
    xdata.close()

    return dbldata


# ======================================================================================================
def align_coords(set1, set2, mesg="Coordinate system mismatch"):
    '''Checks for differences between spatial coordinates of two DataArrays'''
    # Check for fundamental differences between spatial coordinates
    if (np.any(set1.coords['lon'] != set2.coords['lon'])
            or np.any(set1.coords['lat'] != set2.coords['lat'])):
        raise ValueError("Error: " + mesg)
    # Streamline coordinates as xr is picky about numerical precission
    newdata = xr.align(set1, set2, join='override')

    return newdata


# ======================================================================================================
def check_vector(vec1, vec2, err, mesg="Difference between two vectors"):
    # Checks whether the difference between two vectors is smaller than the error
    if np.any(abs(vec1 - vec2)) > err * 1.0e+6:
        print("Error: " + mesg)
        exit()


# ======================================================================================================
def check_lsm(field, lsm):
    if field.shape != lsm.shape:
        act_lsm = np.broadcast_to(lsm, field.shape)
    else:
        act_lsm = lsm

    if np.any(np.logical_and(
            field.mask,
            act_lsm > 0.5)):  # Check for missing data at land point ...
        raise ValueError("Data missing for land points")  # ... and exit
    elif np.any(np.logical_and(
            field.mask == False,
            act_lsm < 0.5)):  # Check for data on ocean points ...
        field = np.ma.masked_where(act_lsm < 0.5, field)  # ... and remove them
    return field


# ======================================================================================================
def correct_neg_stor(storage, fluxlist):
    '''Compensate negative storage by scaling all outgoing fluxes'''
    # Find negative storage cells and sum up fluxes
    neg_stor = storage < 0
    flux_sum = np.array(fluxlist).sum(axis=0)
    # Compute reduction ratio
    ratio = np.ma.where(neg_stor, 1.0 - abs(storage / flux_sum), 1)
    if ratio.min() < 0 or ratio.max() > 1:
        raise ValueError('Compensation for negative storage failed')
    # Set storage minimum to zero and apply reduction ratio
    corstor = np.ma.maximum(storage, 0)
    corflx = [np.ma.where(neg_stor, x * ratio, x) for x in fluxlist]

    return corstor, corflx


# ======================================================================================================
def monthly_interpol(field, fdate, bounds='fraction'):
    '''interpolate from monthly climatology to daily value'''
    try:
        today = str(fdate.values).split('T')[0]
    except:
        today = str(fdate).split('T')[0]
    with warn.catch_warnings():
        warn.simplefilter("ignore")
        intfield = field.interp(time=today)
    if bounds == 'fraction':
        intfield = intfield.where(intfield >= 0, 0)
        intfield = intfield.where(intfield <= 1, 1)
    elif bounds == 'zero':
        intfield = intfield.where(intfield >= 0, 0)

    return intfield


# ======================================================================================================
def daylength(date, lats):
    '''compute daylengths for a given date and a vector of latitudes
       based on https://gist.github.com/anttilipp/ed3ab35258c7636d87de6499475301ce
    '''
    year_day = date.timetuple().tm_yday
    last_day = dt.date(year=date.year, month=12, day=31).timetuple().tm_yday
    lat_rad = np.deg2rad(lats)
    daylength = np.zeros_like(lats)

    declEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+year_day)/float(last_day)))

    angle = -np.tan(lat_rad) * np.tan(np.deg2rad(declEarth))

    daylength = np.where(angle <= -1.0, 24.0, daylength)
    daylength = np.where(angle >=  1.0, 0.0, daylength)
    daylength = np.where(np.logical_and(angle > -1.0, angle < 1.0), 2 * np.rad2deg(np.arccos(angle)) / 15.0, daylength)

    return daylength

# ======================================================================================================
def compile_fortran_subroutines(fortfile):
    '''compiles fortran subroutines'''
    from numpy import f2py
    from pathlib import Path
    import sys

    recompile = True
    # Check for fortran shared library
    current = Path().absolute()
    sharedlib = current.joinpath('streamflow.so')
    if sharedlib.is_file():
        # Check modification date of library
        fileage = dt.datetime.now() - dt.datetime.fromtimestamp(sharedlib.stat().st_mtime)
        if fileage < dt.timedelta(days=1):
            # Reuse library
            recompile = False
            print('Using existing fortran shared library')

    # Recompile library
    if recompile:
        parent = Path(__file__).resolve().parent
        with open(parent.joinpath(fortfile)) as sourcefile:
            sourcecode = sourcefile.read()
        cstat = f2py.compile(sourcecode, modulename='streamflow', extension='.f90', verbose=0)
        if cstat == 0:
            print('Compiled fortran subroutines')
            # Rename shared library and add to module search path
            for mymodule in Path().glob('streamflow.*.so'):
                mymodule.rename('streamflow.so')
        else:
            raise ValueError('Failed to compile',fortfile)

    # Add current directory to module path
    sys.path.insert(0, './')


# ======================================================================================================
def clean_files():
    '''Remove unneccessary files from former (crashed) simulations'''
    from pathlib import Path

    for oldstuff in Path().glob('*.so'):
        oldstuff.unlink()
