# -*- coding: utf-8 -*-
'''
FILENAME:
    hydrology_processes.py

DESCRIPTION:
    This file contains functions related to hydrological processes.

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
import xarray as xr
import datetime as dt
import pdb


# ======================================================================================================
# Hydrological processes for snow cover
# ======================================================================================================

def get_rain_and_snow(model, fluxes, states):
    '''This routine computes snowfall according to WIGMOSTA'''
    temp_range = model.opt['snowf_upper'] - model.opt['rainf_lower']
    snowfrct = np.ma.minimum(
        1.0,
        np.ma.maximum(0, (model.opt['snowf_upper'] - states['tsurf']) /
                      temp_range))

    fluxes['snowf'] = fluxes['precip'] * snowfrct
    fluxes['rainf'] = np.ma.maximum(0, fluxes['precip'] - fluxes['snowf'])

# ======================================================================================================
def get_potential_snowmelt(model, fluxes, states, time):
    '''Computes potential snowmelt using the daily degree approach (based on MPI-HM)'''
    if model.opt['meltscheme'].lower() in ['temporal', 'both']:
        # Compute daylength for every grid cell
        today = dt.datetime.strptime(str(time.values)[:10], '%Y-%m-%d')
        daylen = utr.daylength(today, model.grid['lat'].values)
        fdaylen = np.stack((daylen / 24.0,) * len(model.grid['lon']), axis=-1)
    # Compute melt factor for different schemes
    if model.opt['meltscheme'].lower() == 'spatial':
        # Meltfactor depends on orography alone
        melt_factor = model.param.ddfac.fillna(0).values
    elif model.opt['meltscheme'].lower() == 'temporal':
        # Meltfactor depends only on daylength alone
        melt_factor = fdaylen * 8.3 + 0.7
    elif model.opt['meltscheme'].lower() == 'both':
        # Meltfactor depends on daylength and orography
        melt_factor = np.ma.maximum(fdaylen * model.param.ddfac.fillna(0).values * 1.33 , 0.7)
    else:
        raise LookupError(
                'Invalid choice',model.opt['meltscheme'].lower(),'for melt scheme')

    fluxes['smelt'] = np.ma.maximum(
        0, melt_factor * (states['tsurf'] - model.opt['melt_crit']))

    if 'log' in vars(model).keys():
        model.log.add_value(fluxes['smelt'], 'smelt_pot', 'Potential snow melt')
        model.log.add_value(states['tsurf'], 'tsurf', 'Surface temperature', unit='K')

# ======================================================================================================
def update_snow(model, fluxes, states):
    '''This function computes the throughfall onto the canopy based on
       subroutine THROUGH from the old MPI-HM (IRAIME == 12)
    '''
    if 'log' in vars(model).keys():
        model.log.add_value(states['swe'], 'swe_old', 'Snow water equivalent from last time step')
        model.log.add_value(states['wliq'], 'wliq_old', 'Liquid SWE content from last time step')

    # # Refreezing of liquid water content within snow cover if below certain temperature
    with np.errstate(invalid='ignore'):
        freezing = states['tsurf'] < model.opt['t_refreeze']
    states['swe'] = np.ma.where(freezing, states['swe'] + states['wliq'],
                                states['swe'])
    states['wliq'] = np.ma.where(freezing, states['wliq'] * 0,
                                 states['wliq'])

    # Compute new snow cover and reduce potential snowmelt if higher than snow height
    states['swe'] += fluxes['snowf']
    fluxes['smelt'] = np.ma.where(states['swe'] > fluxes['smelt'],
                                  fluxes['smelt'], states['swe'])
    states['swe'] -= fluxes['smelt']

    # Update liquid water content in snow
    wliq_max = states['swe'] * model.opt['frc_liquid']
    states['wliq'] += fluxes['smelt']
    overflow = states['wliq'] > wliq_max
    fluxes['smelt'] = np.ma.where(overflow, states['wliq'] - wliq_max, 0)
    states['wliq'] = np.ma.where(~overflow, states['wliq'], wliq_max)

    # Compute throughfall
    fluxes['rainmelt'] = fluxes['rainf'] + fluxes['smelt']

    if 'log' in vars(model).keys():
        model.log.add_value(states['swe'], 'swe_new', 'Snow water equivalent')
        model.log.add_value(states['wliq'], 'wliq_new', 'Liquid swe content')
        model.log.add_value(fluxes['rainf'], 'rainf', 'Rainfall')
        model.log.add_value(fluxes['snowf'], 'snowf', 'Snowfall')
        model.log.add_value(fluxes['rainmelt'], 'rainmelt', 'Rainfall+Snowmelt')
        model.log.add_value(fluxes['smelt'], 'smelt', 'Snowmelt')

# ======================================================================================================
# Hydrological processes for soil storage
# ======================================================================================================
def get_surface_runoff(model, fluxes, states, fcover):
    '''Separation of throughfall into surface runoff and infiltration'''
    # computed using the Improved ARNO Scheme (MPI-HM IEXC == 5)

    # Prepare temporary fields and shortcuts
    beta = np.ma.masked_where(
        model.grid['lsm'] == 0,
        model.param.bmod.values)  # Modified beta parameter
    rm_cap = np.ma.masked_where(
        model.grid['lsm'] == 0,
        model.param.wcap.values)  # Maximum water holding capacity
    rm_max = np.ma.masked_where(
        model.grid['lsm'] == 0,
        model.param.wmax.values)  # Maximum subgrid soil moisture
    rm_min = np.ma.masked_where(
        model.grid['lsm'] == 0,
        model.param.wmin.values)  # Minimum subgrid soil moisture

    # Compute subgrid root zone soil moisture rm_sub
    rm_sub = rm_max - (rm_max - rm_min) * (1 -
                                           (states['rootmoist'] - rm_min) /
                                           (rm_cap - rm_min))**(1 /
                                                                (1 + beta))
    rm_sub = np.ma.where(states['rootmoist'] <= rm_min,
                         states['rootmoist'], rm_sub)

    # Compute single components of surface runoff equation and set bounds
    c1 = ((rm_max - rm_sub) / (rm_max - rm_min))**(1 + beta)
    c1 = np.minimum(c1, 1)
    c2 = ((rm_max - rm_sub - fluxes['throu']) / (rm_max - rm_min))**(1 + beta)
    c2 = np.maximum(c2, 0) 
    # Compute surface runoff regimes
    no_qs = fluxes['throu'] < 0
    too_dry = rm_sub + fluxes['throu'] <= rm_min
    overflow = rm_sub + fluxes['throu'] >= rm_max

    # Compute subgrid surface runoff and excess flow
    excess = np.ma.where(fluxes['throu'] > (rm_cap - states['rootmoist']),
                         fluxes['throu'] + (states['rootmoist'] - rm_cap),
                         0)
    rm_res = np.ma.where(rm_min - states['rootmoist'] > 0,
                         rm_min - states['rootmoist'], 0)
    qs = fluxes['throu'] - rm_res - ((rm_max - rm_min) /
                                     (1 + beta)) * (c1 - c2)

    # very small throughfall might cause negative surface runoff
    qs = np.maximum(qs, 0)

    # Combine different surface runoff fluxes based on regime and surface state
    qs = np.ma.where(overflow, excess, qs)
    qs = np.ma.where(too_dry, 0, qs)
    qs = np.ma.where(no_qs, 0, qs)
    qs = np.ma.where(fcover['frozen'], fluxes['throu'], qs)

    # Note: Surface runoff is implicitly scaled to non-lake fraction because
    # throughfall is scaled with the non-lake fraction
    fluxes['qs'] = qs

    # Check for negative surface runoff
    if fluxes['qs'].min() < 0:
        pdb.set_trace()
        raise ValueError("Negative surface runoff: ", fluxes['qs'].min())

# ======================================================================================================
def get_drainage(model, fluxes, states, fcover, dt):
    '''Leakage from soil storage'''
    # computed following MPI-HM ISUBFL == 1
    wcap = np.ma.masked_where(
        model.grid['lsm'] == 0,
        model.param.wcap.values)  # Maximum water holding capacity
    # Define drainage regimes
    no_qsb = np.ma.logical_or(
        model.param.wcap.values <= 1.0e-10, states['rootmoist'] <=
        model.param.wcap.values * model.opt['qsb_low'])
    full_qsb = np.logical_and(
        model.param.wcap.values > 1.0e-10, states['rootmoist'] >=
        model.param.wcap.values * model.opt['qsb_hig'])

    # Compute standard and maximum drainage
    qsb = model.opt['qsb_min'] * dt * (states['rootmoist'] / wcap)
    maxqsb = (qsb + dt * (model.opt['qsb_max'] - model.opt['qsb_min']) *
              ((states['rootmoist'] - wcap * model.opt['qsb_hig']) /
               (wcap - wcap * model.opt['qsb_hig']))**model.opt['qsb_exp'])

    # Attribute drainage based on regime
    qsb = np.ma.where(no_qsb, 0, qsb)
    qsb = np.ma.where(full_qsb, maxqsb, qsb)
    qsb = np.ma.where(qsb > states['rootmoist'], states['rootmoist'], qsb)
    qsb = np.ma.where(fcover['frozen'], 0, qsb)

    fluxes['qsb'] = qsb

    # Check for negative drainage
    if fluxes['qsb'].min() < 0:
        raise ValueError("Negative subsurface drainage: ",
                         fluxes['qsb'].min())

# ======================================================================================================
def get_skinevap(model, fluxes, states, fcover, date):
    '''compute evaporation from skin and canopy'''
    # based on the subroutine EVAPSKIN from the old MPI-HM (ISKIN == 1)
    # however, skin and canopy evaporation and storages are computed individually
    if model.opt['with_skin']:
        # Interpolate LAI to daily state
        lai_daily = (utr.monthly_interpol(
            field=model.param['lai'], fdate=date, bounds='zero')).to_masked_array()
        # Update maximum skin and canopy reservoir capacity (local)
        maxcap = {
                'fbare': model.opt['skincap1'],
                'fveg': model.opt['skincap1'] * lai_daily
                }
        skinstor = {
                'fbare': np.ma.where(fcover['fbare'] > 0, states['skinstor'] / fcover['fbare'], 0.0),
                'fveg': np.ma.where(fcover['fveg'] > 0, states['canopystor'] / fcover['fveg'], 0.0),
                }
        for sktype, skevap in zip(
                ['fbare', 'fveg'], ['skinevap', 'canoevap']):
            # Compute wet skin fraction, PET fraction and local evaporation
            wetfract = np.ma.where(maxcap[sktype] > 0,
                (skinstor[sktype] + fluxes['rainmelt']) / maxcap[sktype], 0.0)
            wetfract = np.ma.maximum(0.0, np.ma.minimum(1.0, wetfract))
            petfract = np.ma.where(wetfract * fluxes['potevap'] > 0,
                (skinstor[sktype] + fluxes['rainmelt']) / (wetfract * fluxes['potevap']), 0.0)
            petfract = np.ma.maximum(0.0, np.ma.minimum(1.0, petfract))
            # Compute skin and canopy evaporation and scale to grid cell
            fluxes[skevap] = fluxes['potevap'] * wetfract * petfract * fcover[sktype]
        model.param['canocap'] = xr.DataArray(maxcap['fveg'] * fcover['fveg'],
                coords=model.param.area.coords, dims=model.param.area.dims, name='canocap', attrs={
                    'long_name': 'Maximum canopy moisture storage', 'units': 'm2 m-2'})
        model.param['skincap'] = xr.DataArray(maxcap['fbare'] * fcover['fbare'],
                coords=model.param.area.coords, dims=model.param.area.dims, name='skincap', attrs={
                    'long_name': 'Maximum skin moisture storage', 'units': 'm2 m-2'})

    else:
        fluxes['skinevap'] = fluxes['potevap'] * 0
        fluxes['canoevap'] = fluxes['potevap'] * 0

    # Check for negative bare soil evaporation
    if fluxes['skinevap'].min() < 0:
        raise ValueError("Negative skin evaporation from bare soil: ",
                         fluxes.skinevap.min())
    if fluxes['canoevap'].min() < 0:
        raise ValueError("Negative canopy evaporation from vegetation: ",
                         fluxes.canoevap.min())

# ======================================================================================================
def get_transpiration(model, fluxes, states, fcover):
    '''compute plant transpiration'''
    # based on subroutine EVAPACT from the old MPI-HM (IEVAP == 4)
    wcap = np.ma.masked_where(
        model.grid['lsm'] == 0,
        model.param.wcap.values)  # Maximum water holding capacity
    crit = np.ma.masked_where(
        model.grid['lsm'] == 0,
        model.param.crit.values)  # Maximum water holding capacity
    wilt = np.ma.masked_where(
        model.grid['lsm'] == 0,
        model.param.wilt.values)  # Maximum water holding capacity
    # Define wet and dry soil moisture regimes
    max_transp = np.ma.logical_and(wcap > 1.0e-10,
                                   states['rootmoist'] >= crit)
    no_transp = np.ma.logical_or(wcap <= 1.0e-10,
                                 states['rootmoist'] <= wilt)

    # Reduce potential evaporation by canopy evap
    if model.opt['with_skin']:
        potevap_fveg = np.ma.maximum(0, fluxes['potevap'] - np.ma.where(
            fcover['fveg'] > 0, fluxes['canoevap'] / fcover['fveg'], 0))
    else:
        potevap_fveg = fluxes['potevap']

    # Set transpiration to potential for wet soil moisture regime, zero for dry soil moisture regime
    # and scale linearly with available water for transitional regime
    transp = potevap_fveg * ((states['rootmoist'] - wilt) /
                                  (crit - wilt))
    transp = np.ma.where(max_transp, potevap_fveg, transp)
    transp = np.ma.where(no_transp, 0, transp)
    # Correct transpiration wherever root zone moisture would drop below wilting point
    rm_avail = np.ma.where(states['rootmoist'] - wilt < 0, 0,
                           states['rootmoist'] - wilt)
    transp = np.ma.where(transp > rm_avail, rm_avail, transp)

    fluxes['transp'] = (transp * fcover['fveg']).filled(0)

    # Check for negative transpiration
    if fluxes['transp'].min() < 0:
        pdb.set_trace()
        raise ValueError("Negative transpiration: ", fluxes['transp'].min())

# ======================================================================================================
def get_soilevap(model, fluxes, states, fcover):
    '''compute bare soil evaporation'''
    # based on the subroutine EVAPACT from the old MPI-HM (IEVAP == 4)
    wcap = np.ma.masked_where(
        model.grid['lsm'] == 0,
        model.param.wcap.values)  # Maximum water holding capacity
    wlow = np.ma.masked_where(
        model.grid['lsm'] == 0,
        model.param.wlow.values)  # Maximum water holding capacity
    # Define wet and dry soil moisture regimes
    no_sevap = np.ma.logical_or(wcap <= 1.0e-10,
                                states['rootmoist'] <= wlow)

    # Reduce potential evaporation by skin evap
    if model.opt['with_skin']:
        potevap_baresoil = np.ma.maximum(0, fluxes['potevap'] - np.ma.where(
            fcover['fbare'] > 0, fluxes['skinevap'] / fcover['fbare'], 0))
    else:
        potevap_baresoil = fluxes['potevap']

    # Compute bare soil evaporation
    sevap = potevap_baresoil * ((states['rootmoist'] - wlow) /
                                 (wcap - wlow))
    sevap = np.ma.where(no_sevap, 0, sevap)

    # Correct bare soil evaporation wherever root zone moisture would drop below wlow
    rm_avail = np.ma.where(states['rootmoist'] - wlow < 0, 0,
                           states['rootmoist'] - wlow)
    sevap = np.minimum(rm_avail, sevap)

    fluxes['sevap'] = (sevap * fcover['fbare']).filled(0)

    # Check for negative bare soil evaporation
    if fluxes['sevap'].min() < 0:
        raise ValueError("Negative bare soil evaporation: ",
                         fluxes.sevap.min())

# ======================================================================================================
def update_skincanopy(model, fluxes, states, fcover):
    '''update skin and canopy storage and correct fluxes if necessary'''
    #
    if 'log' in vars(model).keys():
        model.log.add_value(states['skinstor'], 'skinstor_old', 'Skin storage from last time step')
        model.log.add_value(states['canopystor'], 'canopystor_old', 'Canopy storage from last time step')
        model.log.add_value(fluxes['rainmelt'] * (fcover['fbare'] + fcover['fveg']), 'rainmelt_land',
                'Rainfall and Snowmelt scaled to land fraction')
    #
    if model.opt['with_skin']:
        # Update skin and canopy reservoir
        states['skinstor'] += (fluxes['rainmelt'] * fcover['fbare'] - fluxes['skinevap'])
        states['canopystor'] += (fluxes['rainmelt'] * fcover['fveg'] - fluxes['canoevap'])
        # Adapt evaporation in case is reduces storages below zero
        fluxes['skinevap'] = np.ma.where(states['skinstor'] < 0,
                np.maximum(0, fluxes['skinevap'] + states['skinstor']), fluxes['skinevap'])
        states['skinstor'] = np.maximum(0, states['skinstor'])
        fluxes['canoevap'] = np.ma.where(states['canopystor'] < 0,
                np.maximum(0, fluxes['canoevap'] + states['canopystor']), fluxes['canoevap'])
        states['canopystor'] = np.maximum(0, states['canopystor'])
        # Compute throughfall to the soil
        fluxes['throu'] = (np.maximum(0, states['skinstor'] - model.param['skincap'].values)
                + np.maximum(0, states['canopystor'] - model.param['canocap'].values))
        # Reduce skin and canopy content accordingly
        states['skinstor'] = np.minimum(model.param['skincap'].values, states['skinstor'])
        states['canopystor'] = np.minimum(model.param['canocap'].values, states['canopystor'])

    else:
        fluxes['throu'] = fluxes['rainmelt'] * (fcover['fbare'] + fcover['fveg'])

    # Check for negative moisture states
    if states['skinstor'].min() < 0:
        raise ValueError("Negative bare soil skin storage after update_skincanopy: ",
                         states['skinstor'].min())
    if states['canopystor'].min() < 0:
        raise ValueError("Negative canopy storage after update_skincanopy: ",
                         states['canopystor'].min())

    if 'log' in vars(model).keys():
        model.log.add_value(states['skinstor'], 'skinstor_new', 'Skin reservoir')
        model.log.add_value(states['canopystor'], 'canopystor_new', 'Canopy reservoir')
        model.log.add_value(fluxes['throu'], 'throu', 'Throughfall')
        model.log.add_value(fluxes['skinevap'], 'skinevap', 'Skin evaporation')
        model.log.add_value(fluxes['canoevap'], 'canoevap', 'Canopy evaporation')
        model.log.add_value(model.param['skincap'].values, 'skinmax', 'Maximum skin reservoir capacity')
        model.log.add_value(model.param['canocap'].values, 'canomax', 'Maximum canopy reservoir capacity')


# ======================================================================================================
def update_soil(model, fluxes, states):
    '''update soil moisture state and correct fluxes if necessary'''
    # This water balance scheme is not yet flexible enough. Later on, it needs to account
    # for the different fluxes for different land cover types!
    wcap = model.param.wcap.values

    if 'log' in vars(model).keys():
        model.log.add_value(states['rootmoist'], 'rootmoist_old', 'Root zone soil moisture from last time step')

    # Update soil moisture state
    states['rootmoist'] += (fluxes['throu'] - fluxes['qs'])
    states['rootmoist'] -= (fluxes['transp'] + fluxes['sevap'] +
                            fluxes['qsb'])

    # Add soil moisture overflow to surface runoff
    overflow = states['rootmoist'] > wcap
    fluxes['qs'] = np.ma.where(overflow,
                               fluxes['qs'] + states['rootmoist'] - wcap,
                               fluxes['qs'])
    states['rootmoist'] = np.ma.where(overflow, wcap, states['rootmoist'])
    if fluxes['qs'].min() < 0:
        raise ValueError("Negative surface runoff after overflow: ",
                         fluxes['qs'].min())

    # Soil below zero --> reduce evaporation and drainage equally
    if states['rootmoist'].min() < 0:
        states['rootmoist'], corflx = utr.correct_neg_stor(states['rootmoist'],
                [fluxes['transp'], fluxes['sevap'], fluxes['qsb']])
        fluxes['transp'], fluxes['sevap'], fluxes['qsb'] = corflx

    # Check for negative soil moisture
    if states['rootmoist'].min() < 0:
        raise ValueError("Negative soil moisture after update_soil: ",
                         states.rootmoist.min())

    if 'log' in vars(model).keys():
        model.log.add_value(states['rootmoist'], 'rootmoist_new', 'Root zone soil moisture')
        model.log.add_value(fluxes['qs'], 'qs', 'Surface runoff')
        model.log.add_value(fluxes['qsb'], 'qsb', 'Subsurface runoff')
        model.log.add_value(fluxes['transp'], 'transp', 'Transpiration')
        model.log.add_value(fluxes['sevap'], 'sevap', 'Bare soil evaporation')

# ======================================================================================================
# Hydrological processes for lake storage
# ======================================================================================================
def get_lakeevap(model, fluxes, states, fcover):
    '''compute open water evaporation over lakes'''
    levap = fluxes['potevap'] * fcover['flake']

    # Don't evaporation more than the lake contains
    fluxes['levap'] = np.minimum(levap, states['lakestor'])

    # Check for negative evaporation
    if fluxes['levap'].min() < 0:
        raise ValueError("Negative lake evaporation: ",
                         fluxes['levap'].min())

# ======================================================================================================
def get_lakeleak(model, fluxes, states, fcover):
    '''compute leakage from lakes into groundwater using soilmoisture deficit as proxy'''

    if model.opt['with_leakage']:
        # Compute cell average soil moisture deficit
        lleak = np.ma.maximum(0, model.param['wcap'].values - states['rootmoist'])

        # Don't infiltrate more than the lake contains
        lleak = np.ma.minimum(lleak, states['lakestor'])

        # Additionally scale with lake fraction to avoid small lakes leaking
        # all their water into the ground
        fluxes['lleak'] = np.ma.where(fcover['frozen'], 0,
                lleak * fcover['flake']**2)

        # Check for negative evaporation
        if fluxes['lleak'].min() < 0:
            raise ValueError("Negative lake leakage: ",
                             fluxes['lleak'].min())
    else:
        fluxes['lleak'] = fluxes['rainmelt'] * 0

# ======================================================================================================
def update_surface_water(model, fluxes, states, fcover):
    '''update surface water storage ( == old overland flow storage)
       and compute outflow based on retention times
    '''

    if 'log' in vars(model).keys():
        model.log.add_value(states['lakestor'], 'lakestor_old', 'Lake storage from last time step')

    # # Compute rainmelt input for lake fraction
    zeros = fluxes['rainmelt'] * 0
    rainmelt = fluxes['rainmelt'] * fcover['flake']

    # Update cell average lake storage and check for negative values
    states['lakestor'] += (rainmelt + fluxes['qs'] - fluxes['levap'] - fluxes['lleak'])

    # Lake below zero --> reduce evaporation and leakage equally
    if states['lakestor'].min() < 0:
        states['lakestor'], corflx = utr.correct_neg_stor(states['lakestor'],
                [fluxes['levap'], fluxes['lleak']])
        fluxes['levap'], fluxes['lleak'] = corflx


    # If no lake is prescribed, use unscaled lake depth
    celllake = np.where(fcover['flake'] > 0, fcover['flake'], 1)
    # Compute outflow based on storage retention time and surface state
    lake_depth = states['lakestor'] / celllake
    flowcoeff = 1.0 / (model.temporary['lag_land'] + 1.0)
    fluxes['qsl'] = np.ma.maximum(0, np.ma.minimum(states['lakestor'],
        lake_depth * flowcoeff * celllake))
    states['lakestor'] -= fluxes['qsl']

    # Check for negative lake storage
    if states['lakestor'].min() < 0:
        raise ValueError(
            "Negative lake storage after lake outflow computation: ",
            states['lakestor'].min())

    if 'log' in vars(model).keys():
        model.log.add_value(states['lakestor'], 'lakestor_new', 'Lake storage')
        model.log.add_value(fluxes['qsl'], 'qsl', 'Lake runoff')
        model.log.add_value(rainmelt, 'rainmelt_lake', 'Rainfall + Snowmelt on lake fraction')
        model.log.add_value(fluxes['levap'], 'levap', 'Lake evaporation')
        model.log.add_value(fluxes['lleak'], 'lleak', 'Lake leakage into groundwater')

# ======================================================================================================
# Hydrological processes for groundwater storage
# ======================================================================================================

def update_groundwater(model, fluxes, states):
    '''update groundwater storage ( == old baseflow storage)
       and compute outflow based on retention times
    '''
    if 'log' in vars(model).keys():
        model.log.add_value(states['groundwstor'], 'groundwstor_old', 'Groundwater storage from last time step')

    # Add drainage and lake leakage to groundwater storage
    states['groundwstor'] += (fluxes['qsb'] + fluxes['lleak'])
    zeros = fluxes['qsb'] * 0

    # Compute outflow based on storage retention time
    flowcoeff = 1.0 / (model.temporary['lag_base'] + 1.0)
    fluxes['qg'] = np.ma.maximum(0, np.ma.minimum(states['groundwstor'],
        states['groundwstor'] * flowcoeff))
    states['groundwstor'] -= fluxes['qg']

    # Check for negative groundwater storage
    if states['groundwstor'].min() < 0:
        raise ValueError(
            "Negative groundwater storage after update_groundwater: ",
            states.groundwstor.min())

    if 'log' in vars(model).keys():
        model.log.add_value(states['groundwstor'], 'groundwstor_new', 'Groundwater storage')
        model.log.add_value(fluxes['qg'], 'qg', 'Groundwater runoff')

# ======================================================================================================
# Hydrological processes for river routing
# ======================================================================================================
def update_riverflow(model, fluxes, states, dtime):
    '''update river storage using a linear reservoir cascade
    with subtimesteps and routing'''
    import streamflow as sfl

    conv2col = 1.0 / model.grid['landarea'] * 1000
    if 'log' in vars(model).keys():
        model.log.add_value(states['riverstor'].sum(axis=0) * conv2col, 'riverstor_old',
                           'River storage from last time step')

    # Store old inflow amount for water balance correction
    inflow_t0 = states['infl_subtime'] * 1

    states['infl_subtime'], fluxes['rivdis'], fluxes['dis'], fluxes['freshwater'], states['riverstor'], err = sfl.routing_cascade(
            fluxes['qsl'] + fluxes['qg'], states['infl_subtime'], states['riverstor'],
            model.temporary['lag_river'], model.temporary['ncasc_river'],
            model.grid['landarea'], model.temporary['flowsinks'], model.temporary['riverflow'],
            model.opt['rivsubtime'])
    if err > 1:
        raise ValueError("Routing error exceeds threshold: ",err)

    # Time step correction for global water balance due to using
    # inflow from the last time step
    model.temporary['riv_ts_corr'] = (states['infl_subtime'] - inflow_t0) / model.opt['rivsubtime']

    # Check for negative river storage
    if states['riverstor'].min() < 0:
        raise ValueError("Negative riverflow storage after update_riverflow: ",
                         states['riverstor'].min())

    # Write log data
    if 'log' in vars(model).keys():
        model.log.add_value(states['riverstor'].sum(axis=0) * conv2col, 'riverstor_new',
                           'River storage')
        model.log.add_value(fluxes['rivdis'] * conv2col, 'riv_in', 'Upstream inflow')
        model.log.add_value(fluxes['dis'] * conv2col, 'riv_out', 'River discharge')
