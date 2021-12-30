# -*- coding: utf-8 -*-
'''
FILENAME:
    parameter_class.py

DESCRIPTION:
    The model class contains global options and spatial parameters attribute
    dictionaries as well as the computation of derived parameter fields.

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
import calendar as cal
import subprocess as spr
from termcolor import colored
import sys
import json
import os
import pdb


# ======================================================================================================
# HydroPy model class
# ======================================================================================================
class parameter:

    # ======================================================================================================
    # INITIALIZATION
    # ======================================================================================================

    def __init__(self):
        # Class containing model options and parameters as attribute dictionaries as well as process functions

        # Set physical constants
        self.const = {
            'w_freeze': 273.15,  # Freezing temperatur of water [K]
            }

        # Set default global model options
        self.opt = {
            # General information
            'expid': 'hydropy',
            'contact': 'unknown',
            'institute': 'unknown',
            # Pathinformation for forcing, input and output directories
            'forcing': os.getcwd() + "/forcing",
            'input': os.getcwd() + "/input",
            'output': os.getcwd() + "/output",
            'para': os.getcwd() + "/input/hydropy_para.nc",
            'restart': None,
            'restdate': None,
            # Use optimization
            'use_fortran': True,
            # Forcing variables
            'forcvars': {'TSurf': ['tsurf', 'K'],
                         'Precip': ['precip', 'kg m-2 s-1'],
                         'PET': ['potevap', 'kg m-2 s-1']},
            'lsmcheckvar': 'TSurf', # Used for consistency check with land sea mask
            # Output variables and temporal resolution
            'daily': None,
            'monthly': ['precip', 'tws', 'qtot', 'evap', 'dis'],
            'chunk': None,
            'nccmpr' : 9, # netCDF compression rate
            # Enabled model processes
            'with_permafrost': True,
            'with_skin': True,
            'with_leakage': False,
            'with_rivers': True,
            # Global parameter values for snow processes
            'rainf_lower': self.const['w_freeze'] - 1.1,  # Lower threshold for liquid precipitation [K]
            'snowf_upper': self.const['w_freeze'] + 3.3,  # Upper threshold for solid  precipitation [K]
            'melt_crit': self.const['w_freeze'] - 0.0,  # Critical temperature for snow melt [K]
            't_refreeze': self.const['w_freeze'] - 0.0,  # Critical temperature for refreezing water [K]
            'frc_liquid': 0.06,  # Fraction of liquid water in snow cover [/]
            'meltscheme': 'temporal', # snow melt scheme: temporal, spatial, both
            # Global parameter values for soil and skin processes
            'skincap1': 0.2, # Skin reservoir capacity on one layer (ECHAM: 0.2) [kg m-2]
            'rm_crit': 0.75,  # Critical root zone soil moisture fraction [/]
            'qsb_min': 2.77778e-07,  # Minimum drainage parameter [kg m-2 s-1]
            'qsb_max': 2.77778e-05,  # Maximum drainage parameter [kg m-2 s-1]
            'qsb_exp': 1.5,  # ECHAM drainage exponent [/]
            'qsb_low': 0.05,  # Minimum soilmoisture content for drainage  [/]
            'qsb_hig': 0.90,  # Maximum soilmoisture content for drainage  [/]
            'sevap_low': 0.05,  # Minimum soilmoisture content for soil evap [/]
            'wcap_perma': 50.0,  # Maximum water holding capacity for permafrost [kg m-2]
            # Global parameter values for flow processes
            'rivsubtime': 4, # Sub-timesteps for river flow [#]
            'v_lake': 0.01, # Flow velocity for 100% lake fraction 0.1 [m s-1] or None to disable lake retention
            'v_wetl': 0.06, # Flow velocity for 100% wetland fraction 0.1 [m s-1] or None to disable wetland retention
            'cf_wela': 0.5, # Fraction where lake and wetland impact becomes dominant
            'fak_ovr': 1, # Lag Modifikation faktor for overland flow
            'fak_gw': 1, # Lag Modifikation faktor for groundwater flow
            'fak_riv': 1, # Lag Modifikation faktor for river flow
        }

        # Try to add git information if code is from repo
        try:
            gitdir = sys.path[0]+'/.git'
            self.opt['version'] = (spr.check_output(
                    ["git", "--git-dir="+gitdir, "describe", "--always", "--long"]).strip()
                    ).decode(sys.stdout.encoding)
        except:
            self.opt['version'] = 'unknown'
        self.opt['expid'] = self.opt['expid'] + '-' + self.opt['version'].replace('_','-')

        # Dictionary for parameter fields
        self.param = xr.Dataset()

        # Initialize grid dictionary
        self.grid = {}

        # Dictionary for temporary states and information
        self.temporary = {}


    # ======================================================================================================
    # CONFIGURATION OF OPTIONS
    # ======================================================================================================

    def update_from_ini(self, setupfile=os.getcwd() + "/setup.json"):
        # Replace default global model options from setup.json file

        if os.path.isfile(setupfile):
            with open(setupfile) as optfile:
                setupdata = json.load(optfile)
                for optkey, optval in setupdata.items():
                    if optkey not in self.opt.keys():
                        print("Found unknown key in "+setupfile+": ", optkey)
                        sys.exit(1)
                    else:
                        self.opt[optkey] = optval
        else:
            print(colored("No setup file found, using default parameters",
                'yellow'))

    # ======================================================================================================
    def update_from_cli(self, configlist):
        # Replace options from default or setup file with command line options
        for string in configlist:
            try:
                newconfig = json.loads(string[0])
            except:
                print('Cannot convert string'+string[0]+'into json object')
                sys.exit(1)

            for optkey, optval in newconfig.items():
                if optkey in self.opt.keys():
                    self.opt[optkey] = optval
                else:
                    print("Got unknown option from command line: ", optkey)
                    sys.exit(1)

    # ======================================================================================================
    def print_options(self):
        with open("hydropy_options.json", "w") as optfile:
            json.dump(self.opt, optfile, indent=4, sort_keys=True)

        for optkey in sorted(self.opt.keys()):
            line = {'opt': optkey, 'val': str(self.opt[optkey])}
            print('{opt:<20} = {val:<50}'.format(**line))

    # ======================================================================================================
    def update_all(self, debug, spinup):
        '''Set some more global switches'''
        self.expid = self.opt['expid']
        self.with_permafrost = self.opt['with_permafrost']
        self.with_skin = self.opt['with_skin']
        self.with_rivers = self.opt['with_rivers']
        self.debug = debug
        
        # Set switch for spinup state
        if spinup == 0:
            self.spinup = False
        else:
            self.spinup = True

        # Verify sensible use of restart file and date
        if self.opt['restdate'] is None and self.opt['restart'] is not None:
            raise LookupError('External restart file provided but without defined restart date')
        if self.opt['restdate'] is not None and self.opt['restart'] is None:
            raise LookupError('Restart date provided without providing external restart file')

    # ======================================================================================================
    # READING MODEL PARAMETER FIELDS AND BUILD DERIVED FIELDS
    # ======================================================================================================

    def get_parameter(self, parafile):
        try:
            fileobj = utr.dataset2double(xdata=xr.open_dataset(parafile))
        except OSError:
            print("Cannot open file " + parafile)
            sys.exit(1)

        # Read parameter data
        for param in fileobj.data_vars:
            if param.lower() in self.param.data_vars:
                if (fileobj[param] - self.param[param.lower()]).min() != (
                        fileobj[param] - self.param[param.lower()]).max() != 0:
                    raise LookupError(
                        "Error: Parameter", param,
                        "already exists in parameter list but with different values"
                    )
                    sys.exit(1)
            else:
                self.param[param.lower()] = fileobj[param]

        # Sanity checks for specific parameter fields
        for param in ['rout_lat', 'rout_lon']:
            if param in self.param.data_vars:
                if np.any(np.isnan(self.param[param])):
                    raise LookupError(
                        "Error: No nan or missing values allowed in", param)
                    sys.exit(1)

        # Replace missing values with zero
        self.param = self.param.fillna(0)

    # ======================================================================================================
    def update_parameter(self, grid):
        ''' set all non-routing parameter fields to zero for cells without land area
        '''
        # Define list of parameters which might be needed outside of land cells
        globvars = ['lsm', 'area', 'rivdir', 'rout_lat', 'rout_lon', 'topo_std', 'srftopo', 'slope_avg']
        landarea = xr.DataArray(grid['landarea'], coords=self.param['area'].coords, dims=self.param['area'].dims)
        landarea_3d = xr.broadcast(self.param['fveg'], landarea)[1]

        for var in [x for x in self.param.data_vars if x not in globvars]:
            if len(self.param[var].shape) == 2:
                self.param[var] == self.param[var].where(landarea > 0, 0)
            elif len(self.param[var].shape) == 3:
                self.param[var] == self.param[var].where(landarea_3d > 0, 0)
            else:
                raise LookupError('Unexpected number of dimensions in parameter field variable',var)


    # ======================================================================================================
    def get_grid(self, griddata):
        '''Store grid information from forcing data'''
        g = self.grid
        g['coords'] = griddata.stream.coords
        g['dims'] = tuple(griddata.stream[self.opt['lsmcheckvar']].dims)
        g['shape'] = tuple(griddata.stream[self.opt['lsmcheckvar']].shape)
        # Get latitude and longitude
        g['lat'] = griddata.stream['lat']
        g['lon'] = griddata.stream['lon']
        g['nlat'] = len(griddata.stream['lat'].values)
        g['nlon'] = len(griddata.stream['lon'].values)

        # Get time vector and restart time (assuming a regular time interval)
        g['time'] = griddata.stream['time']
        g['ntime'] = len(g['time'])
        g['resttime'] = g['time'][0] - (g['time'][1] - g['time'][0])
        g['restday'] = str(g['resttime'].values).split('T')[0].replace('-', '')
        g['duration'] = g['time'][-1] - g['resttime']
        g['period'] = ' - '.join([
            str(g['time'][0].values).split('T')[0],
            str(g['time'][-1].values).split('T')[0]
            ])

        # Create time series of monthly mean dates for climatology processing
        year=int(str(g['time'][0].values).split('-')[0])
        start, end = dt.datetime(year=year, month=1, day=1), dt.datetime(year=year+1, month=1, day=1)
        alldays = np.array([dt.timedelta(days=x) for x in range((end-start).days)]) + start
        g['monstamp'] = xr.DataArray(
                alldays, coords={'time': alldays}, dims=('time',), name='time').resample(time="1MS").mean()
        g['monstamp'] = g['monstamp'].assign_coords(time=g['monstamp'].values)

        # Store length of actual year
        if cal.isleap(int(str(g['time'].values[0])[:4])):
            g['yearlength'] = 366
        else:
            g['yearlength'] = 365

        # Identify simulation duration and set chunksize
        if g['duration'] / np.timedelta64(1, 's') >= 365 * 24 * 3600:
            # Yearly chunk
            g['timeid'] = str(g['time'].values[0])[:4]
        elif g['duration'] / np.timedelta64(1, 's') >= 29 * 24 * 3600:
            # Monthly chunk
            g['timeid'] = str(g['time'].values[0]).replace('-', '')[:6]
        elif g['duration'] / np.timedelta64(1, 's') >= 24 * 3600:
            # Daily chunk
            g['timeid'] = str(g['time'].values[0]).replace('-', '')[:8]

        # Replace time vector in climatological data with actual year and
        # and add 1 month before and after year for interpolation
        newcoords = {'time': g['monstamp']['time'], 'lat': self.param.lat, 'lon': self.param.lon}
        newdims = ['time', 'lat', 'lon']
        for var in [x for x in self.param.data_vars if 'month' in self.param[x].dims]:
            newdata = self.param[var].values
            ln, un = self.param[var].long_name, self.param[var].units
            field = xr.DataArray(newdata, coords=newcoords, dims=newdims,
                    attrs={'long_name': ln, 'units': un})
            f0, f1 = field.isel(time=-1), field.isel(time=0)
            f0['time'].values = f0['time'].values - np.timedelta64(365,'D')
            f1['time'].values = f1['time'].values + np.timedelta64(365,'D')
            self.param[var] = xr.concat([f0,field,f1], dim='time').transpose('time', 'lat', 'lon')

    # ======================================================================================================
    def get_lsm(self, forcdata):
        '''return minimal lsm based on parameter and actual forcing data'''
        if "lsm" not in self.param.data_vars:
            raise LookupError("Error: No LSM found in parameter files")
        g = self.grid
        # Modify land sea mask: original mask, forcing data and glacier
        g['lsm'] = self.param['lsm']
        g['lsm'] = g['lsm'].where(forcdata > 0, 0.0)
        if 'glacier' in self.param.keys():
            # Substract absolute glacier area from LSM
            abs_glacier = g['lsm'] * self.param['glacier']
            g['lsm'] = (g['lsm'] - abs_glacier).where(g['lsm'] > abs_glacier, 0.0)
            print("\nSubstract glacier fraction from land fractions")
        g['area'] = self.param['area']

        # Compute area and weights for land surface
        g['landarea'] = (g['area'] * g['lsm']).to_masked_array()
        g['landweights'] = g['landarea'] / g['landarea'].sum()

    # ======================================================================================================
    def set_permafrost(self):
        '''this function reduces maximum water capacity and water availability according
        to permafrost fraction
        '''
        # Holding capacity reduction
        wcap_perm = self.param.wcap.where(
            self.param.wcap <= self.opt['wcap_perma'],
            self.param.wcap * 0 + self.opt['wcap_perma'])
        cap_reduc = ((wcap_perm * self.param.perm + self.param.wcap *
                      (1 - self.param.perm)) / self.param.wcap)
        # Apply reduction to all water holding capacity parameters
        for para in ['wcap', 'wava', 'wmin', 'wmax']:
            attrs = self.param[para].attrs
            self.param[para] *= cap_reduc.fillna(0)
            self.param[para].attrs = attrs


    # ======================================================================================================
    def set_soilparam(self):
        '''return derived soil parameter fields'''
        self.param['wilt'] = self.param['wcap'] - self.param['wava']
        self.param['wilt'].attrs = {
            'long_name': 'wilting point',
            'units': 'kg m-2'
        }
        self.param['crit'] = self.param['wcap'] * self.opt['rm_crit']
        self.param['crit'].attrs = {
            'long_name': 'critical soil moisture',
            'units': 'kg m-2'
        }
        self.param['wlow'] = self.param['wcap'] * self.opt['sevap_low']
        self.param['wlow'].attrs = {
            'long_name': 'dry soil limit',
            'units': 'kg m-2'
        }
        self.param['boro'] = ((self.param['topo_std'] - 100.0)
                            / (self.param['topo_std'] + 1000.0))
        self.param['boro'] = self.param['boro'].where(self.param['boro'] > 0, 0)
        self.param['boro'].attrs = {
            'long_name': 'Rescaled orographical standard deviation',
            'units': '/'
        }
        self.param['imax'] = self.param['wcap'] * (1.0 + self.param['boro'])
        self.param['imax'].attrs = {
            'long_name': 'maximum infiltration capacity',
            'units': 'kg m-2'
        }
        self.param['oexp'] = 1.0 / (1.0 + self.param['boro'])
        self.param['oexp'].attrs = {
            'long_name': 'beta parameter exponent',
            'units': '/'
        }
        self.param['bmod'] = (self.param.beta + self.param.boro).where(
            self.param.boro >= 0.01, self.param.beta)
        self.param['bmod'].attrs = {
            'long_name': 'modified beta parameter',
            'units': '/'
        }


    # ======================================================================================================
    def get_fluxvars(self):
        '''returns dictionary with flux variable definitions. New variables have to be added here'''
        modelvars = {
                'precip': {'long': "Total precipitation", 'unit': "kg m-2 s-1",},
                'rainf': {'long': "Rainfall", 'unit': "kg m-2 s-1"},
                'snowf': {'long': "Snowfall", 'unit': "kg m-2 s-1"},
                'smelt': {'long': "Snowmelt", 'unit': "kg m-2 s-1"},
                'rainmelt': {'long': "Sum of liquid water input (rainfall + snowmelt)", 'unit': "kg m-2 s-1"},
                'throu': {'long': "Throughfall onto ground", 'unit': "kg m-2 s-1"},
                'evap': {'long': "Evapotranspiration", 'unit': "kg m-2 s-1"},
                'potevap': {'long': "Potential evapotranspiration", 'unit': "kg m-2 s-1"},
                'transp': {'long': "Plant transpiration", 'unit': "kg m-2 s-1"},
                'sevap': {'long': "Bare soil evaporation", 'unit': "kg m-2 s-1"},
                'levap': {'long': "Lake evaporation", 'unit': "kg m-2 s-1"},
                'lleak': {'long': "Lake leakage into groundwater", 'unit': "kg m-2 s-1"},
                'canoevap': {'long': "Canopy evaporation", 'unit': "kg m-2 s-1"},
                'skinevap': {'long': "Skin evaporation", 'unit': "kg m-2 s-1"},
                'qtot': {'long': "Total runoff", 'unit': "kg m-2 s-1"},
                'qs': {'long': "Surface runoff", 'unit': "kg m-2 s-1"},
                'qsb': {'long': "Subsurface drainage", 'unit': "kg m-2 s-1"},
                'qsl': {'long': "Overland flow into river", 'unit': "kg m-2 s-1"},
                'qg': {'long': "Grundwater runoff", 'unit': "kg m-2 s-1"},
                'rivdis': {'long': "River discharge (inflow)", 'unit': "m3 s-1"},
                'dis': {'long': "River discharge (outflow) and ocean inflow", 'unit': "m3 s-1"},
                'freshwater': {'long': "River discharge into ocean", 'unit': "m3 s-1"},
                }
        return modelvars


    # ======================================================================================================
    def get_statevars(self):
        '''returns dictionary with state variable definitions. New variables have to be added here'''
        modelvars = {
                'tsurf': {'long': "Surface temperature", 'unit': "K", 'rest': False},
                'swe': {'long': "Snow water equivalent", 'unit': "kg m-2", 'rest': True},
                'wliq': {'long': "Snow liquid water content", 'unit': "kg m-2", 'rest': True},
                'rootmoist': {'long': "Root zone soil moisture", 'unit': "kg m-2", 'rest': True},
                'skinstor': {'long': "Skin reservoir", 'unit': "kg m-2", 'rest': True},
                'canopystor': {'long': "Canopy reservoir", 'unit': "kg m-2", 'rest': True},
                'lakestor': {'long': "Surface water storage", 'unit': "kg m-2", 'rest': True},
                'groundwstor': {'long': "Groundwater storage", 'unit': "kg m-2", 'rest': True},
                'riverstor': {'long': "Riverflow storage", 'unit': "m3", 'rest': True, 'add_dim': ['res', np.arange(5), 'reservoir number', '/']},
                'infl_subtime': {'long': "Inflow from upstream cell", 'unit': "m3", 'rest': True},
                'tws': {'long': "Total water storage", 'unit': "kg m-2", 'rest': False},
                }
        return modelvars


    # ======================================================================================================
    def get_covervars(self):
        '''returns dictionary with land cover type variable definitions. New variables have to be added here'''
        modelvars = {
                'flake': {'long': "Surface water fraction", 'unit': "/"},
                'fveg': {'long': "Vegetation fraction", 'unit': "/"},
                'fbare': {'long': "Bare soil fraction", 'unit': "/"},
                }
        return modelvars


    # ======================================================================================================
    def get_flow_properties(self, routing):
        '''returns static list of cell indices and flow target indices'''
        import streamflow as sfl
        # Initialize list and fields
        riverflow = []
        lsm = self.grid['lsm'].values
        area = self.param['area'].values
        topo = self.param['srftopo'].values

        # The following computations are moved from the preproc scripts to the model to be
        # computed at runtime. All values are valid only for daily time steps and 0.5 deg
        # resolution.
        # Check result to generalize.
        #
        # set values calibrated for Vindelaelven catchment and modified with
        # Torneaelven experiment (all done by Stefan)
        ref_ovr = {'k0': 50.5566, 'n0': 1.11070, 'v0': 1.0885, 'dx': 171000.0}
        ref_riv = {'k0': 0.41120, 'n0': 5.47872, 'v0': 1.0039, 'dx': 228000.0}
        vmin = 0.1 # Minimum flow velocity [m s-1] --> 5.79 day for 50 km
        alpha, c = 0.1, 2 # Parameters for Sausen flow velocity computation
        
        # Compute slope, grid cell diameter and overland flow velocity
        slope_subg = self.param['slope_avg'].values
        dx_subg = (area / np.pi)**(0.5) * 2
        vel_subg = np.ma.maximum(vmin, c * slope_subg**alpha)

        if routing:
            # Get flow directions, cell distance and height difference
            rivfl, sinks, ic, dx_cell, dh_cell = sfl.eval_flowfield(
                self.param['rout_lat'].values.astype(np.int),
                self.param['rout_lon'].values.astype(np.int),
                area, topo, np.pi)
            self.temporary['riverflow'] = rivfl[0:ic + 1]
            self.temporary['flowsinks'] = sinks * 1
            # Compute directional slope and river flow velocity
            slope_cell = np.ma.where(dx_cell > 0, dh_cell / dx_cell, 0)
            vel_cell = np.ma.maximum(vmin, c * slope_cell**alpha)
            # Compute retention coefficient for rivers using inter-cell properties
            riv_k = ref_riv['k0'] * dx_cell / ref_riv['dx'] * ref_riv['v0'] / vel_cell
            riv_n = ref_riv['n0'] * np.ones_like(riv_k)
        else:
            # Substitude intra-cell properties with inner cell properties
            dx_cell = dx_subg
            vel_cell = vel_subg

        # Compute retention coefficient for surface water bodies using preferabley subgrid properties
        ovr_k = np.ma.where(slope_subg > 0,
            # Velocity based on subgrid slope
            ref_ovr['k0'] * dx_subg / ref_ovr['dx'] * ref_ovr['v0'] / vel_subg,
            # Velocity based on normal slope
            ref_ovr['k0'] * dx_cell / ref_ovr['dx'] * ref_ovr['v0'] / vel_cell
            )
        ovr_n = ref_ovr['n0'] * np.ones_like(ovr_k)

        # Compute retention coefficients for baseflow based on fixed properties
        oroscale = np.ma.maximum(0.01, self.param['boro'].values)
        base_k = 300.0 / (1.0 - oroscale + 0.01)
        base_k *= (dx_subg / 50000.0)  # Scaling with normalized grid cell size
        base_n = np.ones_like(base_k)

        # Apply correction faktors for sensitivity experiment
        ovr_k *= self.opt['fak_ovr']
        base_k *= self.opt['fak_gw']
        if routing:
            riv_k *= self.opt['fak_riv']

        # Add additional retention due to lakes and wetlands
        for f_wela, v_wela, n_wela in zip([self.param['flake'], self.param['fwetl']],
                                          [self.opt['v_lake'], self.opt['v_wetl']],
                                          ['lake', 'wetland']):
            if v_wela is not None:
                fract = np.ma.maximum(0, np.ma.minimum(1, f_wela.values))
                # Compute lake and wetland impact
                fract_scaling = 0.5 * (np.tanh(4.0 * np.pi * (fract - self.opt['cf_wela'])) + 1.0)
                if routing:
                    # Compute and modify river flow velocity
                    v_riv = np.ma.where(riv_k > 0, dx_cell / ( riv_k * riv_n * 86400.0), vmin)
                    incr_lag = np.ma.logical_and(v_riv > v_wela, fract > 1.0e-3)
                    v_red = np.ma.where(incr_lag,
                            1 - (1.0 - v_wela / v_riv ) * fract_scaling, 1)
                    riv_n = np.ma.where(incr_lag, (riv_n - 1) * (1.0 - fract_scaling) + 1, riv_n)
                    riv_k = np.ma.where(incr_lag, dx_cell / (riv_n * v_riv * v_red * 86400.0), riv_k)
                # Compute and modify surface flow velocity
                v_ovr = np.ma.where(ovr_k > 0, dx_subg / ( ovr_k * ovr_n * 86400.0), vmin)
                incr_lag = np.ma.logical_and(v_ovr > 0.1 * v_wela, fract > 1.0e-3)
                v_red = np.ma.where(incr_lag,
                        1 - (1.0 - (0.1 * v_wela) / v_ovr) * fract_scaling, 1)
                ovr_n = np.ma.where(incr_lag, (ovr_n - 1) * (1.0 - fract_scaling) + 1, ovr_n)
                ovr_k = np.ma.where(incr_lag, dx_subg / (ovr_n * v_ovr * v_red * 86400.0), ovr_k)
                print('Flow velocity for 100%',n_wela,'cover set to',v_wela,' m s-1')
            else:
                print('No flow retention due to',n_wela+'s')

        # Store flow coefficient in temporary fields with unit days
        self.temporary['lag_land'] = ovr_k
        self.temporary['lag_base'] = base_k
        if routing:
            self.temporary['lag_river'] = riv_k
        # Store number of flow cascades
        self.temporary['ncasc_land'] = ovr_n
        self.temporary['ncasc_base'] = base_n
        if routing:
            self.temporary['ncasc_river'] = riv_n

        # Scale all lag values towards integer cascade numbers
        for l in [x for x in self.temporary.keys() if 'lag_' in x]:
            c = l.replace('lag_', 'ncasc_')
            self.temporary[l] *= (self.temporary[c] / self.temporary[c].astype(int))

        print(colored(
            "Compute retention times and cascade numbers for all flow storages",
            'green'))
