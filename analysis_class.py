# -*- coding: utf-8 -*-
'''
FILENAME:
    analysis_class.py

DESCRIPTION:
    This collections contains the different analysis object for variable evaluations done
    during model simulation. It contains:
    - mass balance computation for water
    - spin-up evaluation
    - log cell output as 1-dim netCDF file

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

# Load python functions
import pdb
import numpy as np
import datetime as dt
import xarray as xr
from termcolor import colored


# ======================================================================================================
# Mass balance class
# ======================================================================================================
class mass_balance:
    # ======================================================================================================
    # INITIALIZATION
    # ======================================================================================================
    def __init__(self, gridinfo, btype):
        '''collects source and sink mass flows as well as storage
           states and evaluates the global mass balance.
        '''

        self.area = gridinfo['area'] * 1.0
        self.landarea = gridinfo['landarea'] * 1.0
        self.btype = btype.title()

        # Check mass balance type
        if btype == 'water':
            self.bunit, self.conv = 'm3', 1000.0
            self.globunit, self.globconv = 'km3', 1.0e-9
            self.limitcell, self.limitglob = 1.0e-3, 1.0e-6
        else:
            raise LookupError('Unexpected mass balance type:',btype)
        self.conv2total = self.landarea / self.conv

        # Initialize fields
        self.sources = xr.DataArray(gridinfo['area'] * 0.0,
                                    name='sources',
                                    attrs={
                                        'long_name':
                                        'Accumulated '+btype+' sources',
                                        'units': self.bunit
                                    })
        self.sinks = xr.DataArray(gridinfo['area'] * 0.0,
                                  name='sinks',
                                  attrs={
                                      'long_name': 'Accumulated '+btype+' vertical sinks',
                                      'units': self.bunit
                                  })
        self.latsinks = xr.DataArray(gridinfo['area'] * 0.0,
                                     name='latsinks',
                                     attrs={
                                         'long_name': 'Accumulated '+btype+' lateral flow sinks',
                                         'units': self.bunit
                                     })
        self.ocsinks = xr.DataArray(gridinfo['area'] * 0.0,
                                    name='ocsinks',
                                    attrs={
                                        'long_name': 'Accumulated '+btype+' ocean sinks',
                                        'units': self.bunit
                                    })
        self.stor_start = xr.DataArray(
            gridinfo['area'] * 0.0,
            name='stor_start',
            attrs={
                'long_name': btype+' storage state at simulation start',
                'units': self.bunit
            })
        self.stor_end = xr.DataArray(
            gridinfo['area'] * 0.0,
            name='stor_end',
            attrs={
                'long_name': btype+' storage state at simulation end',
                'units': self.bunit
            })

    # ======================================================================================================
    def add_source(self, source, conv2total=True):
        '''Adds mass source'''
        if conv2total:
            self.sources += (np.nan_to_num(source) * self.conv2total)
        else:
            self.sources += np.nan_to_num(source)

    # ======================================================================================================
    def add_sink(self, sink, conv2total=True):
        '''Adds mass sink to the attribute'''
        if conv2total:
            self.sinks += (np.nan_to_num(sink) * self.conv2total)
        else:
            self.sinks += np.nan_to_num(sink)

    # ======================================================================================================
    def add_sink_latflow(self, latsinks, conv2total=True):
        '''Adds mass sink for lateral flows to the attribute'''
        if conv2total:
            self.latsinks += (np.nan_to_num(latsinks) * self.conv2total)
        else:
            self.latsinks += np.nan_to_num(latsinks)

    # ======================================================================================================
    def add_sink_2ocean(self, ocsinks, conv2total=True):
        '''Adds mass sink for lateral flow into ocean to the attribute'''
        if conv2total:
            self.ocsinks += (np.nan_to_num(ocsinks) * self.conv2total)
        else:
            self.ocsinks += np.nan_to_num(ocsinks)

    # ======================================================================================================
    def add_storage_start(self, storage, conv2total=True):
        '''Adds mass storage to the attribute'''
        if conv2total:
            self.stor_start += (np.nan_to_num(storage) * self.conv2total)
        else:
            self.stor_start += np.nan_to_num(storage)

    # ======================================================================================================
    def add_storage_end(self, storage, conv2total=True):
        '''Adds mass storage to the attribute'''
        if conv2total:
            self.stor_end += (np.nan_to_num(storage) * self.conv2total)
        else:
            self.stor_end += np.nan_to_num(storage)

    # ======================================================================================================
    def check_out(self, chunk, time, debug, infos):
        '''Output mass balance components and balance it'''
        ncout = False
        print("\n"+self.btype+" balance overview:")
        line = {
            'typ': '',
            'src': 'Sources',
            'snk': 'Sinks',
            'str_sta': "Storage_start",
            'str_end': "Storage_end",
            'bal': "Balance"
        }
        print(
            '{typ:<17} {src:<10} + {str_sta:<15} - {snk:<10} - {str_end:<15} = {bal:<10}'
            .format(**line))

        # Global balance for full field
        resi = (self.sources.sum() + self.stor_start.sum() - self.sinks.sum()
                - self.ocsinks.sum() - self.stor_end.sum())

        line = {
            'typ': 'Global sum:',
            'src': "{:8.3e}".format(self.sources.sum().values * self.globconv),
            'snk': "{:8.3e}".format((self.sinks.sum().values + self.ocsinks.sum().values) * self.globconv),
            'str_sta': "{:8.3e}".format(self.stor_start.sum().values * self.globconv),
            'str_end': "{:8.3e}".format(self.stor_end.sum().values * self.globconv),
            'bal': "{:8.3e}".format(resi.values * self.globconv),
            'unit': '['+self.globunit+']'
        }

        if abs(resi * self.globconv) < self.limitglob:  # Unit == km3 for global sum
            print(
                colored(
                    '{typ:<17} {src:<10} + {str_sta:<15} - {snk:<10} - {str_end:<15} = {bal:<10} {unit:<10}'
                    .format(**line), 'green'))
        else:
            print(
                colored(
                    '{typ:<17} {src:<10} + {str_sta:<15} - {snk:<10} - {str_end:<15} = {bal:<10} {unit:<10}'
                    .format(**line), 'red'))
            print(colored("*** "+self.btype+" balance not closed ***", 'red'))
            ncout = True

        # Mean grid cell balance
        resi = self.sources + self.stor_start - self.sinks - self.latsinks - self.stor_end
        land = np.ma.masked_where(
            abs(self.sources) + abs(self.stor_start) + abs(self.sinks) + abs(self.latsinks) +
            abs(self.stor_end) > 0, np.ones_like(resi)).mask
        line = {'typ': 'Grid cell mean:'}
        for nm, fld in zip(
            ['src', 'snk', 'str_sta', 'str_end', 'bal'],
            [self.sources, self.sinks + self.latsinks, self.stor_start, self.stor_end, resi]):
            line[nm] = "{:8.3e}".format(
                np.ma.masked_where(~land, fld).mean() /
                np.ma.masked_where(~land, self.area).mean() * self.conv)
        line['unit'] = '[kg m-2]'

        if abs(resi).max() < self.limitcell:  # Unit == m3 at grid cell level
            print(
                colored(
                    '{typ:<17} {src:<10} + {str_sta:<15} - {snk:<10} - {str_end:<15} = {bal:<10} {unit:<10}'
                    .format(**line), 'green'))
        else:
            print(
                colored(
                    '{typ:<17} {src:<10} + {str_sta:<15} - {snk:<10} - {str_end:<15} = {bal:<10} {unit:<10}'
                    .format(**line), 'red'))
            print(colored("*** "+self.btype+" balance not closed ***", 'red'))
            ncout = True

        # Output single fields if mass balance is violated
        if ncout or debug:
            balance = xr.Dataset()
            balance['source'] = self.sources
            balance['sink'] = self.sinks
            balance['stor_start'] = self.stor_start
            balance['stor_end'] = self.stor_end
            balance['balance'] = xr.DataArray(
                self.sources + self.stor_start - self.sinks - self.latsinks - self.stor_end,
                name='balance',
                attrs={
                    'long_name': self.btype + ' balance residuum',
                    'units': self.bunit
                })

            # Add metadata
            period = '(' + '-'.join(
                [str(time[0].values)[:10],
                 str(time[-1].values)[:10]]) + ')'
            balance.attrs = {
                'title':
                self.btype + ' balance fields for HydroPy ' + period,
                'institute': infos['institute'],
                'contact': infos['contact'],
                'version': infos['version'],
                'history':
                'Created ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            balance.to_netcdf('hydropy_'+self.btype[0].lower()+'b_' + chunk + '.nc')


# ======================================================================================================
# Spin-up evaluation class
# ======================================================================================================
class spinup:
    # ======================================================================================================
    # INITIALIZATION
    # ======================================================================================================
    def __init__(self):
        '''This class collects the restart states at the end of the simulation,
        and evaluates whether they are stable
        '''
        # Add evaluation attributes as needed
        self.start = None
        self.lastdiff = None
        self.newdiff = None
        self.var = []
        self.eva = {}
        self.pos = {}

# ===============================================================================================

    def add_states(self, states, cycle, infos, weights):
        '''Loop over states, and add statistics to
        evaluation time series
        '''

        if cycle == 0:
            # Set first entry to states at simulation start
            self.start = states.where(weights > 0)
        elif cycle == 1:
            # Compute first difference
            self.newdiff = (states - self.start).where(weights > 0)
            self.start = states.where(weights > 0)
        else:
            # Check change in differences
            self.lastdiff = self.newdiff.copy(deep=True)
            self.newdiff = (states - self.start).where(weights > 0)
            self.start = states.where(weights > 0)
            

# ===============================================================================================

    def evaluate(self, cycle):
        '''Simple approach: repeat simulation until either max change is below 0.1 % (my expert option)
        or ignore storage if change is constant (steadily increasing like swe)
        '''

        equil = []
        thres = 0.001

        if self.lastdiff is not None:
            # Write header for statistics overview
            print("\nSpinup state evaluation cycle", cycle)
            line = {
                'n': 'Variable',
                'u': 'Unit',
                'mdiff': 'Max. rel. change',
                'lat': 'Latitude',
                'lon': 'Longitude',
                'chg': u'\u0394Change',
                'stable': 'Equil'
            }
            print(
                '| {n:<15} | {u:<10} | {mdiff:<16} | {lat:>10} | {lon:>10} | {chg:>13} | {stable:>5}'
                .format(**line))

            for var in self.newdiff.data_vars:
                # Compute largest absolute difference
                maxloc = np.unravel_index(
                        self.newdiff[var].where(self.start[var] > 0).argmax(),
                        self.newdiff[var].shape)
                d_new = self.newdiff[var][maxloc]
                d_old = self.lastdiff[var][maxloc]
                state = self.start[var][maxloc]
                line = {
                    'n': var,
                    'u': state.units,
                    'mdiff': round(float(d_new / state), 4),
                    'lat': round(float(state.lat), 2),
                    'lon': round(float(state.lon), 2),
                }
                if abs(float(d_new / state)) < thres or float(state) == 0:
                    # Change in maximum relative difference quite low --> about stable
                    line.update({'chg': float(d_new) - float(d_old), 'stable': 'YES'})
                    equil.append(True)
                    col = 'green'
                elif float(d_new) == float(d_old):
                    # No equilibrium expected anaymore
                    line.update({'chg': float(d_new) - float(d_old), 'stable': 'UNLIKELY'})
                    equil.append(True)
                    col = 'red'
                else:
                    # Spinup is ongoing
                    line.update({'chg': float(d_new) - float(d_old), 'stable': 'SPINUP'})
                    equil.append(False)
                    col = 'yellow'

                print(colored(
                        '| {n:<15} | {u:<10} | {mdiff:<16} | {lat:>10} | {lon:>10} | {chg:13.6e} | {stable:>5}'
                        .format(**line), col))
        else:
            equil.append(False)

        return np.all(np.array(equil))


#  ======================================================================================================
# Log cell class
# ======================================================================================================
class log_cells:
    # ======================================================================================================
    # INITIALIZATION
    # ======================================================================================================
    def __init__(self, lfile, logcells, expid, lsm, area, infos):
        '''This class collects all states and fluxes at grid cell scale'''

        self.cells = {}

        # Add single cells
        for (lat, lon), ic in zip([x for x in logcells], range(len(logcells))):
            # Get nearest coordinates
            glat = float(lsm.lat.sel(lat=lat, method='nearest').values)
            glon = float(lsm.lon.sel(lon=lon, method='nearest').values)
            # Check data and get nearest indices
            if glat not in lsm.lat:
                raise ValueError('Logcell latitude', glat,
                                 'not found in land sea mask')
            if glon not in lsm.lon:
                raise ValueError('Logcell longitude', glon,
                                 'not found in land sea mask')
            if lsm.sel(lat=glat, lon=glon) <= 0:
                print(colored("*** Warning: Log cell at "+str(lat)+'N, '+ str(lon)+"E is not located on land ***", 'yellow'))
            # Prepare dataset for logcell
            self.cells[ic] = {'dataset': xr.Dataset(), 'values': {}}
            self.cells[ic]['dataset'].attrs = {
                    'title': 'Logcell output for HydroPy simulation '+expid+' at position '+str(glat)+'N, '+str(glon)+'E',
                    'institute': infos['institute'],
                    'contact': infos['contact'],
                    'version': infos['version'],
                    'landfract': float(lsm.sel(lat=glat, lon=glon)),
                    'cellarea': float(area.sel(lat=glat, lon=glon)),
                    'coords': str(glat)+'N, '+str(glon)+'E',
                    'index': str(int(np.where(lsm.lat == glat)[0]))+', '+ str(int(np.where(lsm.lon == glon)[0])),
                    'history': 'Created ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }

# ===============================================================================================

    def add_value(self, field, name, long_name, unit='kg m-2'):
        # Add a variable to log output or replace its value
        for ic in self.cells.keys():
            # Initialize data array and value dictionary
            if name not in self.cells[ic]['dataset'].data_vars:
                self.cells[ic]['dataset'][name] = xr.DataArray([],
                        name=name, attrs={'long_name': long_name, 'units': unit})
                self.cells[ic]['values'][name] = []
            # Add value to datalist
            iy, ix = map(int, self.cells[ic]['dataset'].attrs['index'].split(','))
            self.cells[ic]['values'][name].append(field[iy, ix] * 1)

# ===============================================================================================

    def calc_balance(self, tasklist, rivers):
        '''Computes different balances for HydroPy components'''
        for ic in self.cells.keys():
            # Convert lists in numpy arrays
            for var in self.cells[ic]['values'].keys():
                self.cells[ic]['values'][var] = np.array(self.cells[ic]['values'][var])
            # Set shortcut
            vals = self.cells[ic]['values']

            for task in tasklist:
                # Compute snow balance (Snow + Rain + SWE_old + Wliq_old = Rainmelt + SWE_new + Wliq_new)
                if task == 'snow':
                    self.cells[ic]['dataset']['wb_'+task] = xr.DataArray([], name='wb_'+task,
                            attrs={'long_name': 'Snow water balance', 'units': 'kg m-2'})
                    self.cells[ic]['values']['wb_'+task] = (
                            vals['rainmelt'] + vals['swe_new'] + vals['wliq_new']
                            - vals['rainf'] - vals['snowf'] - vals['swe_old'] - vals['wliq_old'])

                # Compute skin and canopy balance (Rainmelt + SkinResOld + CanopyResOld = Throu + SkinEvap + CanoEvap + SkinResNew + CanopyResNew)
                elif task == 'skin':
                    self.cells[ic]['dataset']['wb_'+task] = xr.DataArray([], name='wb_'+task,
                            attrs={'long_name': 'Skin and canopy balance', 'units': 'kg m-2'})
                    self.cells[ic]['values']['wb_'+task] = (
                            vals['throu'] + vals['canoevap'] + vals['skinevap'] + vals['skinstor_new'] + vals['canopystor_new']
                            - vals['rainmelt_land'] - vals['skinstor_old'] - vals['canopystor_old'])

                # Compute soil balance (Throu + Soil_old = SurfRO + Transp + BSevap + Drain + Soil_new)
                elif task == 'soil':
                    self.cells[ic]['dataset']['wb_'+task] = xr.DataArray([], name='wb_'+task,
                            attrs={'long_name': 'Soil water balance', 'units': 'kg m-2'})
                    self.cells[ic]['values']['wb_'+task] = (
                            vals['qs'] + vals['qsb'] + vals['transp'] + vals['sevap'] + vals['rootmoist_new']
                            - vals['throu'] - vals['rootmoist_old'])

                # Compute lake balance (Rainmelt + SurfRO + Lake_old = LakeRO + LakeEvap + Lake leakage + Lake_new)
                elif task == 'lake':
                    self.cells[ic]['dataset']['wb_'+task] = xr.DataArray([], name='wb_'+task,
                            attrs={'long_name': 'Lake water balance', 'units': 'kg m-2'})
                    self.cells[ic]['values']['wb_'+task] = (
                            vals['qsl'] + vals['levap'] + vals['lleak'] + vals['lakestor_new']
                            - vals['rainmelt_lake'] - vals['qs'] - vals['lakestor_old'])

                # Compute groundwater balance (Drainage + Lake leakage + GW_old = GWRO + GW_new)
                elif task == 'groundwater':
                    self.cells[ic]['dataset']['wb_'+task] = xr.DataArray([], name='wb_'+task,
                            attrs={'long_name': 'Groundwater balance', 'units': 'kg m-2'})
                    self.cells[ic]['values']['wb_'+task] = (
                            vals['qg'] + vals['groundwstor_new']
                            - vals['qsb'] - vals['lleak'] - vals['groundwstor_old'])

                # Compute river balance (Inflow + LakeRO + GWRO + River_old = Outflow + River_new)
                elif task == 'river':
                    if rivers:
                        self.cells[ic]['dataset']['wb_'+task] = xr.DataArray([], name='wb_'+task,
                                attrs={'long_name': 'River water balance', 'units': 'kg m-2'})
                        self.cells[ic]['values']['wb_'+task] = (
                                vals['riv_out'] + vals['riverstor_new']
                                - vals['riv_in'] -vals['qsl'] - vals['qg'] - vals['riverstor_old'])
                    else:
                        self.cells[ic]['values']['wb_'+task] = vals['throu'] * 0

                # Compute full water balance for cell (Precip + Storage_old = Evap + Discharge + Storage_new)
                elif task == 'full':
                    self.cells[ic]['dataset']['wb_'+task] = xr.DataArray([], name='wb_'+task,
                            attrs={'long_name': 'Complete water balance', 'units': 'kg m-2'})
                    # Storages
                    stor_new = (vals['swe_new'] +
                                vals['wliq_new'] +
                                vals['rootmoist_new'] +
                                vals['skinstor_new'] +
                                vals['canopystor_new'] +
                                vals['lakestor_new'] +
                                vals['groundwstor_new'])
                    stor_old = (vals['swe_old'] +
                                vals['wliq_old'] +
                                vals['rootmoist_old'] +
                                vals['skinstor_old'] +
                                vals['canopystor_old'] +
                                vals['lakestor_old'] +
                                vals['groundwstor_old'])
                    # Fluxes
                    precip = vals['rainf'] + vals['snowf']
                    evap = vals['sevap'] + vals['levap'] + vals['transp'] + vals['skinevap'] + vals['canoevap']
                    if rivers:
                        outflow = vals['riv_out'] - vals['riv_in']
                        stor_new += vals['riverstor_new']
                        stor_old += vals['riverstor_old']
                    else:
                        outflow = vals['qsl'] + vals['qg']
                    # Compute balance
                    self.cells[ic]['values']['wb_'+task] = precip + stor_old - evap - outflow - stor_new

                else:
                    raise LookupError('Partial balance', task,
                                      'not defined in calc_balance')

# ===============================================================================================

    def write_logfile(self, timeaxis, expid):
        '''write log cell data into 1D netCDF file'''
        for ic in self.cells.keys():
            # Convert all log variables to xarrays
            for var, data in self.cells[ic]['values'].items():
                long_name = self.cells[ic]['dataset'][var].attrs['long_name']
                units = self.cells[ic]['dataset'][var].attrs['units']
                self.cells[ic]['dataset'][var] = xr.DataArray(
                        data, coords=timeaxis.coords, dims=('time'),
                        attrs={'long_name': long_name, 'units': units})
            # Write to disk
            filename = expid + '_cell' + str(ic + 1).zfill(2) + '.nc'
            self.cells[ic]['dataset'].to_netcdf(filename, 
                    format='netCDF4_CLASSIC', engine='netcdf4')
