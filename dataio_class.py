# -*- coding: utf-8 -*-
'''
FILENAME:
    dataio_class.py

DESCRIPTION:
    This collections contains objects for grid definition,
    forcing operations and variable streams.

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
import os
import numpy as np
import xarray as xr
import netCDF4 as net
import utility_routines as utr
from termcolor import colored
from calendar import monthrange
import datetime as dt
import warnings
import sys
import pdb


# ===================== Forcing data class ===================================================
class forcing:
    def __init__(self, filename, mod):
        '''Initialize forcing object with dimensions, attributes and data'''

        # Open files
        try:
            self.stream = utr.dataset2double(
                xdata=xr.open_dataset(filename, drop_variables=['time_bnds']))
        except OSError:
            print("Cannot open file " + filename)
            sys.exit(1)

        forcvars = mod.opt['forcvars'].keys()
        forcunits = [x[1] for x in mod.opt['forcvars'].values()]

        # Convert all datafields to float64
        for var in [
                x for x in self.stream.data_vars
                if self.stream[x].dtype == 'float32'
        ]:
            self.stream[var] = self.stream[var].astype('float64')

        # Add timetag for output
        self.chunkid = filename.split('_')[-1].split('.')[0]

        # Check for required forcing variables
        if np.array([x in self.stream.data_vars for x in forcvars]).all():
            pass
        else:
            raise LookupError('Forcing variable missing, require ',
                              ','.join(forcvars))

        # Check for required forcing variables units
        for var, unit in zip(forcvars, forcunits):
            if self.stream[var].units != unit:
                raise ValueError(var, 'required in', unit, ', found',
                                 self.stream[var].units)

        # Warn setting negative values to zero
        for var in self.stream.data_vars:
            if self.stream[var].min() < 0:
                msg = '*** Warning: correcting negative values for ' + var
                print(colored(msg, 'cyan'))

# ============================================================================================

    def close(self):
        '''Close forcing data file'''
        self.stream.close()


# ===================== Output stream class ==============================================
class output_stream:
    def __init__(self, streamfile, model, title, step='daily'):
        '''Creates output file stream'''

        # Create netCDF file
        self.stream = net.Dataset(streamfile, 'w', format='NETCDF4_CLASSIC')

        # Add metadata to file
        self.stream.title = title
        self.stream.institute = model.opt['institute']
        self.stream.contact = model.opt['contact']
        self.stream.version = model.opt['version']
        self.stream.history = 'Created ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Prepare container for coordinates
        self.coords = {}
        self.vars = {}
        self.info = {'step': step}
        self.miss = 1.0e+20
        self.nccmpr = model.opt['nccmpr']


    def var2stream(self, coords, dims, varinfo):
        '''Add coordinates associated with a given variable'''

        # Check whether dimensions already exist
        for c in dims:
            if c not in self.coords:
                if c in ['time', 'cycle']:
                    # Special treatment for time dimension
                    self.coords[c] = self.stream.createDimension(c, None)
                    self.coords[c] = self.stream.createVariable(
                            c, 'f8', (c, ), zlib=True, complevel=self.nccmpr, fill_value=self.miss)
                    if c == 'time':
                        self.coords[c].units = 'days since 1900-01-01 00:00:00'
                        self.coords[c].calendar = 'proleptic_gregorian'
                else:
                    # Spatial dimensions
                    self.coords[c] = self.stream.createDimension(c, len(coords[c]))
                    self.coords[c] = self.stream.createVariable(
                            c, 'f8', (c, ), zlib=True, complevel=self.nccmpr, fill_value=self.miss)
                    self.coords[c].long_name = coords[c].long_name
                    self.coords[c].units = coords[c].units
                    self.coords[c][:] = coords[c].values

        # Create variable
        name, longname, units = varinfo
        self.vars[name] = self.stream.createVariable(
            name, 'f8', dims, zlib=True, complevel=self.nccmpr, fill_value=self.miss)
        self.vars[name].long_name = longname
        self.vars[name].units = units


    def write_variable(self, name, index, outdata, outtime):
        '''Write variable to output file'''

        # Add time step
        if 'time' in self.coords:
            if len(self.coords['time']) < index + 1:
                self.coords['time'][index] = net.date2num(
                        outtime, units=self.coords['time'].units, calendar=self.coords['time'].calendar)
        elif 'cycle' in self.coords:
            if len(self.coords['cycle']) < index + 1:
                self.coords['cycle'][index] = outtime
        else:
            raise LookupError('Error: no time step coordinate found while writing output')

        # Write variable to file
        self.vars[name][index] = outdata


    def close(self):
        '''Close output data streams'''
        self.stream.close()


# ===================== Variable class ===================================================
class variables:
    def __init__(self, gridinfo, vartype, restart=None, spinup=None):
        '''setup variable object class'''

        # Specific attributes
        self.default_coords = gridinfo['coords']
        self.default_dims = gridinfo['dims']
        self.default_shape = gridinfo['shape']

        # Get restart data if available
        if restart is None:
            self.restfile = None
        else:
            self.restfile = restart

        print(" ".join(["\nAdding variable streams for", vartype, "\n"]))
        line = {
            'n': 'Shortname',
            'ln': 'Long name',
            'u': 'Unit',
            'o': 'Output'
        }
        print('| {n:<15} | {ln:<30} | {u:<10} | {o:<20} |'.format(**line))

        # Prepare dictionary for variable information
        self.varlist = {}

        # Special output stream during spinup simulations
        self.spinup = spinup

        # Store last time step for chunk output
        self.firststep = dt.datetime.strptime(
                str(gridinfo['coords']['time'].values[0])[:23], '%Y-%m-%dT%H:%M:%S.%f')
        self.laststep = dt.datetime.strptime(
                str(gridinfo['coords']['time'].values[-1])[:23], '%Y-%m-%dT%H:%M:%S.%f')


# ============================================================================================

    def add_variable(self, name, longname, units, stream=None, add_dim=None, restart=None):
        '''add variables to list and create output'''


        varinfo = [name, longname, units]

        # Create required dimensions
        if add_dim is not None:
            # Add additional dimension
            new_coords = {
                'time': self.default_coords['time'],
                add_dim[0]: (add_dim[0], add_dim[1], {
                    'long_name': add_dim[2],
                    'units': add_dim[3]
                    }),
                'lat': self.default_coords['lat'],
                'lon': self.default_coords['lon'],
                }
            var_coords = xr.Dataset().assign_coords(new_coords).coords
            var_dims = (self.default_dims[0], add_dim[0]) + self.default_dims[1:]
            var_shape = (self.default_shape[0], len(add_dim[1])) + self.default_shape[1:]
        else:
            var_coords = self.default_coords
            var_dims = self.default_dims
            var_shape = self.default_shape


        # Replace time dimension with cycle for spinup simulations
        if self.spinup:
            new_coords = {
                'cycle': ('cycle', np.arange(1), {
                    'long_name': 'Spinup Cycle',
                    'units': '/'
                    }),}
            for c in var_coords:
                if c != 'time':
                    new_coords[c] = var_coords[c]
            var_coords = xr.Dataset().assign_coords(new_coords).coords
            var_dims = ('cycle',) + var_dims[1:]


        # Store variable information
        self.varlist[name] = {'infos': varinfo[1:], 'coords': var_coords,
                'dims': var_dims, 'shape': var_shape, 'restart': restart}

        if stream is not None:

            # Print log onto screen
            if len(longname) > 30:
                words = longname[:27].split()
                longname = ' '.join(words[:-1] + ['...'])
            output = ','.join([x.info['step'] for x in stream])
            line = {'n': name, 'ln': longname, 'u': units, 'o': output}
            print('| {n:<15} | {ln:<30} | {u:<10} | {o:<20} |'.format(**line))

            # Only generate stream if variable output is requested and prepare field
            # for long-term averages
            for ncfile in stream:
                ncfile.var2stream(coords=var_coords, dims=var_dims, varinfo=varinfo)
                if ncfile.info['step'] != 'daily':
                    # Set up field for long-term averages
                    self.varlist[name][ncfile.info['step']] = {
                            'mean': np.zeros(var_shape[1:]),
                            'step': 0,
                            'index': 0
                            }


# ============================================================================================

    def initialize(self, weights):
        '''initialize variable for all land surface points'''
        print("\nInitializing states")
        line = {
            'n': 'Shortname',
            'u': 'Unit',
            'min': "Minimum",
            'avg': "Mean",
            'max': "Maximum",
        }
        print('| {n:<15} | {u:<6} | {min:>10} {avg:>10} {max:>10} |'.format(
            **line))

        # Initialize restart dataset from file or create empty dataset
        try:
            self.restart = utr.dataset2double(
                xdata=xr.open_dataset(self.restfile))
            msg = ' '.join(['\nInitializing states from file',self.restfile])
            msgcol = 'green'
            found_restart = True
        except OSError:
            self.restart = xr.Dataset()
            msg = ' '.join(['\nRestart file',self.restfile,'not found\n-->',
                            'all states initialized with ZERO'])
            msgcol = 'red'
            found_restart = False


        # Missing variables initialized with zero
        msg_miss = []
        for var in self.varlist:
            if var not in self.restart.data_vars:
                if self.varlist[var]['restart']:
                    msg_miss.append(var)
                vdims = self.varlist[var]['dims'][1:]
                vshape = self.varlist[var]['shape'][1:]
                vcoords = {}
                for c in vdims:
                    vcoords[c] = self.varlist[var]['coords'][c]
                self.restart[var] = xr.DataArray(
                        np.zeros(vshape), coords=vcoords, dims=vdims, name=var, attrs={
                            'long_name': self.varlist[var]['infos'][0],
                            'units': self.varlist[var]['infos'][1]})

        # Add warning for missing variables
        if len(msg_miss) > 0 and found_restart:
            msg_miss = ' '.join(['\nVariables',','.join(msg_miss),'are missing in restart file\n-->',
                'initialized with ZERO'])
            msgcol_miss = 'red'
        else:
            msg_miss = None

        # Print land surface statistics
        for var in self.restart.data_vars:
            vmin = self.restart[var].where(weights > 0).min()
            vmax = self.restart[var].where(weights > 0).max()
            if self.restart[var].units == 'm3':
                vavg = (self.restart[var].where(weights > 0)).mean()
            else:
                vavg = (self.restart[var] * weights).sum()
            line = {
                'n': var,
                'u': self.restart[var].units,
                'min': float(vmin),
                'avg': float(vavg),
                'max': float(vmax),
            }
            print('| {n:<15} | {u:<6} | {min:10.4g} {avg:10.4g} {max:10.4g} |'.
                  format(**line))

        print(colored(msg, msgcol))

        if msg_miss is not None:
            print(colored(msg_miss, msgcol_miss))

# ============================================================================================

    def write2stream(self, streams, date, step, fields, area, dtime, flux=False):
        '''write time step data to output stream'''

        # Normalization for fluxes
        if flux:
            scale = 1.0 / dtime
        else:
            scale = 1.0

        # Get time information
        today = dt.datetime.strptime(str(date.values)[:23], '%Y-%m-%dT%H:%M:%S.%f')
        tomorrow = today + dt.timedelta(seconds=dtime)

        # Prepare output for variables
        for var in streams.keys():
            if var in fields.keys():
                # Prepare output field
                if var in ['freshwater', 'rivdis', 'riverstor', 'dis']:
                    # Don't apply lsm, as field contains ocean points
                    varfield = fields[var] * scale
                else:
                    varfield = np.ma.masked_where(area == 0, fields[var] * scale)
                outvar = self.varlist[var]
                # Write to all associated streams
                for stream in streams[var]:
                    write2file = False
                    outstep = stream.info['step']
                    # Add to long term mean
                    if outstep == 'daily':
                        outdata = varfield
                        outtime = today
                        index = step
                        write2file = True
                    else:
                        if np.ma.is_masked(varfield) and not np.ma.is_masked(outvar[outstep]['mean']):
                            outvar[outstep]['mean'] = np.ma.masked_where(varfield.mask, outvar[outstep]['mean'])
                        outvar[outstep]['mean'] += varfield
                        outvar[outstep]['step'] += 1
                        # Check whether output time step is due
                        if outstep == 'monthly':
                            # Monthly mean output
                            if tomorrow.month != today.month:
                                outdata = (outvar[outstep]['mean'] / outvar[outstep]['step'])
                                index = outvar[outstep]['index']
                                outvar[outstep]['mean'] = np.zeros(outvar['shape'][1:])
                                outvar[outstep]['step'] = 0
                                outvar[outstep]['index'] += 1
                                outtime = (dt.datetime(year=today.year, month=today.month, day=1)
                                        + dt.timedelta(days=monthrange(today.year, today.month)[1]*0.5))
                                write2file = True
                        elif outstep == 'chunk':
                            # Simulation mean output
                            if today == self.laststep:
                                outdata = (outvar[outstep]['mean'] / outvar[outstep]['step'])
                                index = outvar[outstep]['index']
                                outvar[outstep]['mean'] = np.zeros(outvar['shape'][1:])
                                outvar[outstep]['step'] = 0
                                outvar[outstep]['index'] += 1
                                if self.spinup:
                                    outtime = index + 1
                                else:
                                    outtime = self.firststep + (self.laststep - self.firststep) * 0.5
                                write2file = True
                        else:
                            raise LookupError('Error: Unexpected output time step',outstep)

                    if write2file:
                        stream.write_variable(var,index,outdata,outtime)


# ============================================================================================

    def write_restart(self, rfile, fields, time, expid, infos, weights):
        '''write last time step into restart file'''
        
        # Convert arrays to xarray
        restdata = xr.Dataset()
        for var in [x for x in self.varlist.keys() if self.varlist[x]['restart']]:
            vdims = self.varlist[var]['dims'][1:]
            vshape = self.varlist[var]['shape'][1:]
            vcoords = {}
            for c in vdims:
                vcoords[c] = self.varlist[var]['coords'][c]
            restdata[var] = xr.DataArray(
                    fields[var], coords=vcoords, dims=vdims, name=var,
                    attrs={'long_name': self.varlist[var]['infos'][0],
                           'units': self.varlist[var]['infos'][1]})

        # Write some statistics
        print("\nWriting restart data for experiment " + expid.upper())
        line = {
            'n': 'Shortname',
            'u': 'Units',
            'min': "Minimum",
            'avg': "Mean",
            'max': "Maximum"
        }
        print('| {n:<15} | {u:<6} | {min:>10} {avg:>10} {max:>10} |'.format(
            **line))
        for var in restdata.data_vars:
            vmin = restdata[var].where(weights > 0).min()
            vmax = restdata[var].where(weights > 0).max()
            if restdata[var].units == 'm3':
                vavg = (restdata[var].where(weights > 0)).mean()
            else:
                vavg = (restdata[var] * weights).sum()
            line = {
                'n': var,
                'u': restdata[var].units,
                'min': float(vmin),
                'avg': float(vavg),
                'max': float(vmax),
            }
            print('| {n:<15} | {u:<6} | {min:10.4g} {avg:10.4g} {max:10.4g} |'.
                  format(**line))

        # Add metadata
        restdata.attrs = {
            'title':
            'Restart data for HydroPy experiment ' + expid + ' for the ' +
            str(time.values),
            'institute': infos['institute'],
            'contact': infos['contact'],
            'version': infos['version'],
            'history':
            'Created ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Write to disk
        restdata.to_netcdf(rfile.replace('restart', expid + '_restart'))

        # Return for spinup runs
        return restdata


# ===================== Timestepfields class ===================================================
class timestepfields:
    def __init__(self, grid, fields=None):
        '''Initialize dataset for actual time step'''

        if fields is None:
            self.data = {}
        else:
            try:
                # Convert Dataset to numy dictionary
                self.data = {
                    x: fields[x].fillna(0).values
                    for x in fields.data_vars
                }
            except:
                self.data = {
                    x: fields[x].filled(0) * 1 if np.ma.is_masked(x) else fields[x] * 1
                    for x in fields
                }

        self.lsm = grid['lsm']

# ============================================================================================

    def update_forcing(self, forcing, var, time, delta_time=None):
        '''Get forcing for a given time step with missing values for ocean points'''

        # Select forcing for actual time step
        forcvar = forcing.sel(time=time)

        # Correct for negative values
        forcvar = forcvar.where(forcvar > 0, 0.0)

        # Apply land sea mask to forcing
        forcvar = forcvar.where(self.lsm > 0)

        # Multiply with time step length for fluxes
        if delta_time is not None:
            forcvar *= delta_time

        # Return as numpy array with missing values set to zero
        self.data[var] = forcvar.to_masked_array().filled(0)
