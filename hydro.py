#!/usr/bin/env python3
'''
FILENAME:
    hydro.py

DESCRIPTION:
    This is the driver script for the HydroPy global hydrology model, a fork
    of the MPI-HM.
    It organizes the I/O of forcing, boundary data, restart data and output.
    It also included the main time loop and calls the (formerly fortran) subroutines.

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
__version__ = '1.0.1'
__date__ = '2021/03/23'

# Load python functions
import numpy as np
import argparse as agp
import datetime as dt
import warnings as warn
import xarray as xr
from termcolor import colored
import sys
import os
import pdb

# Load HydroPy functions
import parameter_class as pac
import landcover_processes as lcp
import hydrology_processes as hydro
import dataio_class as dio
import analysis_class as anc
import utility_routines as utr

# ===================== Main HydroPy driver routine ===========================================
if __name__ == "__main__":

    # ==========================================================================================
    # *** Evaluate command line options ***
    # ==========================================================================================
    modeldescript = "HydroPy - global hydrological model written in python"
    parser = agp.ArgumentParser(description=modeldescript)
    parser.add_argument(
        '-c',
        '--config',
        action="append",
        nargs=1,
        dest="config",
        default=[],
        metavar='{"key": value}',
        help=
        "Overwrite default or setup file options using json style, can be called multiple times"
    )
    parser.add_argument('-d',
                        '--debug',
                        action='store_true',
                        dest="debug",
                        help="Enables additional debug output and warnings")
    parser.add_argument('-f',
                        '--forcing',
                        action='store',
                        dest="forcing_file",
                        help="Add specific forcing file",
                        default=None)
    parser.add_argument('-l',
                        '--logcells',
                        type=float,
                        action='append',
                        nargs=2,
                        dest="logcells",
                        default=[],
                        metavar=('LAT', 'LON'),
                        help="Log cell positions, " +
                        "can be called multiple times")
    parser.add_argument('-p',
                        '--print-options',
                        action='store_true',
                        dest="printopt",
                        help="Print global model options to screen and json file and exit")
    parser.add_argument('-r',
                        '--recompile',
                        action='store_true',
                        dest="recompile",
                        help="Recompile fortran shared library")
    parser.add_argument('-s',
                        '--setup',
                        action="store",
                        type=str,
                        dest="setup_file",
                        help="Path and name of the hydropy setup file *.json",
                        default=os.getcwd() + "/setup.json")
    parser.add_argument('-u',
                        '--spin-up',
                        action='store',
                        dest="spinup",
                        type=int,
                        default=0,
                        metavar='N',
                        help="Spin-up model for N simulation chunks or until states are stable" +
                        "(variation < 1 kg m-2)")
    parser.add_argument('-v',
                        '--version',
                        action='version',
                        version="HydroPy v" + __version__,
                        help="Print model version")
    arg = parser.parse_args()

    # ==========================================================================================
    # *** Initialize model and options ***
    # ==========================================================================================
    model = pac.parameter()

    # Get informations from the setup file
    model.update_from_ini(setupfile=arg.setup_file)

    # Update with options from command line
    if arg.config != []:
        model.update_from_cli(configlist=arg.config)

    # Output options
    if arg.printopt:
        print("\nHydroPy global model options\n")
        model.print_options()
        sys.exit(0)

    # Create some option shortcuts
    model.update_all(debug=arg.debug, spinup=arg.spinup)

    print("\nHydroPy simulation",model.expid.upper()," starting at",
          dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('HydroPy version:', model.opt['version'], "\n")

    if model.spinup:
        msg = "Spin-up mode active\nAll non chunk output disabled\n"
        print(colored(msg, 'red'))
        steplist = ['chunk']
    else:
        steplist = ['daily', 'monthly', 'chunk']

    if model.debug:
        print(colored("*** Debug mode active", 'red'))
        np.seterr(invalid='ignore')
        np.seterr(divide='ignore')
        warn.filterwarnings("error")
    else:
        warn.filterwarnings("ignore")

    if arg.recompile or not model.opt['use_fortran']:
        utr.clean_files()

    if model.opt['use_fortran']:
        utr.compile_fortran_subroutines('streamflow_fortran.f95')

    # ==========================================================================================
    # *** Read parameter fields for different model parts into one object ***
    # ==========================================================================================
    print("Reading parameter fields from " + model.opt['para'])
    model.get_parameter(parafile=model.opt['para'])

    # ==========================================================================================
    # *** Initialize forcing data ***
    # ==========================================================================================
    if arg.forcing_file is None:
        forcfile = model.opt['forcing'] + '/hydropy_forcing_' + str(
            model.opt['year']) + '.nc'
    else:
        forcfile = arg.forcing_file
    print("Reading forcing data stream from", forcfile)
    forcing = dio.forcing(filename=forcfile, mod=model)
    forcing.stream, model.param = utr.align_coords(
        forcing.stream,
        model.param,
        mesg="Mismatching coordinates between " +
        "parameter and forcing fields")

    # ==========================================================================================
    # *** Data consistency checks and derived variables ***
    # ==========================================================================================
    print("Initialize grid from forcing data")
    lsmvar=model.opt['lsmcheckvar']
    model.get_grid(griddata=forcing)
    model.get_lsm(forcdata=forcing.stream[lsmvar].isel(time=0, drop=True))
    model.update_parameter(model.grid)
    grid = model.grid

    # ==========================================================================================
    # *** Initialize log cell output if enabled ***
    # ==========================================================================================
    if arg.logcells != []:
        log_file = model.opt['output'] + '/cell_' + grid['timeid'] + '.log'
        model.log = anc.log_cells(lfile=log_file,
                                  logcells=arg.logcells,
                                  expid=model.expid,
                                  lsm=model.param.lsm,
                                  area=model.param.area,
                                  infos=model.opt)

    # ==========================================================================================
    # *** Setup variables streams ***
    # ==========================================================================================
    # Set output time steps for variables
    varstreams = {}
    outputstream = {}
    # Prepare output files
    for step in steplist:
        if model.opt[step] is not None:
            outfile = '_'.join([model.opt['output']+'/'+model.expid,step,forcing.chunkid + '.nc'])
            outtitle = 'HydroPy '+step+' output for experiment ' + model.expid + ', ' + grid['period']
            outputstream[step] = dio.output_stream(outfile, model, outtitle, step=step)
    # Read output time step requests
    for step in [x for x in steplist if model.opt[x] is not None]:
        for var in model.opt[step]:
            if var not in varstreams.keys():
                varstreams[var]=[]
            varstreams[var].append(outputstream[step])

    # Get restart information
    if model.opt['restdate'] is not None:
        # Restart date sanity check
        if model.opt['restdate'] > int(grid['restday']):
            msg = ' '.join(['\nModel restart date',str(model.opt['restdate']),
                'later than actual restart date --> Check setup file'])
            print(colored(msg, 'cyan'))
        if model.opt['restdate'] == int(grid['restday']):
            # use external restart file
            restfile = model.opt['restart']
    if 'restfile' not in globals():
        # search for restart file from last time period
        restfile = model.opt['input'] + '/' + model.expid + '_restart_' + grid[
            'restday'] + '.nc'

    # Define variables for fluxes
    stream_fluxes = dio.variables(
            gridinfo=grid, vartype="fluxes", spinup=model.spinup)
    fluxvars = model.get_fluxvars()
    for var in fluxvars:
        stream_fluxes.add_variable(name=var, longname=fluxvars[var]['long'],
                units=fluxvars[var]['unit'], stream=varstreams.get(var))

    # Define variables for states
    stream_states = dio.variables(
            gridinfo=grid, vartype="states", restart=restfile, spinup=model.spinup)
    statevars = model.get_statevars()
    for var in statevars:
        stream_states.add_variable(name=var, longname=statevars[var]['long'],
                units=statevars[var]['unit'], stream=varstreams.get(var),
                add_dim=statevars[var].get('add_dim'), restart=statevars[var]['rest'])

    # Define variables for land cover fractions
    stream_fcover = dio.variables(gridinfo=grid, vartype="fcover", spinup=model.spinup)
    covervars = model.get_covervars()
    for var in covervars:
        stream_fcover.add_variable(name=var, longname=covervars[var]['long'],
                units=covervars[var]['unit'], stream=varstreams.get(var))

    # ... and initialize states from restart or with zero
    stream_states.initialize(grid['landweights'])

    # ==========================================================================================
    # *** Check for spin-up request ***
    # ==========================================================================================
    if model.spinup:
        run_number = 0
        spinup = anc.spinup()
        spinup.add_states(states=stream_states.restart, cycle=run_number, infos=model.opt, 
                weights=grid['landweights'])

    # ==========================================================================================
    # *** Derive sekundary parameter fields or modify primary where necessary (move to model class at some time) ***
    # ==========================================================================================
    # Set permafrost scaling and soil parameters
    if model.with_permafrost:
        print('\nPermafrost scaling enabled')
        model.set_permafrost()
    else:
        print('\nPermafrost scaling disabled')
    model.set_soilparam()

    # Apply upper limit for soil moisture
    stream_states.restart['rootmoist'] = stream_states.restart[
        'rootmoist'].where(stream_states.restart.rootmoist <= model.param.wcap,
                           model.param.wcap)

    # Set land cover types
    landcovertypes = ['flake', 'fveg', 'fbare']

    # Construct routing information
    if model.with_rivers:
        print('River routing active')
    else:
        print('River routing disabled')
    model.get_flow_properties(routing=model.with_rivers)

    # ==========================================================================================
    # *** START SIMULATION ***
    # ==========================================================================================
    while True:

        # ==========================================================================================
        # *** Initialize water balance tracking ***
        # ==========================================================================================
        wb = anc.mass_balance(gridinfo=grid, btype='water')
        for s in ['swe', 'wliq', 'rootmoist', 'skinstor', 'canopystor']:
            wb.add_storage_start(storage=stream_states.restart[s].values)
        for s in ['lakestor', 'groundwstor']:
            wb.add_storage_start(storage=stream_states.restart[s].values)
        wb.add_storage_start(storage=stream_states.restart['riverstor'].sum(
            dim='res').values, conv2total=False)

        # ==========================================================================================
        # *** Time step loop ***
        # ==========================================================================================
        print("\n\nStarted: time loop started at",
              str(grid['time'][0].values).split('.')[0], '\n')
        for itime, time in enumerate(grid['time']):

            # Print status on every first of month or daily
            if ((grid['duration'] / np.timedelta64(1, 's') >= 365 * 24 * 3600
                 and str(time.values).split('-')[2][:2] == '01') or
                (grid['duration'] / np.timedelta64(1, 's') < 65 * 24 * 3600)):
                print("Current date:", str(time.values).split('.')[0])

            # Compute length of actual time steps in seconds
            if itime > 0:
                delta_time = (time - grid['time'][itime - 1]) / np.timedelta64(
                    1, 's')
            else:
                delta_time = (time - grid['resttime']) / np.timedelta64(1, 's')
            delta_time = float(delta_time.values)

            # Create temporary fields for actual time step which later are written
            # to the output streams
            fluxes = dio.timestepfields(grid)
            fcover = dio.timestepfields(grid)
            if itime == 0:
                # Initialize from restart
                laststates = stream_states.restart * 1
            else:
                # Initialize from last time step
                laststates = states.data
            states = dio.timestepfields(grid, fields=laststates)

            #                  ===============================
            #                  ||    Get actual forcing     ||
            #                  ===============================

            # Get forcing for actual time step and adapt to LSM
            for forcvar, forcinfo in model.opt['forcvars'].items():
                forcname, forcunit = forcinfo
                if 's-1' in forcunit:
                    # Convert fluxes to model time step length
                    fluxes.update_forcing(
                            forcing.stream[forcvar], forcname, time, delta_time)
                else:
                    # Use states as provided by forcing
                    states.update_forcing(
                            forcing.stream[forcvar], forcname, time)

            #                  ===============================
            #                  ||    Update landcover       ||
            #                  ===============================

            # *** Update land cover fractions from climatology to daily value
            lcp.get_daily_cover(model=model,
                                fcover=fcover.data,
                                ctypes=landcovertypes,
                                fluxes=fluxes.data,
                                states=states.data,
                                date=time)

            #                  ===============================
            #                  ||    Snow processes         ||
            #                  ===============================

            # *** Compute rain- and snowfall if not provided by forcing
            if 'rainf' in fluxes.data.keys() and 'snowf' in fluxes.data.keys():
                wb.add_source(source=fluxes.data['rainf'])
                wb.add_source(source=fluxes.data['snowf'])
            else:
                wb.add_source(source=fluxes.data['precip'])
                hydro.get_rain_and_snow(model=model, fluxes=fluxes.data, states=states.data)

            # *** Compute potential snow melt
            hydro.get_potential_snowmelt(model=model, fluxes=fluxes.data,
                                       states=states.data, time=time)

            # *** Update snow cover and compute throughfall
            hydro.update_snow(model=model, fluxes=fluxes.data, states=states.data)

            # *** Diagnose frozen ground
            lcp.diag_frozen_ground(model=model,
                                   states=states.data,
                                   fcover=fcover.data)

            #                  ===============================
            #                  || Skin and canopy processes ||
            #                  ===============================
            hydro.get_skinevap(model=model, fluxes=fluxes.data, states=states.data,
                             fcover=fcover.data, date=time)

            hydro.update_skincanopy(model=model, fluxes=fluxes.data, states=states.data,
                                  fcover=fcover.data)

            #                  ===============================
            #                  ||    Runoff processes       ||
            #                  ===============================

            # *** Compute surface runoff
            hydro.get_surface_runoff(model=model, fluxes=fluxes.data, states=states.data,
                                   fcover=fcover.data)

            # *** Compute subsurface drainage
            hydro.get_drainage(model=model, fluxes=fluxes.data, states=states.data,
                             fcover=fcover.data, dt=delta_time)

            # *** Compute lake leakage into ground
            hydro.get_lakeleak(model=model, fluxes=fluxes.data, states=states.data,
                             fcover=fcover.data)

            #                  ===============================
            #                  ||    Evaporation processes  ||
            #                  ===============================

            # *** Compute open water evaporation for lakes
            hydro.get_lakeevap(model=model, fluxes=fluxes.data, states=states.data,
                             fcover=fcover.data)

            # *** Compute cell average plant transpiration
            hydro.get_transpiration(model=model, fluxes=fluxes.data, states=states.data,
                                  fcover=fcover.data)

            # *** Compute soil evaporation for bare soil part
            hydro.get_soilevap(model=model, fluxes=fluxes.data, states=states.data,
                               fcover=fcover.data)

            #                  ===============================
            #                  ||    Water storage updates  ||
            #                  ===============================

            # *** Update soil
            hydro.update_soil(model=model, fluxes=fluxes.data, states=states.data)

            # *** Update surface water storage
            hydro.update_surface_water(model=model, fluxes=fluxes.data, states=states.data,
                                       fcover=fcover.data)

            # *** Update groundwater storage
            hydro.update_groundwater(model=model, fluxes=fluxes.data, states=states.data)

            #                  ===============================
            #                  ||    River routing          ||
            #                  ===============================

            if model.with_rivers:
                # *** Update river storage
                hydro.update_riverflow(model=model, fluxes=fluxes.data, states=states.data, dtime=delta_time)
                wb.add_sink_latflow(latsinks=(fluxes.data['dis'] - fluxes.data['rivdis']), conv2total=False)
                wb.add_sink_2ocean(ocsinks=(fluxes.data['freshwater'] + model.temporary['riv_ts_corr']), conv2total=False)

            else:
                # *** Track water outflow flows without rivers
                wb.add_sink(sink=fluxes.data['qsl'])
                wb.add_sink(sink=fluxes.data['qg'])


            # *** Diagnose total fluxes of a given type
            #     and add to water balance if appropriate
            # Total Evaporation
            fluxes.data['evap'] = (
                    fluxes.data['transp']
                    + fluxes.data['sevap']
                    + fluxes.data['levap']
                    + fluxes.data['skinevap']
                    + fluxes.data['canoevap'])
            # Total Runoff
            fluxes.data['qtot'] = (
                    fluxes.data['qs']
                    + fluxes.data['qsb'])
            # Total TWS
            states.data['tws'] = (states.data['swe']
                    + states.data['wliq']
                    + states.data['rootmoist']
                    + states.data['skinstor']
                    + states.data['canopystor']
                    + states.data['lakestor']
                    + states.data['groundwstor']
                    + states.data['riverstor'].sum(axis=0) / model.grid['landarea'] * 1000 
                    )
            # Combine river discharge and ocean inflow
            if model.with_rivers:
                fluxes.data['dis'] += fluxes.data['freshwater']

            wb.add_sink(sink=fluxes.data['evap'])

            #                  ===============================
            #                  || Output and Logfile stuff  ||
            #                  ===============================

            # Write output to stream variable
            stream_fluxes.write2stream(streams=varstreams,
                                       date=time, step=itime,
                                       fields=fluxes.data,
                                       area=grid['landarea'],
                                       dtime=delta_time,
                                       flux=True)
            stream_states.write2stream(streams=varstreams,
                                       date=time, step=itime,
                                       fields=states.data,
                                       area=grid['landarea'],
                                       dtime=delta_time)
            stream_fcover.write2stream(streams=varstreams,
                                       date=time, step=itime,
                                       fields=fcover.data,
                                       area=grid['landarea'],
                                       dtime=delta_time)

        print("\nFinished: time loop ended at", time.values, '\n')

        # ==========================================================================================
        # Calculate grid cell balances and write data to file
        # ==========================================================================================
        if 'log' in vars(model).keys():
            model.log.calc_balance(
                    tasklist=['snow', 'skin', 'soil', 'lake', 'groundwater',
                        'river', 'full'], rivers=model.with_rivers)
            model.log.write_logfile(timeaxis=grid['time'], expid=model.expid)

        # ==========================================================================================
        # *** Evaluate global water balance ***
        # ==========================================================================================

        for s in ['swe', 'wliq', 'rootmoist', 'skinstor', 'canopystor']:
            wb.add_storage_end(storage=states.data[s])
        for s in ['lakestor', 'groundwstor']:
            wb.add_storage_end(storage=states.data[s])
        wb.add_storage_end(storage=states.data['riverstor'].sum(axis=0), conv2total=False)
        wb.check_out(chunk=forcing.chunkid, time=grid['time'], debug=model.debug, infos=model.opt)

        # ==========================================================================================
        # *** Write restart data ***
        # ==========================================================================================
        restdate = str(time.values).split('T')[0].replace('-', '')
        restfile = model.opt['input'] + '/restart_' + restdate + '.nc'
        stream_states.restart = stream_states.write_restart(
                restfile, states.data, time, model.expid, model.opt, grid['landweights'])

        # ==========================================================================================
        # *** Repeat simulations for specific model setups e.g. Spin-up ***
        # ==========================================================================================
        if model.spinup:
            run_number += 1
            spinup.add_states(states=stream_states.restart, cycle=run_number, infos=model.opt,
                    weights=grid['landweights'])
            equilibrium = spinup.evaluate(run_number)
            if equilibrium or run_number >= arg.spinup:
                break
        else:
            break


    # ==========================================================================================
    # *** Finished simulation ***
    # ==========================================================================================
    forcing.close()

    for outstep in outputstream:
        outputstream[outstep].close()

    print("\nHydroPy simulation",model.expid.upper(),"finished at",
          dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "\n")
