# -*- coding: utf-8 -*-
'''
FILENAME:
    streamflow.py

DESCRIPTION:
    This file contains all functions used to actually compute water flows and storages.
    It uses only numpy arrays and should be subject to further optimization.

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
from itertools import product


# ======================================================================================================
def eval_flowfield(rout_lat, rout_lon, area, topo, pi):
    '''return list of upstream and downstream grid cells, together with fields of
       distance and height difference
    '''

    r_earth = 6371000.0
    nlat, nlon = area.shape
    flowsinks = np.ones((nlat, nlon))
    dx_lat = (pi * r_earth) / nlat
    dx_lon = area[:,1] / dx_lat
    dx = 2.0 * (area / pi)**(0.5)
    dh = np.zeros((nlat, nlon))
    riverflow = np.zeros((nlat * nlon, 4), dtype=int)
    ic = -1

    for iy, ix in product(np.arange(nlat), np.arange(nlon)):
        trg_y = int(rout_lat[iy, ix])
        trg_x = int(rout_lon[iy, ix])
        if iy != trg_y or ix != trg_x:
            ic += 1
            riverflow[ic, :] = np.array([iy, ix, trg_y, trg_x])
            flowsinks[iy, ix] = 0
            dx_x = 0.5 * (dx_lon[iy] + dx_lon[trg_y]) * (trg_x - ix)
            dx_y = dx_lat * (trg_y - iy)
            dx[iy,ix] = (dx_x**2 + dx_y**2)**(0.5)
            dh[iy,ix] = max(0.1, max(0, topo[iy,ix]) - max(0, topo[trg_y,trg_x]))

    # Print output to demonstrate processes.py is used
    print('   *** Optimization: Pure python code is used')

    return riverflow, flowsinks, ic, dx, dh


# ======================================================================================================
def linear_cascade(inflow, reservoir, ncasc, lag, substeps=1):
    '''Linear reservoir cascade used for lake, groundwater and
     river reservoirs'''

    # Adapt lagtime and get maximum cascade number
    div_sub = 1.0 / float(substeps)
    lagtime = 1.0 / (lag + div_sub)
    max_casc = int(ncasc.max())
    rsrvr = reservoir * 1

    # Linear reservoir cascade and scale fluxes with substep time step
    for nc in range(max_casc):
        active = ncasc >= nc + 1
        rsrvr[nc] = np.where(active, rsrvr[nc] + inflow, rsrvr[nc])
        act_outfl = np.where(active, rsrvr[nc] * lagtime * div_sub, inflow)
        rsrvr[nc] = np.where(active, rsrvr[nc] - act_outfl, rsrvr[nc])
        inflow = act_outfl * 1

    # Return outflow and reservoir state
    return rsrvr, act_outfl


# ======================================================================================================
def river_routing(upstream, sinks, flowcells, ncells):
    '''lateral transport of fluxes between grid cells'''
    downstream = np.where(sinks > 0.5, upstream, 0)

    for cells in flowcells:
        us_y, us_x = cells[0], cells[1]
        ds_y, ds_x = cells[2], cells[3]
        downstream[ds_y, ds_x] += upstream[us_y, us_x] * 1

    # Debug checks
    rout_err = abs(upstream.sum() - downstream.sum())

    return downstream, rout_err


# ======================================================================================================
def routing_cascade(local_infl, upstream_infl, reservoir, lag, ncasc, landarea, sinks, flowcells, substeps):
    '''returns combined sub-timestep flow cascade and routing results'''

    # setup fields
    routerr = 0
    accflow_in = np.zeros_like(local_infl)
    accflow_out = np.zeros_like(local_infl)
    outlets = np.zeros_like(local_infl)

    # Convert variables to volume and subtimestep
    f_subfl = 1.0 / substeps
    actflow_in = upstream_infl * f_subfl
    local_infl_vol = np.nan_to_num(local_infl) * landarea * 0.001

    # Walk through linear cascade for all subtimesteps
    for sub in range(substeps):
        # Compute inflow for actual subtimestep
        accflow_in += actflow_in
        # Compute storage outflow from inflow and states
        storage, outflow = linear_cascade(actflow_in,
                reservoir, ncasc, lag, substeps=substeps)
        upstream = outflow + (local_infl_vol) * f_subfl
        # Rout river discharge to downstream grid cell
        if np.any(np.isnan(upstream)):
            upstream = np.nan_to_num(upstream)
        actflow_in, err = river_routing(
                upstream, sinks, flowcells, len(flowcells))
        routerr += err

        # Update storages
        accflow_out += upstream
        reservoir = storage * 1
        # Collect ocean inflow and substract from cell inflow
        outlets += np.where(sinks > 0.5, actflow_in, 0)
        actflow_in = np.where(sinks < 0.5, actflow_in, 0)

    return actflow_in * substeps, accflow_in, accflow_out, outlets, reservoir, err
