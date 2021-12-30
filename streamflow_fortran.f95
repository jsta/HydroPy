! FILENAME:
!     streamflow_fortran.f95
! 
! DESCRIPTION:
!     This file contains a collection of functions for the HydroPy Model
!     written in fortran. It is used for production runs with HydroPy
!     requiring a higher performance.
!     Results provided by this subrouintes must always be identical to
!     results computed by routing.py
! 
! AUTHOR:
!     Tobias Stacke
! 
! Copyright (C):
!     2021 Helmholtz-Zentrum Hereon
!     2020 Helmholtz-Zentrum Geesthacht
! 
! LICENSE:
!     This program is free software: you can redistribute it and/or modify it under the
!     terms of the GNU General Public License as published by the Free Software Foundation,
!     either version 3 of the License, or (at your option) any later version.
! 
!     This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
!     without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
!     See the GNU General Public License for more details.
! 
!     You should have received a copy of the GNU General Public License along with this program.
!     If not, see http://www.gnu.org/licenses/.
! 
! __author__ = 'Tobias Stacke'
! __copyright__ = 'Copyright (C) 2021 Helmholtz-Zentrum Hereon, 2020 Helmholtz-Zentrum Geesthacht'
! __license__ = 'GPLv3'

! ======================================================================================================
subroutine eval_flowfield(rout_lat, rout_lon, area, topo, pi, nlat, nlon, &
                          riverflow, flowsinks, ic, dx, dh)
    implicit none
    ! Generates list of upstream and downstream grid cell indices for
    ! river routing

    integer, parameter :: dp = selected_real_kind(15, 307)

    ! Interface variables
    integer,  intent(in)  :: nlat, nlon
    real(dp), intent(in)  :: rout_lat(nlat, nlon)
    real(dp), intent(in)  :: rout_lon(nlat, nlon)
    real(dp), intent(in)  :: area(nlat, nlon)
    real(dp), intent(in)  :: topo(nlat, nlon)
    real(dp), intent(in)  :: pi
    integer,  intent(out) :: ic
    integer,  intent(out) :: riverflow(nlat * nlon, 4)
    integer,  intent(out) :: flowsinks(nlat, nlon)
    real(dp), intent(out) :: dx(nlat, nlon)
    real(dp), intent(out) :: dh(nlat, nlon)

    ! Local variables
    integer :: iy, ix, trg_y, trg_x, py, px
    real(dp), parameter :: r_earth = 6371000
    real(dp) :: dx_lat
    real(dp) :: dx_lon(nlat)
    real(dp) :: dx_x, dx_y

    ic = 0
    riverflow(:,:) = 0
    flowsinks(:,:) = 1
    dx(:,:) = 2.0_dp * (area(:,:) / pi)**(0.5_dp)
    dh(:,:) = 0.0_dp

    ! Get meridonal and zonal distance fields
    dx_lat = (pi * r_earth) / dble(nlat)
    dx_lon(:) = area(:,1) / dx_lat

    do iy = 1, nlat
    do ix = 1, nlon

    ! Attention: as fortran fields start at 1 and python start at 0,
    ! 1 has to be substracted from iy, ix and ic
    ! Define source and target indices for python adress range
    py = iy - 1
    px = ix - 1
    trg_y = int(rout_lat(iy, ix))
    trg_x = int(rout_lon(iy, ix))
    if (py /= trg_y .or. px /= trg_x) then
        ic = ic + 1
        riverflow(ic, :)  = (/py, px, trg_y, trg_x/)
        flowsinks(iy, ix) = 0
        dx_x = 0.5_dp * (dx_lon(iy) + dx_lon(trg_y+1)) * (trg_x + 1 - ix)
        dx_y = dx_lat * (trg_y + 1 - iy)
        dx(iy,ix) = sqrt(dx_x**2 + dx_y**2)
        dh(iy,ix) = max(0.1_dp, max(0.0_dp, topo(iy,ix)) - max(0.0_dp, topo(trg_y+1,trg_x+1)))
    end if

    end do
    end do

    ! Substract 1 for python indexing
    ic = ic - 1

    ! Print output to demonstrate routing_fortran.f95 is used
    write(*,*) '  *** Routing: Fortran subroutines are used'

end subroutine eval_flowfield


! ======================================================================================================
subroutine river_routing(upstream, sinks, flowcells, ncells, nlat, nlon, downstream, rout_err)
    implicit none
    ! lateral transport of fluxes between grid cells'''

    integer, parameter :: dp = selected_real_kind(15, 307)

    ! Interface variables
    integer,  intent(in)  :: ncells, nlat, nlon
    integer,  intent(in)  :: sinks(nlat, nlon)
    integer,  intent(in)  :: flowcells(ncells,4)
    real(dp), intent(in)  :: upstream(nlat, nlon)
    real(dp), intent(out) :: downstream(nlat, nlon)
    real(dp), intent(out) :: rout_err

    ! Local variables
    integer :: icell
    integer :: us_y, us_x, ds_y, ds_x

    where(sinks(:,:) > 0.5)
        downstream(:,:) = upstream(:,:)
    elsewhere
        downstream(:,:) = 0.0
    endwhere

    do icell = 1, ncells
        us_y = flowcells(icell,1) + 1
        us_x = flowcells(icell,2) + 1
        ds_y = flowcells(icell,3) + 1
        ds_x = flowcells(icell,4) + 1
        downstream(ds_y, ds_x) = downstream(ds_y, ds_x) + upstream(us_y, us_x)
    end do

    ! Debug checks
    rout_err = abs(sum(upstream) - sum(downstream))

end subroutine river_routing


! ======================================================================================================
subroutine linear_cascade(inflow, reservoir, ncasc, lag, substeps, nres, nlon, nlat, res_new, act_outfl)
    implicit none
    ! Linear reservoir cascade used for lake, groundwater and
    ! river reservoirs

    integer, parameter :: dp = selected_real_kind(15, 307)

    ! Interface variables
    integer,  intent(in)  :: nres, nlat, nlon, substeps
    real(dp), intent(in)  :: inflow(nlat, nlon)
    real(dp), intent(in)  :: ncasc(nlat, nlon)
    real(dp), intent(in)  :: lag(nlat, nlon)
    real(dp), intent(in)  :: reservoir(nres, nlat, nlon)
    real(dp), intent(out) :: res_new(nres, nlat, nlon)
    real(dp), intent(out) :: act_outfl(nlat, nlon)

    ! Local variables
    integer  :: icas, max_casc
    real(dp) :: div_sub
    real(dp) :: lagtime(nlat, nlon)
    real(dp) :: act_infl(nlat, nlon)

    ! Adapt lagtime and get maximum cascade number
    div_sub = 1.0 / float(substeps)
    max_casc = int(maxval(ncasc(:,:)))
    lagtime(:,:)   = 1.0 / (lag(:,:) + div_sub)
    res_new(:,:,:) = reservoir(:,:,:)
    act_infl(:,:)  = inflow(:,:)

    ! Linear reservoir cascade and scale fluxes with substep time step
    do icas = 1, max_casc
        where(ncasc(:,:) >= icas - epsilon(1.0))
            res_new(icas,:,:) = res_new(icas,:,:) + act_infl(:,:)
            act_outfl(:,:)    = res_new(icas,:,:) * lagtime(:,:) * div_sub
            res_new(icas,:,:) = res_new(icas,:,:) - act_outfl(:,:)
        elsewhere
            act_outfl(:,:) = act_infl(:,:)
        endwhere
        act_infl(:,:) = act_outfl(:,:)
    end do

end subroutine linear_cascade


! ======================================================================================================
subroutine routing_cascade(local_infl, upstream_infl, reservoir_start, lag, ncasc, landarea, &
        sinks, flowcells, substeps, ncells, nres, nlon, nlat, &
        actflow_in, accflow_in, accflow_out, outlets, reservoir, routerr)
    implicit none
    ! returns combined sub-timestep flow cascade and routing results

    integer, parameter :: dp = selected_real_kind(15, 307)

    ! Interface variables
    integer,  intent(in)  :: ncells, nres, nlat, nlon, substeps
    integer,  intent(in)  :: flowcells(ncells,4)
    integer,  intent(in)  :: sinks(nlat, nlon)
    real(dp), intent(in)  :: local_infl(nlat, nlon)
    real(dp), intent(in)  :: upstream_infl(nlat, nlon)
    real(dp), intent(in)  :: reservoir_start(nres, nlat, nlon)
    real(dp), intent(in)  :: lag(nlat, nlon)
    real(dp), intent(in)  :: ncasc(nlat, nlon)
    real(dp), intent(in)  :: landarea(nlat, nlon)
    real(dp), intent(out) :: actflow_in(nlat, nlon)
    real(dp), intent(out) :: accflow_in(nlat, nlon)
    real(dp), intent(out) :: accflow_out(nlat, nlon)
    real(dp), intent(out) :: outlets(nlat, nlon)
    real(dp), intent(out) :: reservoir(nres, nlat, nlon)
    real(dp), intent(out) :: routerr

    ! Local variables
    integer  :: sub
    real(dp) :: div_sub, err
    real(dp) :: local_infl_vol(nlat, nlon)
    real(dp) :: storage(nres, nlat, nlon)
    real(dp) :: outflow(nlat, nlon)
    real(dp) :: upstream(nlat, nlon)

    ! setup fields
    routerr          = 0._dp
    accflow_in(:,:)  = 0._dp
    accflow_out(:,:) = 0._dp
    outlets(:,:)     = 0._dp
    reservoir(:,:,:) = reservoir_start(:,:,:)

    ! Convert variables to volume and subtimestep
    div_sub         = 1._dp / dble(substeps)
    actflow_in(:,:) = upstream_infl(:,:) * div_sub
    where (local_infl(:,:) /= local_infl(:,:))
        local_infl_vol(:,:) = 0._dp
    elsewhere
        local_infl_vol(:,:) = local_infl(:,:) * landarea(:,:) * 0.001_dp
    endwhere

    ! Walk through linear cascade for all subtimesteps
    do sub = 1, substeps
        ! Compute inflow for actual subtimestep
        accflow_in(:,:) = accflow_in(:,:) + actflow_in(:,:)
        ! Compute storage outflow from inflow and states
        call linear_cascade(actflow_in, reservoir, ncasc, lag, substeps, &
                            nres, nlat, nlon, storage, outflow)
        upstream(:,:) = outflow(:,:) + local_infl_vol(:,:) * div_sub
        ! Rout river discharge to downstream grid cell
        where (upstream(:,:) /= upstream(:,:))
            upstream(:,:) = 0._dp
        elsewhere
            upstream(:,:) = upstream(:,:)
        endwhere
        call river_routing(upstream, sinks, flowcells, ncells, nlat, nlon, actflow_in, err)
        routerr = routerr + err

        ! Update storages
        accflow_out(:,:) = accflow_out(:,:) + upstream(:,:)
        reservoir(:,:,:) = storage(:,:,:)
        ! Collect ocean inflow and substract from cell inflow
        where (sinks(:,:) > 0.5)
            outlets(:,:)    = outlets(:,:) + actflow_in(:,:)
            actflow_in(:,:) = 0._dp
        elsewhere
            actflow_in(:,:) = actflow_in(:,:)
            outlets(:,:)    = 0._dp
        endwhere
    end do

    actflow_in(:,:) = actflow_in(:,:) * substeps

end subroutine routing_cascade
