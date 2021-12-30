# Change log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.0.2] - 2021-04-30

### Changed
* made use of setup file optional
* changed setup file style and command line config options to json style
* changed forcing data definitions from hard-coded defaults to setup options
* processes are distributed into separate files according to general topic (land cover, hydrology, routing, ...)
* use netCDF interface to write simulation output timestep-wise instead of writing full chunks with xarray
* moved variable definitions into model class
* moved subtimestep loop for river routing into streamflow functions

### Fixed
* Outflow computation for lakes and groundwater in simulations without river routing
* Check for missing restart data fields
* removed missing data in restart fields

## [1.0.1] - 2021-03-23

### Added
* Change log file

### Changed
* Updated license statements
* Updated installation und setup instructions in readme file
