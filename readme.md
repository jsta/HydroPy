## Synopsis

HydroPy is a global hydrological model combining hydrological land surface processes and river routing. It is based on the older Max Planck Institut for Meteorologys Hydrology Model (MPI-HM), however it is fully revised and written in Python. This model is actively developed and will get regular updates. Currently, its development is founded by the Helmholtz-Zentrum Hereon and the DAAD project The Ocean's Alkalinity.

## Motivation

While the older Fortran version of the MPI-HM is a very fast model, its development is hampered by an inflexible structure and a large overhead whenever new processes are implemented. Additionally, it only outputs service format data and time consuming post-processing is needed to produce netCDF files. The python version uses much more high-level routines and is strongly object-orienting enabling a much faster and easier development. HydroPy is the official successor of the MPI-HM.

## Installation
HydroPy requires a python 3.x environment with numpy, netCDF4, xarray and termcolor packages installed. The most convinient way would be to setup an  [anaconda installation](https://www.anaconda.com/products/individual) and create a new environment dedicated to the hydropy model, e.g.
```
    conda create -n hydropy numpy xarray netCDF4 termcolor
```
and switch to this environment using
```
    conda activate hydropy
```

## Run HydroPy
HydroPy is called from command line using the main function `./hydro.py`. Required options are the paths to a netCDF file containing meteorological forcing data and a json-style setup file containing the path of the land surface conditions file, and optionally further path information and model parameters:
```
    ./hydro.py -f forcing.nc -s setup.json
```

The forcing files has to provide the variables TSurf [K], Precip [kg m-2 s-1], and PET [kg m-2 s-1] at a temporal resolution of 1 day and a spatial resolution similar to the land surface conditions file.

A full list of all available options with either the default or user defined values (if set in the setup.json file or via command line) can be printed with
```
    ./hydro.py -p [-s setup.json] [-c '{"option": value, ...}']
```
Using this command also creates a file called `hydropy_options.json`, in which all options are stored. This can be used as a template to create a user-specific setup file and also provides an overview about all available settings.

The setup file must follow [json-style syntax](https://www.json.org/json-en.html), e.g.:
```
{
    "para": "hydropy_landsurface_parameter_v1.0.0.nc",
    "opt1" : val1,
    "opt2" : val2
}
```
Only those options need to be set that are different from the defaults. If options are provided via command line using the `-c` parameter, syntax has to follow json-style as well. Command line options precide options defined in the setup file.

Further options can be displayed with
```
    hydro.py --help
```


## Documentation and References

A documentation paper preprint is published in [GMDD](https://gmd.copernicus.org/preprints/gmd-2021-53/). A public version of the [HydroPy Model](https://doi.org/10.5281/zenodo.4541380) and an [example land surface parameter dataset at 0.5 degree](https://doi.org/10.5281/zenodo.4541238) can be found on Zenodo.

## Contributors

* Tobias Stacke (tobias.stacke@hereon.de)
* Stefan Hagemann (stefan.hagemann@hereon.de)

## License

HydroPy<br>
Copyright (C) 2021 Helmholtz-Zentrum Hereon
Copyright (C) 2020 Helmholtz-Zentrum Geesthacht

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
