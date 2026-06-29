
# Mrd2nii

## Overview
`Mrd2nii` converts ISMRMRD data into NIfTI images with BIDS metadata.

## Features
- Convert spatial images
- Convert raw data (currently not supported)
- Convert waveforms (currently only respiratory data is supported)

## Getting started
- Clone this repository and navigate to the project directory.

```bash
git clone https://github.com/shimming-toolbox/mrd2nii.git
cd mrd2nii
```

- Create a virtual environment or use an existing one.

```bash
python -m venv mrd2nii_venv
source mrd2nii_venv/bin/activate  # On Windows: mrd2nii_venv\Scripts\activate
```

- Install mrd2nii in the virtual environment.
```bash
pip install .
```

## Usage
On the command line, run the following command to display the full list of options

```bash
mrd2nii -h

> Usage: mrd2nii [OPTIONS]
> 
> Options:
>   -i, --input PATH   Input path to MRD folder/file  [required]
>   -o, --output PATH  Path to output folder  [required]
>   -h, --help         Show this message and exit.
```

If you run into any issues, please report them in the [GitHub Issues](https://github.com/shimming-toolbox/mrd2nii/issues)

## Development
Download the full suite of tests:

```bash
TODO
```

Run tests and checks from the project root.

```bash
pytest
```
