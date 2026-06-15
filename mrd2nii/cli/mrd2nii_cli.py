#!/usr/bin/python3
# -*- coding: utf-8 -*

import logging
import os

import click
import ismrmrd

from mrd2nii.mrd2nii_main import mrd2nii_dset

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--input', 'path_mrd', type=click.Path(), required=True,
              help="Input path to MRD folder/file")
@click.option('-o', '--output', 'path_output', type=click.Path(), required=True,
              help="Path to output folder")
@click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info',
              help="Be more verbose")
def mrd2nii_int(path_mrd, path_output, verbose):
    set_all_loggers(verbose)

    if not os.path.exists(path_mrd):
        raise FileNotFoundError(f"Output folder does not exist: {path_mrd}")

    if os.path.isdir(path_mrd):
        list_files = os.listdir(path_mrd)
    elif os.path.isfile(path_mrd):
        list_files = [path_mrd]
    else:
        raise ValueError(f"Input path is neither a folder nor a file: {path_mrd}")

    logger.info("Starting MRD to NIfTI conversion...")

    n_files_converted = 0
    for file in list_files:
        if not os.path.splitext(file)[1] in [".mrd", ".h5"]:
            continue

        n_files_converted += 1
        dset = ismrmrd.Dataset(os.path.join(path_mrd, file), dataset_name="dataset", create_if_needed=False)
        mrd2nii_dset(dset, path_output)
        dset.close()

    if n_files_converted <= 0:
        raise ValueError(f"No MRD files found in the input path: {path_mrd}")


def set_all_loggers(verbose, list_exclude=('matplotlib', 'indexed_gzip')):
    """ Set all loggers in the root manager to the verbosity level. Exclude any logger with the name in list_exclude

    Args:
        verbose (str): Verbosity level: 'info', 'debug', 'warning', 'critical', 'error'
        list_exclude: List of string to exclude from logging
    """
    loggers = []
    # For every logger name
    for name in logging.root.manager.loggerDict:

        # Exclude the setting level if it is in the excluded list
        is_excluded = False
        for exclude in list_exclude:
            if name.startswith(exclude):
                is_excluded = True

        if not is_excluded:
            loggers.append(logging.getLogger(name))

    for a_logger in loggers:
        a_logger.setLevel(verbose.upper())