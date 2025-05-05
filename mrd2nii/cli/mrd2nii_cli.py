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
              help="Input path to MRD folder")
@click.option('-o', '--output', 'path_output', type=click.Path(), required=True,
              help="Path to output folder")
# @click.option('-v', '--verbose', type=click.Choice(['info', 'debug']), default='info',
#               help="Be more verbose")
def mrd2nii_int(path_mrd, path_output):

    list_files = os.listdir(path_mrd)
    for file in list_files:
        if not os.path.splitext(file)[1] in [".mrd", ".h5"]:
            continue

        dset = ismrmrd.Dataset(os.path.join(path_mrd, file), dataset_name="dataset", create_if_needed=False)
        mrd2nii_dset(dset, path_output)
        dset.close()
