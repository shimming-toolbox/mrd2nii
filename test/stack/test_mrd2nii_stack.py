#!/usr/bin/python3
# -*- coding: utf-8 -*

import ismrmrd
import nibabel as nib
import numpy as np
import os

from mrd2nii.mrd2nii_main import mrd2nii_stack
from mrd2nii.nii_utils import orient_nii_to
from mrd2nii import __dir_testing__


def test_mrd2nii_stack():
    path_mrd = os.path.join(__dir_testing__, "dset1", "mrd", "ep2d_bold_shimming_2025-04-15-160334_93.h5")
    path_output = os.path.join(__dir_testing__, "dset1", "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("image_0", 1)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, -0., 0., 130.62679434],
                       [-0., 2.60416667, -0., -128.96042665],
                       [-0., 0.,   15., -11.24641132],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)


def test_mrd2nii_stack2():
    path_mrd = os.path.join(__dir_testing__, "dset7", "mrd", "T1w_2025-08-27-163253_63.h5")
    path_output = os.path.join(__dir_testing__, "dset7", "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("image_0", 50)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[0, 0., 1., 45.5],
                       [0., 1, 0., -116.39277077],
                       [1., 0., 0, -163.78072289],
                       [0., -0., 0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
