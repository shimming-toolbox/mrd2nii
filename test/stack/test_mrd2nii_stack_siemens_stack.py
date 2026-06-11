#!/usr/bin/python3
# -*- coding: utf-8 -*

import ismrmrd
import nibabel as nib
import numpy as np
import os

from mrd2nii.mrd2nii_main import mrd2nii_stack
from mrd2nii.nii_utils import orient_nii_to
from mrd2nii import __dir_testing__


def test_mrd2nii_stack_tra_ap_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_AP_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_TRA_AP_2025-09-26-182929_64.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 2.60416667, 0., -135.31448015],
                       [0., 0., 15., 3.09810638],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "10_dicoms_ep2d_bold_ST_TRA_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 7, 0], atol=1)


def test_mrd2nii_stack_tra_pa_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_PA_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_TRA_PA_2025-09-26-183117_97.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 2.60416667, 0., -135.31448015],
                       [0., 0., 15., 3.09810638],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "13_dicoms_ep2d_bold_ST_TRA_PA_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 7, 0], atol=1)


def test_mrd2nii_stack_tra_rl_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_RL_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_TRA_RL_2025-09-26-183343_59.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 2.60416667, 0., -135.31448015],
                       [0., 0., 15., 3.09810638],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "16_dicoms_ep2d_bold_ST_TRA_RL_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 7, 0], atol=1)


def test_mrd2nii_stack_tra_lr_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_LR_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_TRA_LR_2025-09-26-183409_86.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 2.60416667, 0., -135.31448015],
                       [0., 0., 15., 3.09810638],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "17_dicoms_ep2d_bold_ST_TRA_LR_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 7, 0], atol=1)


def test_mrd2nii_stack_tra_rot_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_ROT_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_TRA_AP_ROT_2025-09-26-183736_26.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.51815322e+00, -7.96357465e-08, 3.82330562e+00, 1.30429621e+02],
                       [1.19199037e-01, 2.56183199e+00, 2.60472061e+00, -1.32534504e+02],
                       [6.52977790e-01, -4.67654662e-01, 1.42687689e+01, -8.09314910e+00],
                       [0.00000000e+00, -0.00000000e+00, -0.00000000e+00, 1.00000000e+00]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(
        os.path.join(path_dataset, "nii", "22_dicoms_ep2d_bold_ST_TRA_AP_ROT_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 7, 0], atol=1)


def test_mrd2nii_stack_tra_asc():
    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_ASC")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_TRA_asc_2025-09-26-185707_77.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 2.60416667, 0., -135.31448015],
                       [-0., -0., 15., -56.90189552],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(
        os.path.join(path_dataset, "nii", "31_dicoms_ep2d_bold_ST_TRA_asc_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 3, 0], atol=1)


def test_mrd2nii_stack_tra_desc():
    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_DESC")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_TRA_desc_2025-09-26-185733_37.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 2.60416667, 0., -135.31448015],
                       [-0., -0., 15., -11.90189552],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(
        os.path.join(path_dataset, "nii", "32_dicoms_ep2d_bold_ST_TRA_desc_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 6, 0], atol=1)


def test_mrd2nii_stack_sag_ap_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_AP_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_SAG_AP_2025-09-26-183020_76.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[0., 0., 15., -37.5],
                       [-2.60416667, 0., 0., 112.08135319],
                       [0., 2.60416667, 0., -156.79772823],
                       [0., 0., 0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "11_dicoms_ep2d_bold_ST_SAG_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[:, :, 0], nii_expected.get_fdata()[:, :, 2, 0], atol=1)


def test_mrd2nii_stack_sag_pa_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_PA_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_SAG_PA_2025-09-26-183238_13.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[0., 0., 15., -37.5],
                       [-2.60416667, 0., 0., 112.08135319],
                       [0., 2.60416667, 0., -156.79772823],
                       [0., 0., 0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "14_dicoms_ep2d_bold_ST_SAG_PA_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[:, :, 0], nii_expected.get_fdata()[:, :, 2, 0], atol=1)


def test_mrd2nii_stack_sag_hf_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_HF_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_SAG_HF_2025-09-26-183435_64.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-1.05181454e-27, 0.00000000e+00, 1.50000000e+01, -3.75000000e+01],
                       [-2.60416667e+00, -1.32901356e-11, -6.05845175e-27, 1.12081353e+02],
                       [-1.32901356e-11, 2.60416667e+00, -3.09187757e-38, -1.56797728e+02],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "18_dicoms_ep2d_bold_ST_SAG_HF_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[:, :, 0], nii_expected.get_fdata()[:, :, 2, 0], atol=1)


def test_mrd2nii_stack_sag_fh_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_FH_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_SAG_FH_2025-09-26-183500_43.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[1.05181454e-27, 0.00000000e+00, 1.50000000e+01, -3.75000000e+01],
                       [-2.60416667e+00, -1.27515335e-11, 6.05845175e-27, 1.12081353e+02],
                       [-1.27515335e-11, 2.60416667e+00, 2.96657474e-38, -1.56797728e+02],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "19_dicoms_ep2d_bold_ST_SAG_FH_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[:, :, 0], nii_expected.get_fdata()[:, :, 2, 0], atol=1)


def test_mrd2nii_stack_sag_rot_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_ROT_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_SAG_AP_ROT_2025-09-26-183904_78.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-4.67654746e-01, -6.52978418e-01, 1.42687678e+01, 1.74655003e+01],
                       [-2.56183197e+00, 1.19199299e-01, -2.60472071e+00, 1.10958721e+02],
                       [4.77438359e-08, 2.51815304e+00, 3.82330945e+00, -1.62313353e+02],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(
        os.path.join(path_dataset, "nii", "23_dicoms_ep2d_bold_ST_SAG_AP_ROT_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[:, :, 0], nii_expected.get_fdata()[:, :, 2, 0], atol=1)


def test_mrd2nii_stack_sag_asc():
    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_ASC")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_SAG_asc_2025-09-26-190149_28.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[0., 0., 15., 22.5],
                       [-2.60416667, 0., 0., 112.08135319],
                       [0., 2.60416667, 0., -156.79772823],
                       [0., 0., 0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(
        os.path.join(path_dataset, "nii", "33_dicoms_ep2d_bold_ST_SAG_asc_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[:, :, 0], nii_expected.get_fdata()[:, :, 6, 0], atol=1)


def test_mrd2nii_stack_sag_desc():
    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_DESC")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_SAG_desc_2025-09-26-190214_60.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[0., 0., 15., -22.5],
                       [-2.60416667, 0., 0., 112.08135319],
                       [0., 2.60416667, 0., -156.79772823],
                       [0., 0., 0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(
        os.path.join(path_dataset, "nii", "34_dicoms_ep2d_bold_ST_SAG_desc_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[:, :, 0], nii_expected.get_fdata()[:, :, 3, 0], atol=1)


def test_mrd2nii_stack_cor_rl_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_RL_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_COR_RL_2025-09-26-183046_72.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 0., -15., -50.41864777],
                       [0., 2.60416667, 0., -156.79772823],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "012_dicoms_ep2d_bold_ST_COR_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 7, 0], atol=1)


def test_mrd2nii_stack_cor_lr_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_LR_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_COR_LR_2025-09-26-183304_92.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 0., -15., -50.41864777],
                       [0., 2.60416667, 0., -156.79772823],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "015_dicoms_ep2d_bold_ST_COR_LR_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 7, 0], atol=1)


def test_mrd2nii_stack_cor_hf_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_HF_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_COR_HF_2025-09-26-183544_3.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 0., -15., -50.41864777],
                       [0., 2.60416667, 0., -156.79772823],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "20_dicoms_ep2d_bold_ST_COR_HF_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 7, 0], atol=1)


def test_mrd2nii_stack_cor_fh_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_FH_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_COR_FH_2025-09-26-183610_77.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 0., -15., -50.41864777],
                       [0., 2.60416667, 0., -156.79772823],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(os.path.join(path_dataset, "nii", "21_dicoms_ep2d_bold_ST_COR_FH_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 7, 0], atol=1)


def test_mrd2nii_stack_cor_rot_int():
    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_ROT_INT")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_COR_RL_ROT_2025-09-26-183937_71.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.56183198e+00, 1.19198985e-01, 2.60472089e+00, 1.23877383e+02],
                       [-4.67654696e-01, -6.52977064e-01, -1.42687699e+01, 4.54678648e+00],
                       [-7.43580494e-09, 2.51815341e+00, -3.82330145e+00, -1.62313367e+02],
                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(
        os.path.join(path_dataset, "nii", "24_dicoms_ep2d_bold_ST_COR_RL_ROT_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 7, 0], atol=1)


def test_mrd2nii_stack_cor_asc():
    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_ASC")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_COR_asc_2025-09-26-190246_92.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 0., -15., 9.58135319],
                       [0., 2.60416667, 0., -156.79772823],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(
        os.path.join(path_dataset, "nii", "35_dicoms_ep2d_bold_ST_COR_asc_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 3, 0], atol=1)


def test_mrd2nii_stack_cor_desc():
    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_DESC")
    path_mrd = os.path.join(path_dataset, "mrd", "ep2d_bold_ST_COR_desc_2025-09-26-190312_41.h5")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    fname_output = os.path.join(path_output, "single_slice.nii.gz")

    if os.path.exists(fname_output):
        os.remove(fname_output)

    dset = ismrmrd.Dataset(path_mrd, dataset_name="dataset", create_if_needed=False)
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    image = dset.read_image("images_0", 3)
    nii = mrd2nii_stack(metadata, image, include_slice_gap=True)
    nib.save(nii, fname_output)
    expected_affine = [[-2.60416667, 0., 0., 125.],
                       [0., 0., -15., -35.41864777],
                       [0., 2.60416667, 0., -156.79772823],
                       [0., -0., -0., 1.]]
    assert np.allclose(nii.affine, expected_affine)
    nii_expected = nib.load(
        os.path.join(path_dataset, "nii", "36_dicoms_ep2d_bold_ST_COR_desc_20250926122558.nii.gz"))
    nii_reorient = (orient_nii_to(nii, nii_expected.header.get_dim_info()))
    assert np.allclose(nii_reorient.get_fdata()[..., 0], nii_expected.get_fdata()[:, :, 6, 0], atol=1)
