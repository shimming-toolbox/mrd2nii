#!/usr/bin/python3
# -*- coding: utf-8 -*

from click.testing import CliRunner
import nibabel as nib
import numpy as np
import os
import shutil

from mrd2nii.cli.mrd2nii_cli import mrd2nii_int
from mrd2nii import __dir_testing__

from test.test_utils import verify_sidecar


def test_mrd2nii_siemens_ep2d_tra_ap_int():
    """Test MRD to NIfTI conversion for dataset TRA EPI AP interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_AP_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "113_ep2d_bold_ST_TRA_magnitude_echo-1"
    file_name_expected_nii = "10_dicoms_ep2d_bold_ST_TRA_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_siemens_ep2d_tra_pa_int():
    """Test MRD to NIfTI conversion for dataset TRA EPI PA interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_PA_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "119_ep2d_bold_ST_TRA_PA_magnitude_echo-1"
    file_name_expected_nii = "13_dicoms_ep2d_bold_ST_TRA_PA_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_siemens_ep2d_tra_rl_int():
    """Test MRD to NIfTI conversion for dataset TRA EPI RL interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_RL_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "126_ep2d_bold_ST_TRA_RL_magnitude_echo-1"
    file_name_expected_nii = "16_dicoms_ep2d_bold_ST_TRA_RL_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_siemens_ep2d_tra_lr_int():
    """Test MRD to NIfTI conversion for dataset TRA EPI LR interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_LR_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "128_ep2d_bold_ST_TRA_LR_magnitude_echo-1"
    file_name_expected_nii = "17_dicoms_ep2d_bold_ST_TRA_LR_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_tra_rot_int():
    """Test MRD to NIfTI conversion for dataset TRA EPI ROT interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_ROT_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "138_ep2d_bold_ST_TRA_AP_ROT_magnitude_echo-1"
    file_name_expected_nii = "22_dicoms_ep2d_bold_ST_TRA_AP_ROT_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_tra_asc():
    """Test MRD to NIfTI conversion for dataset TRA EPI ascending"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_ASC")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "149_ep2d_bold_ST_TRA_asc_magnitude_echo-1"
    file_name_expected_nii = "31_dicoms_ep2d_bold_ST_TRA_asc_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_tra_desc():
    """Test MRD to NIfTI conversion for dataset TRA EPI descending"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_DESC")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "151_ep2d_bold_ST_TRA_desc_magnitude_echo-1"
    file_name_expected_nii = "32_dicoms_ep2d_bold_ST_TRA_desc_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_sag_ap_int():
    """Test MRD to NIfTI conversion for dataset SAG EPI AP interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_AP_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "115_ep2d_bold_ST_SAG_magnitude_echo-1"
    file_name_expected_nii = "11_dicoms_ep2d_bold_ST_SAG_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_siemens_ep2d_sag_pa_int():
    """Test MRD to NIfTI conversion for dataset SAG EPI PA interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_PA_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "121_ep2d_bold_ST_SAG_PA_magnitude_echo-1"
    file_name_expected_nii = "14_dicoms_ep2d_bold_ST_SAG_PA_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_siemens_ep2d_sag_hf_int():
    """Test MRD to NIfTI conversion for dataset SAG EPI HF interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_HF_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "130_ep2d_bold_ST_SAG_HF_magnitude_echo-1"
    file_name_expected_nii = "18_dicoms_ep2d_bold_ST_SAG_HF_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_siemens_ep2d_sag_fh_int():
    """Test MRD to NIfTI conversion for dataset SAG EPI FH interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_FH_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "132_ep2d_bold_ST_SAG_FH_magnitude_echo-1"
    file_name_expected_nii = "19_dicoms_ep2d_bold_ST_SAG_FH_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_siemens_ep2d_sag_rot_int():
    """Test MRD to NIfTI conversion for dataset SAG EPI ROT interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_ROT_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "140_ep2d_bold_ST_SAG_AP_ROT_magnitude_echo-1"
    file_name_expected_nii = "23_dicoms_ep2d_bold_ST_SAG_AP_ROT_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_sag_asc():
    """Test MRD to NIfTI conversion for dataset SAG EPI ascending"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_ASC")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "153_ep2d_bold_ST_SAG_asc_magnitude_echo-1"
    file_name_expected_nii = "33_dicoms_ep2d_bold_ST_SAG_asc_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_sag_desc():
    """Test MRD to NIfTI conversion for dataset SAG EPI descending"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_DESC")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "155_ep2d_bold_ST_SAG_desc_magnitude_echo-1"
    file_name_expected_nii = "34_dicoms_ep2d_bold_ST_SAG_desc_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_cor_rl_int():
    """Test MRD to NIfTI conversion for dataset COR EPI RL interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_RL_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "117_ep2d_bold_ST_COR_magnitude_echo-1"
    file_name_expected_nii = "012_dicoms_ep2d_bold_ST_COR_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_siemens_ep2d_cor_lr_int():
    """Test MRD to NIfTI conversion for dataset COR EPI LR interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_LR_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "123_ep2d_bold_ST_COR_LR_magnitude_echo-1"
    file_name_expected_nii = "015_dicoms_ep2d_bold_ST_COR_LR_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    # Skip tag AcquisitionTime because dcm2niix does not put leading 0s
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_cor_hf_int():
    """Test MRD to NIfTI conversion for dataset COR EPI HF interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_HF_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "134_ep2d_bold_ST_COR_HF_magnitude_echo-1"
    file_name_expected_nii = "20_dicoms_ep2d_bold_ST_COR_HF_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_siemens_ep2d_cor_fh_int():
    """Test MRD to NIfTI conversion for dataset COR EPI FH interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_FH_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "136_ep2d_bold_ST_COR_FH_magnitude_echo-1"
    file_name_expected_nii = "21_dicoms_ep2d_bold_ST_COR_FH_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    # Skip tag AcquisitionTime because dcm2niix does not put leading 0s
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_cor_rot_int():
    """Test MRD to NIfTI conversion for dataset COR EPI ROT interleaved"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_ROT_INT")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "142_ep2d_bold_ST_COR_RL_ROT_magnitude_echo-1"
    file_name_expected_nii = "24_dicoms_ep2d_bold_ST_COR_RL_ROT_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_cor_asc():
    """Test MRD to NIfTI conversion for dataset COR EPI ascending"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_ASC")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "157_ep2d_bold_ST_COR_asc_magnitude_echo-1"
    file_name_expected_nii = "35_dicoms_ep2d_bold_ST_COR_asc_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_cor_desc():
    """Test MRD to NIfTI conversion for dataset COR EPI descending"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_DESC")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "159_ep2d_bold_ST_COR_desc_magnitude_echo-1"
    file_name_expected_nii = "36_dicoms_ep2d_bold_ST_COR_desc_20250926122558"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_siemens_ep2d_tra_1slice():
    """Test MRD to NIfTI conversion for dataset TRA EPI with 1 slice"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_TRA_1SLICE")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "99_ep2d_bold_shimming_save_data_TRA_magnitude_echo-1"
    file_name_expected_nii = "41_dicoms_ep2d_bold_shimming_save_data_TRA"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["PhaseEncodingSteps"])


def test_mrd2nii_siemens_ep2d_sag_1slice():
    """Test MRD to NIfTI conversion for dataset SAG EPI with 1 slice"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_SAG_1SLICE")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "101_ep2d_bold_shimming_save_data_SAG_magnitude_echo-1"
    file_name_expected_nii = "42_dicoms_ep2d_bold_shimming_save_data_SAG"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["PhaseEncodingSteps"])


def test_mrd2nii_siemens_ep2d_cor_1slice():
    """Test MRD to NIfTI conversion for dataset COR EPI with 1 slice"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_COR_1SLICE")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "103_ep2d_bold_shimming_save_data_COR_magnitude_echo-1"
    file_name_expected_nii = "43_dicoms_ep2d_bold_shimming_save_data_COR"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine, atol=1e-3, rtol=1e-3)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["PhaseEncodingSteps"])


def test_mrd2nii_siemens_ep2d_1slice_2vols():
    """Test MRD to NIfTI conversion for dataset TRA EPI with 1 slice, 2 volumes"""

    path_dataset = os.path.join(__dir_testing__, "EP2D_1SLICE_2VOLS")
    # Define the path to the MRD file and output directory
    path_mrd = os.path.join(path_dataset, "mrd")
    path_output = os.path.join(path_dataset, "mrd2nii")
    if os.path.exists(path_output):
        shutil.rmtree(path_output)

    runner = CliRunner()

    res = runner.invoke(mrd2nii_int,
                        [
                            '--input', path_mrd,
                            '--output', path_output
                        ],
                        catch_exceptions=False)

    assert res.exit_code == 0, f"Error: {res.exit_code} - {res.output}"
    file_name_converted_nii = "105_ep2d_bold_shimming_save_data_TRA_2meas_magnitude_echo-1"
    file_name_expected_nii = "44_dicoms_ep2d_bold_shimming_save_data_TRA_2meas"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["PhaseEncodingSteps"])
