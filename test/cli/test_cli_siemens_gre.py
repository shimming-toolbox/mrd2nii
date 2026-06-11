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


def test_mrd2nii_siemens_gre_tra_rot_int():
    """Test MRD to NIfTI conversion for dataset TRA GRE ROT interleaved"""

    path_dataset = os.path.join(__dir_testing__, "GRE_TRA_ROT_INT")
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

    for echo in [1, 2]:
        for modality in ["magnitude", "phase"]:
            file_name_converted_nii = "77_gre_fmap_baseline_tra_rot"
            if modality == "phase":
                file_name_expected_nii = "39_dicoms_gre_fmap_baseline_tra_rot_20251007123810"
                file_name_converted_nii = f"{file_name_converted_nii}_phase_echo-{echo}"
                file_name_expected_nii = f"{file_name_expected_nii}_e{echo}_ph"
            else:
                file_name_expected_nii = "38_dicoms_gre_fmap_baseline_tra_rot_20251007123810"
                file_name_converted_nii = f"{file_name_converted_nii}_magnitude_echo-{echo}"
                file_name_expected_nii = f"{file_name_expected_nii}_e{echo}"
            nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
            nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
            assert np.allclose(nii.affine, nii_expected.affine)
            if modality == "phase":
                assert np.allclose(nii.get_fdata(), (nii_expected.get_fdata() + 4096) / 2, atol=1)
            else:
                assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
            assert nii.header.get_dim_info()[2] == 2
            fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
            fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
            verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime", "ScanningSequence",
                                                                       "ScanOptions", "ImageType"])


def test_mrd2nii_siemens_gre_sag_rot_int():
    """Test MRD to NIfTI conversion for dataset SAG GRE ROT interleaved"""

    path_dataset = os.path.join(__dir_testing__, "GRE_SAG_ROT_INT")
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

    for echo in [1, 2]:
        for modality in ["magnitude", "phase"]:
            file_name_converted_nii = "79_gre_fmap_baseline_sag_rot"
            if modality == "phase":
                file_name_expected_nii = "42_dicoms_gre_fmap_baseline_sag_rot_20251007123810"
                file_name_converted_nii = f"{file_name_converted_nii}_phase_echo-{echo}"
                file_name_expected_nii = f"{file_name_expected_nii}_e{echo}_ph"
            else:
                file_name_expected_nii = "41_dicoms_gre_fmap_baseline_sag_rot_20251007123810"
                file_name_converted_nii = f"{file_name_converted_nii}_magnitude_echo-{echo}"
                file_name_expected_nii = f"{file_name_expected_nii}_e{echo}"
            nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
            nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
            assert np.allclose(nii.affine, nii_expected.affine)
            if modality == "phase":
                assert np.allclose(nii.get_fdata(), (nii_expected.get_fdata() + 4096) / 2, atol=1)
            else:
                assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
            assert nii.header.get_dim_info()[2] == 2
            fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
            fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
            verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime", "ScanningSequence",
                                                                       "ScanOptions", "ImageType"])


def test_mrd2nii_siemens_gre_cor_rot_int():
    """Test MRD to NIfTI conversion for dataset COR GRE ROT interleaved"""

    path_dataset = os.path.join(__dir_testing__, "GRE_COR_ROT_INT")
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

    for echo in [1, 2]:
        for modality in ["magnitude", "phase"]:
            file_name_converted_nii = "80_gre_fmap_baseline_cor_rot"
            if modality == "phase":
                file_name_expected_nii = "44_dicoms_gre_fmap_baseline_cor_rot_20251007123810"
                file_name_converted_nii = f"{file_name_converted_nii}_phase_echo-{echo}"
                file_name_expected_nii = f"{file_name_expected_nii}_e{echo}_ph"
            else:
                file_name_expected_nii = "43_dicoms_gre_fmap_baseline_cor_rot_20251007123810"
                file_name_converted_nii = f"{file_name_converted_nii}_magnitude_echo-{echo}"
                file_name_expected_nii = f"{file_name_expected_nii}_e{echo}"
            nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
            nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
            assert np.allclose(nii.affine, nii_expected.affine)
            if modality == "phase":
                assert np.allclose(nii.get_fdata(), (nii_expected.get_fdata() + 4096) / 2, atol=1)
            else:
                assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
            assert nii.header.get_dim_info()[2] == 2
            fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
            fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
            verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime", "ScanningSequence",
                                                                       "ScanOptions", "ImageType"])


def test_mrd2nii_siemens_loc_1slice():
    """Test MRD to NIfTI conversion for dataset TRA EPI with 1 slice, 2 volumes"""
    # Todo: FrequencyEncodingSteps is different between converted and expected sidecar

    path_dataset = os.path.join(__dir_testing__, "LOC_1SLICE")
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
    file_name_converted_nii = "106_localizer_for_segmentation_magnitude_echo-1"
    file_name_expected_nii = "45_dicoms_localizer_for_segmentation"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    assert nii.header.get_dim_info()[2] == 2
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["ScanningSequence", "ScanOptions",
                                                               "ImageType", "PhaseEncodingSteps",
                                                               "FrequencyEncodingSteps", "AcquisitionMatrixPE"])
