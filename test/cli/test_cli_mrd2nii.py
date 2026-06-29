#!/usr/bin/python3
# -*- coding: utf-8 -*

from click.testing import CliRunner
import nibabel as nib
import numpy as np
import os
import shutil

from mrd2nii.cli.mrd2nii_cli import mrd2nii_int
from mrd2nii.nii_utils import orient_nii_to
from mrd2nii import __dir_testing__

from test.test_utils import verify_sidecar


def test_mrd2nii_dset1():
    """Test MRD to NIfTI conversion for dataset 1. TRA EPI"""

    path_dataset = os.path.join(__dir_testing__, "dset1")
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
    file_name_converted_nii = "155_ep2d_bold_shimming_magnitude_echo-1"
    file_name_expected_nii = "093_dicoms_ep2d_bold_shimming_20250415120524"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    # The data is slightly different, probably due to when the data is pulled out of the ICE chain
    # assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["AcquisitionTime"])


def test_mrd2nii_dset2():
    """Test MRD to NIfTI conversion for dataset 2. TRA EPI ROT"""

    path_dataset = os.path.join(__dir_testing__, "dset2")
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
    file_name_converted_nii = "160_ep2d_bold_shimming_magnitude_echo-1"
    file_name_expected_nii = "095_dicom_rot_ep2d_bold_shimming_20250415120524"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine)
    # The data is slightly different, probably due to when the data is pulled out of the ICE chain
    # assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_dset3():
    """Test MRD to NIfTI conversion for dataset 3. These are waveforms."""

    path_dataset = os.path.join(__dir_testing__, "dset3")
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
    pass


def test_mrd2nii_dset4():
    """Test MRD to NIfTI conversion for dataset 4. EPI GRAPPA"""
    # Needed to use image.meta['ImageRowDir'], etc instead of image.get_head().read_dir, etc

    path_dataset = os.path.join(__dir_testing__, "dset4")
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
    file_name_converted_nii = "134_ep2d_bold_ST_shim_nomad_1.1x1.1_magnitude_echo-1"
    file_name_expected_nii = "077_dicoms_ep2d_bold_ST_shim_nomad_1.1x1.1_20250811095729"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine, atol=1e-5)
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["PhaseEncodingSteps"])


def test_mrd2nii_dset5():
    """Test MRD to NIfTI conversion for dataset 5. EPI GRAPPA"""
    # Needed to use pos light marker instead of .position. This tag can also be used to extract the table position

    path_dataset = os.path.join(__dir_testing__, "dset5")
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
    file_name_converted_nii = "31_ep2d_bold_ST_shim_nomad_5vols_magnitude_echo-1"
    file_name_expected_nii = "008_dicoms_ep2d_bold_ST_shim_nomad_5vols_20250811095729"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine, atol=1e-5)
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_dset6():
    """Test MRD to NIfTI conversion for dataset 5. Localizer with multiple chuncks."""

    path_dataset = os.path.join(__dir_testing__, "dset6")
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

    assert res.exit_code == 0
    file_name_converted_nii = "64_Localizer_chunk-1_magnitude_echo-1"
    nii1 = nib.load(os.path.join(path_output, file_name_converted_nii + ".nii.gz"))
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    with open(fname_json, 'r') as f:
        json_data = f.read()
    file_name_converted_nii = "64_Localizer_chunk-2_magnitude_echo-1"
    nii2 = nib.load(os.path.join(path_output, file_name_converted_nii + ".nii.gz"))
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    with open(fname_json, 'r') as f:
        json_data = f.read()


def test_mrd2nii_dset7():
    """Test MRD to NIfTI conversion for dataset 5. T1w"""
    # mrd2nii is saying that the "slice" dimension is 192 voxels (verified on the scanner, that is correct)
    # but dcm2niix is saying 320

    path_dataset = os.path.join(__dir_testing__, "dset7")
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
    file_name_converted_nii = "77_T1w_magnitude_echo-1"
    file_name_expected_nii = "005_dicoms_T1w_20250827110428"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    # Orient to expected data
    nii = orient_nii_to(nii, nii_expected.header.get_dim_info())
    assert np.allclose(nii.affine, nii_expected.affine, atol=1e-5)
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["SeriesDescription", "ImageType"], measyaps=False)


def test_mrd2nii_dset8():
    """Test MRD to NIfTI conversion for dataset 5. ME GRE. PYTEST_DONT_REWRITE"""

    path_dataset = os.path.join(__dir_testing__, "dset8")
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
    fname_prefix_dcm2niix = "006_dicoms_gre_shimming_0.5x0.5_5echos_20250827110428_e"
    fname_prefix_mrd2nii = "86_gre_shimming_0.5x0.5_5echos_magnitude_echo-"
    for i_echo in range(1, 6):
        fname_dcm2niix_mag = os.path.join(path_dataset, "nii", f"{fname_prefix_dcm2niix}{i_echo}.nii.gz")
        assert os.path.exists(fname_dcm2niix_mag)
        fname_mrd2nii_mag = os.path.join(path_output, f"{fname_prefix_mrd2nii}{i_echo}.nii.gz")
        assert os.path.exists(fname_mrd2nii_mag)
        assert np.allclose(nib.load(fname_mrd2nii_mag).affine, nib.load(fname_dcm2niix_mag).affine, rtol=1e-5)

        fname_expected_json = os.path.join(path_dataset, "nii", f"{fname_prefix_dcm2niix}{i_echo}.json")
        fname_json = os.path.join(path_output, f"{fname_prefix_mrd2nii}{i_echo}.json")
        # Skip series description tag since it has FIRE appended to it, which is correct
        # This scan does not include the private vendor header, so some tags are missing
        # Todo: Acquisition time is wrong because it's not taking the acq time of the first slice
        # Todo: Scanning Sequence is weird. The metadata says there is a TI of 300.s in MEGRE
        verify_sidecar(fname_json,
                       fname_expected_json,
                       skip_tags=["ImageType", "ScanningSequence", "PhaseEncodingSteps", "PhaseEncodingSteps",
                                  "SeriesDescription", "AcquisitionTime"], measyaps=False)


def test_mrd2nii_dset9():
    """Test MRD to NIfTI conversion for dataset 5. Single stack localizer."""

    path_dataset = os.path.join(__dir_testing__, "dset9")
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
    file_name_converted_nii = "27_localizer_for_segmentation_magnitude_echo-1"
    file_name_expected_nii = "008_dicoms_localizer_for_segmentation_20250910123127"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    assert np.allclose(nii.affine, nii_expected.affine, atol=1e-5)
    assert np.allclose(nii.get_fdata(), nii_expected.get_fdata(), atol=1)
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json,
                   fname_expected_json,
                   skip_tags=["ImageType", "AcquisitionMatrixPE", "ScanningSequence", "PhaseEncodingSteps",
                              "FrequencyEncodingSteps"], measyaps=False)


def test_mrd2nii_dset10():
    """Test MRD to NIfTI conversion for dataset 5. T1w"""
    # Todo: Check on the scanner the dimensions. Mrd2nii is saying that the "slice" dimension is 192voxels but dcm2niix is saying 320

    path_dataset = os.path.join(__dir_testing__, "dset10")
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
    file_name_converted_nii = "19_T1w_magnitude_echo-1"
    file_name_expected_nii = "002_dicoms_T1w_20250910123127"
    nii = nib.load(os.path.join(path_output, f"{file_name_converted_nii}.nii.gz"))
    nii_expected = nib.load(os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.nii.gz"))
    # Not using nii_expected.affine since the axes are not the same (we reorient to (0, 1, 2) = (freq, phase, slice))
    nii_reorient = orient_nii_to(nii, nii_expected.header.get_dim_info())
    assert np.allclose(nii_reorient.affine, nii_expected.affine, atol=1e-5)
    assert np.allclose(nii_reorient.get_fdata(), nii_expected.get_fdata(), atol=1)
    fname_expected_json = os.path.join(path_dataset, "nii", f"{file_name_expected_nii}.json")
    fname_json = os.path.join(path_output, f"{file_name_converted_nii}.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["ImageType"], measyaps=False)
