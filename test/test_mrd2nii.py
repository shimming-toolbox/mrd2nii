#!/usr/bin/python3
# -*- coding: utf-8 -*

from click.testing import CliRunner
import json
import nibabel as nib
import numpy as np
import os
import shutil

from mrd2nii.cli.mrd2nii_cli import mrd2nii_int
from mrd2nii import __dir_testing__


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
    nii = nib.load(os.path.join(path_output, "155_ep2d_bold_shimming_magnitude_echo-0.nii.gz"))
    expected_affine = [[-2.60416675, 0., 0., 130.62680054],
                       [0., 2.60416675, 0., -128.96043396],
                       [0., 0., 15., -56.24641037],
                       [0., 0., 0., 1.]]
    assert np.all(np.isclose(nii.affine, expected_affine))
    fname_expected_json = os.path.join(path_dataset, "nii", "093_dicoms_ep2d_bold_shimming_20250415120524.json")
    fname_json = os.path.join(path_output, "155_ep2d_bold_shimming_magnitude_echo-0.json")
    verify_sidecar(fname_json, fname_expected_json)


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
    nii = nib.load(os.path.join(path_output, "160_ep2d_bold_shimming_magnitude_echo-0.nii.gz"))
    expected_affine = [[-2.45199919, -7.93683483e-08, 5.05236149, 100.587128],
                       [0.161766902, 2.55949664, 2.60471988, -146.346985],
                       [0.862099826, -0.480271578, 13.8812542, -70.0200882],
                       [0.0, 0.0, 0.0, 1.0]]
    assert np.all(np.isclose(nii.affine, expected_affine))
    fname_expected_json = os.path.join(path_dataset, "nii", "095_dicom_rot_ep2d_bold_shimming_20250415120524.json")
    fname_json = os.path.join(path_output, "160_ep2d_bold_shimming_magnitude_echo-0.json")
    verify_sidecar(fname_json, fname_expected_json)


def test_mrd2nii_dset3():
    """Test MRD to NIfTI conversion for dataset 2. TRA EPI ROT"""

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
    # nii = nib.load(os.path.join(path_output, "160_ep2d_bold_shimming_magnitude_echo-0.nii.gz"))
    # expected_affine = [[-2.45199919, -7.93683483e-08, 5.05236149, 100.587128],
    #                    [0.161766902, 2.55949664, 2.60471988, -146.346985],
    #                    [0.862099826, -0.480271578, 13.8812542, -70.0200882],
    #                    [0.0, 0.0, 0.0, 1.0]]
    # assert np.all(np.isclose(nii.affine, expected_affine))
    # fname_expected_json = os.path.join(path_dataset, "nii", "095_dicom_rot_ep2d_bold_shimming_20250415120524.json")
    # fname_json = os.path.join(path_output, "160_ep2d_bold_shimming_magnitude_echo-0.json")
    # verify_sidecar(fname_json, fname_expected_json)
    pass


def test_mrd2nii_dset4():
    """Test MRD to NIfTI conversion for dataset 2. EPI GRAPPA"""
    """ Todo: Currently it is flipped"""

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
    nii = nib.load(os.path.join(path_output, "134_ep2d_bold_ST_shim_nomad_1.1x1.1_magnitude_echo-0.nii.gz"))

    expected_affine = [[-1.125, 0, 0, 126],
                       [0, 1.09350049, 3.52468681, -103.258766],
                       [0, -0.264351517, 14.58, -65.0683289],
                       [0, 0, 0, 1]]
    assert np.all(np.isclose(nii.affine, expected_affine, atol=1e-5))
    fname_expected_json = os.path.join(path_dataset, "nii", "077_dicoms_ep2d_bold_ST_shim_nomad_1.1x1.1_20250811095729.json")
    fname_json = os.path.join(path_output, "134_ep2d_bold_ST_shim_nomad_1.1x1.1_magnitude_echo-0.json")
    verify_sidecar(fname_json, fname_expected_json, skip_tags=["PhaseEncodingSteps"])


def verify_sidecar(fname_sidecar, fname_expected_sidecar, skip_tags=None):
    """Verify the sidecar JSON file against the expected values."""

    tags_to_check = [
        "ImagingFrequency",
        "Manufacturer",
        "ManufacturersModelName",
        "InstitutionName",
        "DeviceSerialNumber",
        "PatientPosition",
        "SeriesDescription",
        "ProtocolName",
        "ScanningSequence",
        "ScanOptions",
        "PulseSequenceName",
        "ImageType",
        "AcquisitionTime",
        "AcquisitionNumber",
        "SliceThickness",
        "SpacingBetweenSlices",
        "TablePosition",
        "EchoTime",
        "RepetitionTime",
        "FlipAngle",
        "BaseResolution",
        "ShimSetting",
        "ReceiveCoilName",
        "ReceiveCoilActiveElements",
        "CoilString",
        "PulseSequenceDetails",
        "ConsistencyInfo",
        "PercentPhaseFOV",
        "PercentSampling",
        "PhaseEncodingSteps",
        "BandwidthPerPixelPhaseEncode",
        "DerivedVendorReportedEchoSpacing",
        "DwellTime",
        "SliceTiming",
        "ImageOrientationPatientDICOM",
        "RefLinesPE",
        "FrequencyEncodingSteps",
        "AcquisitionMatrixPE",
        "ReconMatrixPE",
        "ParallelReductionFactorInPlane",
        "ParallelReductionFactorOutOfPlane",
        "MRAcquisitionType"
    ]

    with open(fname_sidecar, 'r', encoding='utf-8') as f:
        sidecar_data = json.load(f)

    with open(fname_expected_sidecar, 'r', encoding='utf-8') as f:
        expected_sidecar_data = json.load(f)

    for tag in tags_to_check:
        if skip_tags is not None and tag in skip_tags:
            continue

        # If the tag is not in either files, that's good
        if tag not in sidecar_data and tag not in expected_sidecar_data:
            continue

        assert tag in sidecar_data, f"Missing tag {tag} in sidecar file"
        assert tag in expected_sidecar_data, f"Missing tag {tag} in expected sidecar file"
        if isinstance(sidecar_data[tag], list):
            assert len(sidecar_data[tag]) == len(expected_sidecar_data[tag]), f"Length mismatch for tag {tag}"
            for i, val in enumerate(sidecar_data[tag]):
                if isinstance(val, str):
                    assert expected_sidecar_data[tag][i] == val, f"Value mismatch for tag {tag} at index {i}"
                else:
                    assert np.isclose(val, expected_sidecar_data[tag][i], rtol=0.05), f"Value mismatch for tag {tag} at index {i}"
        else:
            if isinstance(sidecar_data[tag], str):
                assert expected_sidecar_data[tag] == sidecar_data[tag], f"Value mismatch for tag {tag} at index {i}"
            else:
                assert np.isclose(sidecar_data[tag], expected_sidecar_data[tag], rtol=0.001), f"Value mismatch for tag {tag}"
