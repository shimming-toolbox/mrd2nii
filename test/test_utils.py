#!/usr/bin/python3
# -*- coding: utf-8 -*

import json
import numpy as np


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
        "EchoNumber",
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
        "MRAcquisitionType",
        "NonlinearGradientCorrection"
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
                if tag == "SliceTiming":
                    # Allow larger tolerance for SliceTiming due to potential differences in precision
                    assert np.isclose(val, expected_sidecar_data[tag][i],
                                      atol=0.0025, rtol=0.05), f"Value mismatch for tag {tag} at index {i}"
                if isinstance(val, str):
                    assert val == expected_sidecar_data[tag][i], f"Value mismatch for tag {tag} at index {i}"
                else:
                    assert np.isclose(val, expected_sidecar_data[tag][i],
                                      rtol=0.05), f"Value mismatch for tag {tag} at index {i}"
        else:
            if isinstance(sidecar_data[tag], str):
                assert sidecar_data[tag] == expected_sidecar_data[tag], f"Value mismatch for tag {tag}"
            else:
                assert np.isclose(sidecar_data[tag], expected_sidecar_data[tag],
                                  rtol=0.001), f"Value mismatch for tag {tag}"
