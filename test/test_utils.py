#!/usr/bin/python3
# -*- coding: utf-8 -*

import json
import numpy as np

TAGS_FROM_MRD_HEADER = [
    "SliceThickness",
    "TablePosition",
    "EchoTime",
    "SliceTiming"
]

TAGS_FROM_DSET_METADATA = [
    "MagneticFieldStrength",
    "ImagingFrequency",
    "Manufacturer",
    "ManufacturersModelName",
    "InstitutionName",
    "DeviceSerialNumber",
    "PatientPosition",
    "MRAcquisitionType",
    "ProtocolName",
    "ScanningSequence",
    "EchoNumber",
    "EchoTime",
    "RepetitionTime",
    "FlipAngle",
    "FrequencyEncodingSteps",
    "AcquisitionMatrixPE",
    "ReconMatrixPE",
    "ParallelReductionFactorInPlane",
    "ParallelReductionFactorOutOfPlane",
    "DerivedVendorReportedEchoSpacing",
    "DwellTime",
    "SliceTiming"

]

TAGS_FROM_IMAGE_META = [
    "TablePosition",
    "PhaseEncodingDirection",
    "ImageOrientationPatientDICOM"
]

TAGS_FROM_ICE_MINI_HEADER = [
    "SeriesDescription",
    "ImageType",
    "PulseSequenceName",
    "ImageType",
    "NonlinearGradientCorrection",
    "AcquisitionTime",
    "AcquisitionNumber",
    "SpacingBetweenSlices",
    "EchoNumber",
    "ReceiveCoilActiveElements",
    "PercentPhaseFOV",
    "PercentSampling",
    "PhaseEncodingSteps",
    "BandwidthPerPixelPhaseEncode",
    "SliceTiming"
]

TAGS_FROM_MEASYAPS_PARAM_SET = [
    "ScanOptions",
    "BaseResolution",
    "ShimSetting",
    "ReceiveCoilName",
    "CoilString",
    "PulseSequenceDetails",
    "RefLinesPE",
    "ConsistencyInfo",
    "PhaseEncodingDirection"

]

TAGS_FROM_MEAS_PARAM_SET = [
    "InstitutionAddress",
    "SoftwareVersions",
]


def verify_sidecar(fname_sidecar, fname_expected_sidecar, skip_tags=None, dset_meta=True, mrd_head=True,
                   image_meta=True, ice_mini_head=True, measyaps=True, meas=False):
    """Verify the sidecar JSON file against the expected values."""

    DEFAULT_SKIPPED_TAGS = ["MagneticFieldStrength"]

    if skip_tags is not None:
        skip_tags = set(skip_tags + DEFAULT_SKIPPED_TAGS)
    else:
        skip_tags = set(DEFAULT_SKIPPED_TAGS)

    tags_to_check = set(TAGS_FROM_DSET_METADATA +
                        TAGS_FROM_MRD_HEADER +
                        TAGS_FROM_IMAGE_META +
                        TAGS_FROM_ICE_MINI_HEADER +
                        TAGS_FROM_MEASYAPS_PARAM_SET +
                        TAGS_FROM_MEAS_PARAM_SET)

    with open(fname_sidecar, 'r', encoding='utf-8') as f:
        sidecar_data = json.load(f)

    with open(fname_expected_sidecar, 'r', encoding='utf-8') as f:
        expected_sidecar_data = json.load(f)

    for tag in tags_to_check:
        if skip_tags is not None and tag in skip_tags:
            continue

        # Look if this parameter was not processed because the parameter set was not available. If so, skip the check for this parameter.
        if ((not dset_meta and tag in TAGS_FROM_DSET_METADATA) or
            (not mrd_head and tag in TAGS_FROM_MRD_HEADER) or
            (not image_meta and tag in TAGS_FROM_IMAGE_META) or
            (not ice_mini_head and tag in TAGS_FROM_ICE_MINI_HEADER) or
            (not measyaps and tag in TAGS_FROM_MEASYAPS_PARAM_SET) or
            (not meas and tag in TAGS_FROM_MEAS_PARAM_SET)):
            continue

        # If the tag is not in either files, that's good
        if tag not in sidecar_data and tag not in expected_sidecar_data:
            continue
        # If the is in the sidecar but not in the dcm2niix conversion, we can't check it, so skip it
        if tag in sidecar_data and tag not in expected_sidecar_data:
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
