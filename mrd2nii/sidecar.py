#!/usr/bin/python3
# -*- coding: utf-8 -*

import base64
import logging

import ismrmrd


def create_bids_sidecar(metadata, volume_images):

    # Parse Mini hdr
    img_metas = []
    for image in volume_images:
        img_metas.append(read_vendor_header_img(image))

    sidecar = {
        "Modality": "MR",
        "MagneticFieldStrength": metadata.acquisitionSystemInformation.systemFieldStrength_T,
        "ImagingFrequency": metadata.experimentalConditions.H1resonanceFrequency_Hz / 1e6,
        "Manufacturer": metadata.acquisitionSystemInformation.systemVendor,
        "ManufacturersModelName": metadata.acquisitionSystemInformation.systemModel,
        "InstitutionName": metadata.acquisitionSystemInformation.institutionName,
        # "InstitutionAddress": "",
        "DeviceSerialNumber": extract_device_serial_number(metadata),
        # "StationName": "",
        # "BodyPart": "",
        "PatientPosition": metadata.measurementInformation.patientPosition.value,
        # "ProcedureStepDescription": "",
        # "SoftwareVersions": "",
        "MRAcquisitionType": "",
        # "StudyDescription": "",
        "SeriesDescription": img_metas[0].get("SequenceDescription"),
        "ProtocolName": metadata.measurementInformation.protocolName,
        "ScanningSequence": extract_sequence_type(metadata),
        # "SequenceVariant": "",
        "ScanOptions": "",
        "PulseSequenceName": img_metas[0].get("SequenceString"),
        "ImageType": extract_image_type(img_metas[0]),
        # "ImageTypeText": [],
        # "NonlinearGradientCorrection": "",
        # "SeriesNumber": volume_images[0].getHead().measurement_uid, not right
        # "SeriesNumber": "",
        "AcquisitionTime": extract_acq_time(img_metas[0]),
        "AcquisitionNumber": img_metas[0].get("AcquisitionNumber"),
        # "ImageComments": "",
        "SliceThickness": volume_images[0].getHead().field_of_view[2],
        "SpacingBetweenSlices": img_metas[0].get("SpacingBetweenSlices"),
        "TablePosition": [],
        "EchoTime": None,
        "RepetitionTime": None,
        # "MTState": "",
        "FlipAngle": metadata.sequenceParameters.flipAngle_deg[0],
        # "PartialFourier": "",
        "BaseResolution": "",
        "ShimSetting": [],
        # "TxRefAmp": "",
        # "PhaseResolution": "",
        "ReceiveCoilName": "",
        "ReceiveCoilActiveElements": img_metas[0].get("CoilString"),
        "CoilString": "",
        "PulseSequenceDetails": "",
        "RefLinesPE": None,
        # "CoilCombinationMethod": "",
        "ConsistencyInfo": "",
        # "MatrixCoilMode": "",
        "PercentPhaseFOV": img_metas[0].get("PercentPhaseFoV"),
        "PercentSampling": img_metas[0].get("PercentSampling"),
        # "EchoTrainLength": "",
        # "EchoTrainLength": img_metas[0].get("EchoTrainLength"), Does not work
        # "PartialFourierDirection": "",
        "PhaseEncodingSteps": img_metas[0].get("NoOfPhaseEncodingSteps"),
        "FrequencyEncodingSteps": metadata.encoding[0].reconSpace.matrixSize.x,
        "AcquisitionMatrixPE": metadata.encoding[0].encodedSpace.matrixSize.y,
        "ReconMatrixPE": metadata.encoding[0].reconSpace.matrixSize.y,
        "BandwidthPerPixelPhaseEncode": img_metas[0].get("BandwidthPerPixelPhaseEncode"),
        "ParallelReductionFactorInPlane": None,
        "ParallelReductionFactorOutOfPlane": None,
        # "ParallelAcquisitionTechnique": "",
        # "EffectiveEchoSpacing": None,
        "DerivedVendorReportedEchoSpacing": None,
        # "AcquisitionDuration": None,
        # "TotalReadoutTime": None,
        # "PixelBandwidth": None,
        "DwellTime": None,
        # "PhaseEncodingDirection": "",
        "SliceTiming": [],
        "ImageOrientationPatientDICOM": extract_image_orientation_patient_dicom(volume_images[0]),
        # "InPlanePhaseEncodingDirectionDICOM": "",
        "ConversionSoftware": "mrd2nii"
    }

    # sidecar["SliceTiming"] = extract_slice_timing(metadata, volume_images)
    sidecar["SliceTiming"] = extract_slice_timing_ice_mini_hdr(metadata, img_metas, volume_images)
    sidecar["EchoTime"] = extract_te(metadata, volume_images)
    sidecar["RepetitionTime"] = extract_tr(metadata)
    sidecar["DerivedVendorReportedEchoSpacing"] = extract_vendor_echo_spacing(metadata)
    sidecar["DwellTime"] = extract_dwell_time(metadata)
    sidecar["MRAcquisitionType"] = extract_acq_type(metadata)

    sidecar["TablePosition"] = extract_table_position(volume_images[0])

    if metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1 != 1:
        sidecar["ParallelReductionFactorInPlane"] = metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1
    else:
        sidecar.pop("ParallelReductionFactorInPlane")

    if metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2 != 1:
        sidecar["ParallelReductionFactorOutOfPlane"] = metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2
    else:
        sidecar.pop("ParallelReductionFactorOutOfPlane")

    vendor_header = read_vendor_header_metadata(metadata)
    sidecar["ScanOptions"] = extract_scan_options(metadata, vendor_header)
    if vendor_header is not None:
        sidecar["PulseSequenceDetails"] = vendor_header.get("tSequenceFileName")
        sidecar["BaseResolution"] = int(vendor_header["sKSpace"].get("lBaseResolution"))
        sidecar["ShimSetting"] = extract_shim_settings(vendor_header)
        # sCoilSelectMeas.aRxCoilSelectData[0].asList[0].sCoilElementID.tCoilID
        sidecar["ReceiveCoilName"] = vendor_header["sCoilSelectMeas"]["aRxCoilSelectData[0]"]["asList[0]"]["sCoilElementID"]["tCoilID"]
        sidecar["CoilString"] = vendor_header["sCoilSelectMeas"]["aRxCoilSelectData[0]"]["asList[0]"]["sCoilElementID"]["tCoilID"]
        sidecar["ConsistencyInfo"] = vendor_header.get("ulVersion")
        if vendor_header["sPat"].get("lRefLinesPE") is None:
            sidecar.pop("RefLinesPE")
        else:
            sidecar["RefLinesPE"] = int(vendor_header["sPat"].get("lRefLinesPE"))

    # Todo: GRAPPA: sPat['ucPATMode']. Need to verify what a sense scan does

    # logging.info(sidecar)
    return sidecar


def extract_scan_options(metadata, vendor_header):
    scan_options = ""

    if vendor_header is not None:
        phase_partial_fourier = True if vendor_header['sKSpace']['ucPhasePartialFourier'] != '16' else False
        if phase_partial_fourier:
            scan_options += "PFP\\"

        readout_partial_fourier = True if vendor_header['sKSpace']['ucReadoutPartialFourier'] != '16' else False
        if readout_partial_fourier:
            scan_options += "PFF\\"

    if has_fatsat(metadata):
        scan_options += "FS\\"

    return scan_options if len(scan_options) < 2 else scan_options[:-1]


def extract_acq_type(metadata):
    if "2D" in metadata.encoding[0].trajectoryDescription.comment:
        return "2D"
    elif "3D" in metadata.encoding[0].trajectoryDescription.comment:
        return "3D"


def extract_image_orientation_patient_dicom(image):
    dir_enc = image.meta['ImageRowDir']
    dir_enc.extend(image.meta['ImageColumnDir'])
    dir_enc = [float(x) for x in dir_enc]
    return dir_enc


def extract_shim_settings(vhdr_metadata):
    return [
        int(vhdr_metadata["sGRADSPEC"]["asGPAData[0]"]["lOffsetX"]),
        int(vhdr_metadata["sGRADSPEC"]["asGPAData[0]"]["lOffsetY"]),
        int(vhdr_metadata["sGRADSPEC"]["asGPAData[0]"]["lOffsetZ"]),
        int(vhdr_metadata["sGRADSPEC"]["alShimCurrent[0]"]),
        int(vhdr_metadata["sGRADSPEC"]["alShimCurrent[1]"]),
        int(vhdr_metadata["sGRADSPEC"]["alShimCurrent[2]"]),
        int(vhdr_metadata["sGRADSPEC"]["alShimCurrent[3]"]),
        int(vhdr_metadata["sGRADSPEC"]["alShimCurrent[4]"])
    ]


def extract_acq_time(img_meta):
    acq_time = img_meta.get("AcquisitionTime")
    if acq_time is None:
        return ""

    if len(acq_time) != 13:
        logging.warning(f"acq time does not have the expected length in extract_acq_time: {acq_time}")

    parsed_acq_time = f"{acq_time[0:2]}:{acq_time[2:4]}:{acq_time[4:]}"

    return parsed_acq_time


def extract_table_position(image):
    # I have not found a good table position tag. Using the SlicePosLightMarker tag and the position seems to do the
    # trick
    return [0, 0, image.getHead().position[2] - float(image.meta.get('SlicePosLightMarker')[2])]


def extract_sequence_type(metadata):
    if metadata.sequenceParameters.sequence_type == "EPI":
        return "EP"
    else:
        return ""


def extract_device_serial_number(metadata):
    measurement_id = metadata.measurementInformation.measurementID
    if measurement_id.find("_") > 0:
        return measurement_id[:measurement_id.find("_")]
    else:
        return ""


def extract_image_type(img_meta):
    image_type = img_meta.get("ImageType")
    if image_type is not None:
        image_type = image_type.split("\\")
    else:
        image_type = []

    if img_meta.get("ComplexImageComponent") is not None:
        image_type.append(img_meta.get("ComplexImageComponent"))
    return image_type


def read_vendor_header_img(image):
    meta = ismrmrd.Meta.deserialize(image.attribute_string)
    # logging.info(meta.keys())
    # ['AcquisitionContrast', 'DistortionCorrection', 'EchoTime', 'FrameOfReference', 'IceImageControl', 'IceMiniHead', 'ImageColumnDir', 'ImageHistory', 'ImageRowDir', 'ImageSliceNormDir', 'ImageType', 'Keep_image_geometry', 'ReadPhaseSeqSwap', 'RepetitionTime', 'SequenceDescription', 'SlicePosLightMarker']
    vendor_header = None
    if 'IceMiniHead' in meta:
        vendor_header = base64.b64decode(meta['IceMiniHead']).decode('utf-8')

    if vendor_header is None:
        return None

    def parse(head: str):
        # <XProtocol>{example}

        def extract_param_type_and_name(siemens_hdr_name: str):
            # <ParamString."SequenceString">
            if siemens_hdr_name.find(".") == -1:
                return "", siemens_hdr_name

            param_type = siemens_hdr_name.strip("<")[:siemens_hdr_name.find(".")]
            param_name = siemens_hdr_name[
                         siemens_hdr_name.find("\"") + 1:len(siemens_hdr_name) - siemens_hdr_name[::-1].find("\"") - 1]

            return param_type, param_name

        parsed = {}
        idx_master = 0
        while len(head.strip().strip("\n")) > idx_master:
            idx_start_name = head.find("<", idx_master) + 1
            if idx_start_name == -1:
                return head.strip()
            idx_end_name = head.find(">", idx_master)
            if idx_end_name == -1:
                return head.strip()
            idx_start_value = head.find("{", idx_master) + 1
            if idx_start_value == -1:
                return head.strip()
            bracket_running_open_close = 1
            idx = idx_start_value
            while bracket_running_open_close:
                idx_open = head.find("{", idx)
                idx_close = head.find("}", idx)
                if idx_open != -1 and idx_close != -1:
                    if idx_open < idx_close:
                        idx = idx_open + 1
                        bracket_running_open_close += 1
                    else:
                        idx = idx_close + 1
                        bracket_running_open_close -= 1
                elif idx_open != -1:
                    idx = idx_open + 1
                    bracket_running_open_close += 1
                elif idx_close != -1:
                    idx = idx_close + 1
                    bracket_running_open_close -= 1
                else:
                    raise RuntimeError("No closing bracket found")

            idx_end_value = idx - 1
            if idx_end_value == -1:
                return head.strip()

            param_type, param_name = extract_param_type_and_name(head[idx_start_name:idx_end_name])
            if param_type == "ParamArray":
                logging.warning(f"ParamArray not implemented, not processing: {head[idx_start_value:idx_end_value]}")
            elif param_type == "ParamMap":
                parsed[param_name] = parse(head[idx_start_value:idx_end_value])
            elif param_type == "ParamLong":
                if head[idx_start_value:idx_end_value].strip() != "":
                    parsed[param_name] = int(head[idx_start_value:idx_end_value].strip())
                else:
                    parsed[param_name] = head[idx_start_value:idx_end_value].strip()
            elif param_type == "ParamDouble":
                if head[idx_start_value:idx_end_value].strip() != "":
                    parsed[param_name] = float(head[idx_start_value:idx_end_value].strip())
                else:
                    parsed[param_name] = head[idx_start_value:idx_end_value].strip()
            elif param_type == "ParamString":
                parsed[param_name] = head[idx_start_value:idx_end_value].strip().strip("\"")
            else:
                parsed[param_name] = parse(head[idx_start_value:idx_end_value])
            idx_master = idx_end_value + 1

        return parsed

    head_dict = parse(vendor_header)
    head_dict = head_dict["XProtocol"][""]["DICOM"]
    if head_dict.get('SliceNo') == '':
        head_dict['SliceNo'] = 0
    if head_dict.get('TimeAfterStart') == '':
        head_dict['TimeAfterStart'] = 0
    if head_dict.get('ProtocolSliceNumber') == '':
        head_dict['ProtocolSliceNumber'] = 0

    # example
    # {'NumberInSeries': ' 2 ',
    # 'AcquisitionDate': ' "20250415" ',
    # 'AcquisitionTime': ' "172708.825000" ',
    # 'AcquisitionNumber': ' 1 ',
    # 'EchoNumber': ' 1 ',
    # 'SliceNo': {},
    # 'TR': ' 1170 ',
    # 'TE': ' 40 ',
    # 'SliceMeasurementDuration': ' 21000 ',
    # 'SequenceDescription': ' "ep2d_bold_shimming" ',
    # 'TimeAfterStart': {},
    # 'SpacingBetweenSlices': ' 15 ',
    # 'UsedChannelString': ' "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" ',
    # 'SequenceString': ' "epfid2d1_96" ',
    # 'EchoColumnPosition': ' 48 ',
    # 'EchoLinePosition': ' 48 ',
    # 'EchoPartitionPosition': ' 32 ',
    # 'RealDwellTime': ' 2700 ',
    # 'EchoTrainLength': ' 1 ',
    # 'NoOfAverages': ' 1 ',
    # 'NoOfPhaseEncodingSteps': ' 96 ',
    # 'PercentPhaseFoV': ' 100 ',
    # 'PercentSampling': ' 100 ',
    # 'ProtocolSliceNumber': ' 1 ',
    # 'SequenceMask': {},
    # 'BandwidthPerPixelPhaseEncode': ' 17.96 ',
    # 'BitsStored': ' 12 ',
    # 'CoilString': ' "HC1-7;NC1" ',
    # 'AcquisitionContrast': ' "UNKNOWN" ',
    # 'ImageType': ' "ORIGINAL\\PRIMARY\\FMRI\\NONE" ',
    # 'ImageHistory': ' "ChannelMixing:ND=true_CMM=1_CDM=1\\ACCAlgo:16" ',
    # 'ImageTypeValue3': ' "M" ',
    # 'ComplexImageComponent': ' "MAGNITUDE" ',
    # 'PixelRepresentation': {},
    # 'SOPInstanceUID': ' "1.3.12.2.1107.5.2.43.167006.2025041517270898436002838" '}
    # logging.info(head_dict)
    return head_dict


def read_vendor_header_metadata(metadata):
    vendor_header = None
    for param in metadata.userParameters.userParameterBase64:
        if param.name == "SiemensBuffer_PROTOCOL_MeasYaps":
            vendor_header = param.value

    if vendor_header is None:
        logging.info("No vendor header")
        return None

    header_dict = {}
    for i, line in enumerate(vendor_header.decode().split("\n")):
        # Skip comments and termination lines
        if line == "\x00":
            continue
        if line.startswith("###"):
            continue
        line = line.replace(" ", "").replace("\t", "")
        eq_idx = line.find("=")

        logging.debug(line)
        current = header_dict
        for i_tag, tag in enumerate(line[:eq_idx].split(".")):

            if tag not in current:
                current[tag] = {}

            if i_tag == len(line[:eq_idx].split(".")) - 1:
                # Add value
                current[tag] = line[eq_idx + 1:].strip("\"")

            current = current[tag]

    return header_dict


def extract_dwell_time(metadata):
    dwell_time = ""
    for param in metadata.userParameters.userParameterDouble:
        if param.name == "DwellTime_0":
            dwell_time = param.value / 1e6
            break

    return dwell_time


def extract_vendor_echo_spacing(metadata):
    if len(metadata.sequenceParameters.echo_spacing) >= 2:
        raise NotImplementedError("Conversion for multiple echo_spacings is not supported")

    return metadata.sequenceParameters.echo_spacing[0] / 1000


def extract_tr(metadata):
    if len(metadata.sequenceParameters.TR) >= 2:
        raise NotImplementedError("Conversion for multiple TRs is not supported")

    return metadata.sequenceParameters.TR[0] / 1000


def extract_te(metadata, volume_images):
    # Find which contrast and extract the appropriate TE
    contrast = None
    for image in volume_images:
        if contrast is None:
            contrast = image.getHead().contrast
        elif contrast != image.getHead().contrast:
            raise RuntimeError("Multiple contrasts fed to extract_te()")
        else:
            pass

    return metadata.sequenceParameters.TE[contrast] / 1000


def extract_slice_timing_ice_mini_hdr(metadata, img_metas, volume_images):
    """ img_metas and volume_images must be ordered the same """
    # Todo: Con cross check with TimeAfterStart tag in img_metas
    nb_slices = int(metadata.encoding[0].encodingLimits.slice.maximum) + 1

    # Extract ordering
    mrd_idx_to_order_idx = {}

    for param in metadata.userParameters.userParameterLong:
        if param.name.startswith("RelativeSliceNumber_"):
            i_slice = int(param.name.split("_")[-1]) - 1
            mrd_idx_to_order_idx[i_slice] = int(param.value)

    slice_timing = [0 for _ in range(nb_slices)]

    def convert_acq_time_string_to_s_past_midnight(acq_time: str):
        acq_time = acq_time.split(".")
        if len(acq_time) != 2 or len(acq_time[0]) != 6 or len(acq_time[1]) != 6:
            raise ValueError("Acquisition time not in the expected format")
        atime = 3600 * int(acq_time[0][:2]) + 60 * int(acq_time[0][2:4]) + int(acq_time[0][4:]) + (int(acq_time[1]) / 1e6)
        return atime

    first_timestamp = convert_acq_time_string_to_s_past_midnight(img_metas[0]['AcquisitionTime'])
    for img_meta in img_metas:
        if convert_acq_time_string_to_s_past_midnight(img_meta['AcquisitionTime']) < first_timestamp:
            first_timestamp = convert_acq_time_string_to_s_past_midnight(img_meta['AcquisitionTime'])

    if first_timestamp is None:
        raise RuntimeError("First slice unavailable, can't extract_slice_timing")

    for i, image in enumerate(volume_images):
        hdr = image.getHead()
        atime = convert_acq_time_string_to_s_past_midnight(img_metas[i]['AcquisitionTime'])
        slice_timing[mrd_idx_to_order_idx[hdr.slice]] = round(atime - first_timestamp, 3)

    return slice_timing


def extract_slice_timing(metadata, volume_images):
    """
    Not sure why, but this way of extracting the slice timing is wrong by a factor of 2.5. See
    extract_slice_timing_ice_mini_hdr for the "correct" way to extract the slice timing.
    """
    # Todo: This is probably in 'ticks', which is 2.5 ms/tick
    logging.warning("Slice timing is different (/2.5) than dcm2niix but seems to be in the right order")
    nb_slices = int(metadata.encoding[0].encodingLimits.slice.maximum) + 1

    # Extract ordering
    mrd_idx_to_order_idx = {}

    for i_param, param in enumerate(metadata.userParameters.userParameterLong):
        if param.name == f"RelativeSliceNumber_{i_param + 1}":
            mrd_idx_to_order_idx[i_param] = int(param.value)

    slice_timing = [0 for _ in range(nb_slices)]

    first_timestamp = volume_images[0].getHead().acquisition_time_stamp
    for image in volume_images:
        hdr = image.getHead()
        if hdr.acquisition_time_stamp < first_timestamp:
            first_timestamp = hdr.acquisition_time_stamp

    if first_timestamp is None:
        raise RuntimeError("First slice unavailable, can't extract_slice_timing")

    for image in volume_images:
        hdr = image.getHead()
        slice_timing[mrd_idx_to_order_idx[hdr.slice]] = (hdr.acquisition_time_stamp - first_timestamp) / 1000

    return slice_timing


def has_fatsat(metadata):
    fatsat = False
    for i_param, param in enumerate(metadata.userParameters.userParameterString):
        if param.name == "FatWaterContrast":
            fatsat = True

    return fatsat
