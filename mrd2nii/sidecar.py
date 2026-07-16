#!/usr/bin/python3
# -*- coding: utf-8 -*

import copy
import base64
from datetime import datetime, timedelta, time
import logging
import ismrmrd
import numpy as np
import math

logger = logging.getLogger(__name__)


def create_bids_sidecar(metadata, volume_images, dim_info=(None, None, None)):

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
        "InstitutionAddress": "",
        "DeviceSerialNumber": extract_device_serial_number(metadata),
        # "StationName": "",
        # "BodyPart": "",
        "PatientPosition": metadata.measurementInformation.patientPosition.value,
        # "ProcedureStepDescription": "",
        "SoftwareVersions": "",
        "MRAcquisitionType": extract_acq_type(metadata),
        # "StudyDescription": "",
        "SeriesDescription": img_metas[0].get("SequenceDescription"),
        "ProtocolName": metadata.measurementInformation.protocolName,
        "ScanningSequence": extract_scanning_sequence(metadata),
        # "SequenceVariant": "",
        "ScanOptions": "",
        "PulseSequenceName": img_metas[0].get("SequenceString"),
        "ImageType": extract_image_type(img_metas[0]),
        # "ImageTypeText": [],
        "NonlinearGradientCorrection": extract_non_lin_gradient_corr(img_metas[0]),
        # "SeriesNumber": volume_images[0].getHead().measurement_uid, not right
        # "SeriesNumber": "",
        "AcquisitionTime": extract_acq_time(img_metas[0]),
        "AcquisitionNumber": img_metas[0].get("AcquisitionNumber"),
        # "ImageComments": "",
        "SliceThickness": volume_images[0].getHead().field_of_view[2],
        "SpacingBetweenSlices": img_metas[0].get("SpacingBetweenSlices"),
        "TablePosition": extract_table_position(volume_images[0]),
        "EchoNumber": img_metas[0].get("EchoNumber") if metadata.encoding[0].encodingLimits.contrast.maximum > 0 else None,
        "EchoTime": extract_te(metadata, volume_images),
        "RepetitionTime": extract_tr(metadata),
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
        "DerivedVendorReportedEchoSpacing": extract_vendor_echo_spacing(metadata),
        # "AcquisitionDuration": None,
        # "TotalReadoutTime": None,
        # "PixelBandwidth": None,
        "DwellTime": extract_dwell_time(metadata),
        "PhaseEncodingDirection": "",
        "SliceTiming": [],
        "ImageOrientationPatientDICOM": extract_image_orientation_patient_dicom(volume_images[0]),
        # "InPlanePhaseEncodingDirectionDICOM": "",
        "ConversionSoftware": "mrd2nii"
    }

    # sidecar["SliceTiming"] = extract_slice_timing(metadata, volume_images)
    sidecar["SliceTiming"] = extract_slice_timing_ice_mini_hdr(metadata, img_metas, volume_images)

    if metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1 != 1:
        sidecar["ParallelReductionFactorInPlane"] = metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1
    else:
        sidecar.pop("ParallelReductionFactorInPlane")

    if metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2 != 1:
        sidecar["ParallelReductionFactorOutOfPlane"] = metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2
    else:
        sidecar.pop("ParallelReductionFactorOutOfPlane")

    # Available with Meas param set and MeasYaps
    measyaps, dicom = read_vendor_header_metadata(metadata)
    if measyaps is not None:
        sidecar["ScanOptions"] = extract_scan_options(metadata, measyaps)
        sidecar["PulseSequenceDetails"] = measyaps.get("tSequenceFileName")
        sidecar["BaseResolution"] = int(measyaps["sKSpace"].get("lBaseResolution"))
        sidecar["ShimSetting"] = extract_shim_settings(measyaps)
        sidecar["ReceiveCoilName"] = extract_receive_coil_name(measyaps)
        sidecar["CoilString"] = extract_coil_string(measyaps)
        sidecar["ConsistencyInfo"] = measyaps.get("ulVersion")
        if measyaps["sPat"].get("lRefLinesPE") is None:
            sidecar.pop("RefLinesPE")
        else:
            sidecar["RefLinesPE"] = int(measyaps["sPat"].get("lRefLinesPE"))
        sidecar["PhaseEncodingDirection"] = extract_phase_encoding_direction(volume_images[0], measyaps, dim_info)

    # Available with MEAS param set
    if dicom is not None:
        sidecar["InstitutionAddress"] = dicom.get("InstitutionAddress")
        sidecar["SoftwareVersions"] = dicom.get("SoftwareVersions")

    # Todo: GRAPPA: sPat['ucPATMode']. Need to verify what a sense scan does

    sidecar = clean_up(sidecar)
    return sidecar


def clean_up(sidecar):
    keys_to_remove = []
    for key, value in sidecar.items():
        if value is None or value == "" or value == []:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        sidecar.pop(key)

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

    if len(metadata.sequenceParameters.TI) > 0 and any(ti > 0 for ti in metadata.sequenceParameters.TI):
        scan_options += "IR\\"

    return scan_options if len(scan_options) < 2 else scan_options[:-1]


def extract_acq_type(metadata):
    if metadata.encoding[0].trajectoryDescription is not None:
        if "2D" in metadata.encoding[0].trajectoryDescription.comment:
            return "2D"
        elif "3D" in metadata.encoding[0].trajectoryDescription.comment:
            return "3D"
    else:
        if metadata.encoding[0].encodingLimits.kspace_encoding_step_2.maximum > 0:
            return "3D"
        else:
            return "2D"


def extract_image_orientation_patient_dicom(image):
    dir_enc = copy.deepcopy(image.meta['ImageRowDir'])
    dir_enc.extend(image.meta['ImageColumnDir'])
    dir_enc = [float(x) for x in dir_enc]
    return dir_enc


def extract_shim_settings(vhdr_metadata):
    if vhdr_metadata.get("sGRADSPEC") is None:
        return []

    shim_settings = []
    # Gradients (1st order)
    grad_names = ["lOffsetX", "lOffsetY", "lOffsetZ"]
    if vhdr_metadata["sGRADSPEC"].get("asGPAData[0]") is not None:
        gpa_name = "asGPAData[0]"
    elif vhdr_metadata["sGRADSPEC"].get("asGPAData") is not None:
        gpa_name = "asGPAData"
    else:
        gpa_name = None

    if gpa_name is not None:
        for grad_name in grad_names:
            if vhdr_metadata["sGRADSPEC"][gpa_name].get(grad_name) is None:
                shim_settings.append(None)
            else:
                shim_settings.append(int(vhdr_metadata["sGRADSPEC"][gpa_name][grad_name]))
    else:
        shim_settings = [None] * 3

    # 2nd order
    for i in range(5):
        if vhdr_metadata["sGRADSPEC"].get(f"alShimCurrent[{i}]") is not None:
            shim_settings.append(int(vhdr_metadata["sGRADSPEC"][f"alShimCurrent[{i}]"]))
        elif vhdr_metadata["sGRADSPEC"].get("alShimCurrent") is not None:
            if not isinstance(vhdr_metadata["sGRADSPEC"]["alShimCurrent"], list):
                shim_settings.append(None)
            else:
                shim_settings.append(int(vhdr_metadata["sGRADSPEC"]["alShimCurrent"][i]))
        else:
            shim_settings.append(None)

    # If it's an array of None, return empty list
    if np.all([s is None for s in shim_settings]):
        shim_settings = []

    return shim_settings


def extract_receive_coil_name(vhdr_metadata):
    if vhdr_metadata.get("sCoilSelectMeas") is None:
        return ""
    if vhdr_metadata["sCoilSelectMeas"].get("aRxCoilSelectData[0]") is None:
        return ""
    if vhdr_metadata["sCoilSelectMeas"]["aRxCoilSelectData[0]"].get("asList[0]") is None:
        return ""
    if vhdr_metadata["sCoilSelectMeas"]["aRxCoilSelectData[0]"]["asList[0]"].get("sCoilElementID") is None:
        return ""
    if vhdr_metadata["sCoilSelectMeas"]["aRxCoilSelectData[0]"]["asList[0]"]["sCoilElementID"].get("tCoilID") is None:
        return ""

    return vhdr_metadata["sCoilSelectMeas"]["aRxCoilSelectData[0]"]["asList[0]"]["sCoilElementID"]["tCoilID"]


def extract_coil_string(vhdr_metadata):
    return extract_receive_coil_name(vhdr_metadata)


def extract_acq_time(img_meta):
    acq_time = img_meta.get("AcquisitionTime")
    if acq_time is None:
        return ""

    if len(acq_time) != 13:
        logger.warning(f"acq time does not have the expected length in extract_acq_time: {acq_time}")

    acq_time = acq_time[:2] + ":" + acq_time[2:4] + ":" + acq_time[4:]

    dummy_date = datetime.today().date()
    if img_meta.get("TimeAfterStart") is None:
        acq_start = time.fromisoformat(acq_time).isoformat()
    else:
        dt_acq_start = datetime.combine(dummy_date, time.fromisoformat(acq_time)) - timedelta(seconds=img_meta.get("TimeAfterStart"))
        acq_start = dt_acq_start.time().isoformat()
    return acq_start


def extract_table_position(image):
    # I have not found a good table position tag. Using the SlicePosLightMarker tag and the position seems to do the
    # trick
    return [0, 0, image.getHead().position[2] - float(image.meta.get('SlicePosLightMarker')[2])]


def extract_scanning_sequence(metadata):
    seq_type = ""
    if metadata.sequenceParameters.sequence_type == "EPI":
        seq_type += "EP\\"
    elif metadata.sequenceParameters.sequence_type == "Flash":
        seq_type += "GR\\"

    if len(metadata.sequenceParameters.TI) > 0 and any(ti > 0 for ti in metadata.sequenceParameters.TI):
        seq_type += "IR\\"
    if len(seq_type) != 0:
        seq_type = seq_type[:-1]
    return seq_type


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


def extract_non_lin_gradient_corr(img_meta):
    list_im_type = img_meta.get("ImageTypeValue4")
    if list_im_type is not None:
        if 'DIS2D' in list_im_type or 'ND' in list_im_type:
            return True
    else:
        return ""


def parse_xproto(head: str, array_data=None):
    # <XProtocol>{example}
    def parse_data(head, type, array_value):
        def my_bool_comp(a_string):
            if not isinstance(a_string, str):
                raise ValueError(f"Expected a string for boolean conversion, got {type(a_string)}")
            a_string = a_string.strip().strip("\n").strip("\"")
            if a_string.lower() in ['true', '1']:
                return True
            elif a_string.lower() in ['false', '0']:
                return False
            else:
                raise ValueError(f"Cannot convert string to boolean: {a_string}")

        type_conversion = {
            'int': int,
            'long': int,
            'float': float,
            'double': float,
            'bool': my_bool_comp
        }
        default_value = {
            'int': 0,
            'long': 0,
            'float': 0.0,
            'double': 0.0,
            'bool': None
        }

        def _parse_data(data, type):
            data = data.strip()
            if data != "":
                if len(data.split(" ")) > 1:
                    # Create a list of values
                    data = data.strip().strip("\n")
                    # Remove comments if any:
                    if data.find("<Comment>") != -1:
                        idx_start = data.find("<Comment>")
                        idx_end = data.find("\n", idx_start)
                        data = data[:idx_start] + data[idx_end:]
                        data = data.strip().strip("\n")

                        if data.strip("\"") == "":
                            return None

                    value = []
                    vals = data.split(" ")
                    for i_val, val in enumerate(vals):
                        if val.strip().strip("\n") == "<Default>":
                            value = type_conversion[type](vals[i_val + 1])
                            break
                        if val.strip().strip("\n") != "":
                            value.append(type_conversion[type](val.strip().strip("\n")))
                else:
                    value = type_conversion[type](data)
            else:
                # Default values
                value = default_value[type]

            return value

        value = _parse_data(head, type)

        if array_value is not None:
            val = _parse_data(array_value, type)
            if val != value and value != default_value[type]:
                logger.debug(f"Array value found {val} vs value found: {value}")
            value = val

        return value

    def parse_choice(head):
        head = head.strip().strip("\n")
        if "<Limit>" not in head[:7]:
            raise RuntimeError("No Limit tag in ParamChoice")

        idx_open = head.find("{")
        idx_close = head.find("}")
        if idx_open == -1 or idx_close == -1:
            raise RuntimeError("No opening or closing bracket in ParamChoice")
        limit = parse_list_of_strings(head[idx_open + 1:idx_close])

        head = head[idx_close + 1:].strip()

        if head.find("<Default>") == -1:
            raise RuntimeError("No Default tag in ParamChoice")

        head = head.strip("<Default>")
        end_idx = head.find("\n")
        default_value = head.strip()[1:end_idx].strip()

        if head.strip(f"\"{default_value}\"").strip() != "":
            raise RuntimeError("Extra information after Default tag in ParamChoice")

        return {
            "Limit": limit,
            "Default": default_value
        }

    def parse_list_of_strings(head):
        value = []
        while len(head) > 0:
            idx_open = head.find("\"")
            idx_close = head.find("\"", idx_open + 1)
            if idx_open == -1 or idx_close == -1:
                break
            value.append(str(head[idx_open+1:idx_close]))
            head = head[idx_close+1:]
        return value

    def parse_array(head: str, array_data=None):
        try:
            head = head.strip().strip("\n")
            if "<DefaultSize>" not in head[:13]:
                raise RuntimeError("No DefaultSize tag in ParamArray")

            head = head.strip("<DefaultSize>")
            ind = head.find("\n")
            if ind == -1:
                raise RuntimeError("No new line after DefaultSize tag in ParamArray")

            default_size = int(head[:ind].strip())

            head = head[ind:].strip()
            if "<MaxSize>" not in head[:9]:
                raise RuntimeError("No MaxSize tag in ParamArray")

            head = head.strip("<MaxSize>")
            ind = head.find("\n")
            if ind == -1:
                raise RuntimeError("No new line after MaxSize tag in ParamArray")

            max_size = int(head[:ind].strip())

            head = head[ind:].strip()
            if "<Default>" not in head[:9]:
                raise RuntimeError("No Default tag in ParamArray")
            head = head.strip("<Default>").strip()
            if "<ParamString.\"\">" in head[:16]:
                ind = head.find("\n")
                head = head[ind:].strip()
                output = []
                while len(head) > 0:
                    ind_end_bracket = head.find("}")
                    entry = head[:ind_end_bracket].strip("{").strip("}").strip()
                    output.append(entry.strip("\"") if entry != "" else "")
                    head = head[ind_end_bracket + 1:]
            elif "<ParamMap.\"\">" in head[:13]:
                # extract_header from data
                idxs = find_matching_brackets(head)
                if idxs is None:
                    raise RuntimeError("No matching brackets found in ParamArray (header)")
                head_start, head_end = idxs
                new_head = head[head_start:head_end]

                array_list = []
                more_elements = True
                value_end = head_end
                array_end = 0
                while more_elements:
                    if head[value_end+1:] == "":
                        break
                    idxs = find_matching_brackets(head, value_end + 1)
                    if idxs is None:
                        raise RuntimeError("No idx found")

                    value_start, value_end = idxs
                    value = head[value_start:value_end]

                    if array_data is not None:
                        idxs = find_matching_brackets(array_data, array_end + 1)
                        if idxs is None:
                            if array_data[array_end + 1:].strip().strip("\n") != "":
                                logger.warning(f"Not the same amount of data in array and header. header: {value}, array: {array_data[array_end + 1:]}")
                        else:
                            array_start, array_end = idxs
                            value = array_data[array_start:array_end]
                    array_list.append(parse_xproto(new_head, value))
                if len(array_list) > 1:
                    output = array_list
                else:
                    output = array_list[0]
            elif "<ParamDouble.\"\">" in head[:16]:
                output = head
            elif "<ParamLong.\"\">" in head[:14]:
                output = head
            elif "<ParamArray.\"\">" in head[:15]:
                output = head
            else:
                raise RuntimeError("No ParamString or ParamMap tag in ParamArray")

            return output

        except Exception as e:
            logger.warning(f"Failed to parse ParamArray:{head}. Error: {e}")
            return []

    def extract_param_type_and_name(siemens_hdr_name: str):
        # <ParamString."SequenceString">
        if siemens_hdr_name.find(".") == -1:
            if siemens_hdr_name == "Class":
                return "Class", ""
            else:
                return "", siemens_hdr_name

        param_type = siemens_hdr_name.strip("<")[:siemens_hdr_name.find(".")]
        param_name = siemens_hdr_name[
                     siemens_hdr_name.find("\"") + 1:len(siemens_hdr_name) - siemens_hdr_name[::-1].find("\"") - 1]

        return param_type, param_name

    def find_matching_brackets(head, start_from=0):
        idx_start_value = head.find("{", start_from) + 1
        if idx_start_value == 0:
            return
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
            return
        return idx_start_value, idx_end_value

    def find_idxs_of_tags(head, idx_master=0):
        idx_start_name = head.find("<", idx_master) + 1
        if idx_start_name == -1:
            return
        idx_end_name = head.find(">", idx_master)
        if idx_end_name == -1:
            return

        idxs = find_matching_brackets(head, idx_master)
        if idxs is None:
            return

        idx_start_value, idx_end_value = idxs

        return idx_start_name, idx_end_name, idx_start_value, idx_end_value

    if array_data is not None and array_data.strip() == "":
        array_data = None

    parsed = {}
    idx_master = 0
    idx_start_array = 0
    idx_end_array = 1
    head = head.strip().strip("\n").strip("\x00")
    while len(head) > idx_master + 1:
        idxs = find_idxs_of_tags(head, idx_master)
        if idxs is None:
            return head.strip()
        idx_start_name, idx_end_name, idx_start_value, idx_end_value = idxs
        param_type, param_name = extract_param_type_and_name(head[idx_start_name:idx_end_name])

        if array_data is not None:
            idx_start_array, idx_end_array = find_matching_brackets(array_data, idx_start_array)
            if param_type not in ["ParamLong", "ParamDouble", "ParamMap", "ParamBool", "ParamString", "ParamArray"]:
                logging.warning(f"Not implemented array type for {param_type}. Ignoring array data for this parameter.")

        value = head[idx_start_value:idx_end_value]
        array = array_data[idx_start_array:idx_end_array] if array_data is not None else None
        if param_name in ["SliceInformation", "dInPlaneRot", "asGPAData"]:
            pass
        if param_type == "ParamArray":
            parsed[param_name] = parse_array(value, array)
        elif param_type == "ParamMap":
            parsed[param_name] = parse_xproto(value, array)
        elif param_type == "ParamLong":
            parsed[param_name] = parse_data(value, "long", array)
        elif param_type == "ParamDouble":
            parsed[param_name] = parse_data(value, "double", array)
        elif param_type == "ParamString":
            if array_data is not None:
                parsed[param_name] = array.strip().strip("\"")
            else:
                parsed[param_name] = value.strip().strip("\"")
        elif param_type == "ParamBool":
            parsed[param_name] = parse_data(value, "bool", array)
        elif param_type == "ParamChoice":
            parsed[param_name] = parse_choice(value)
        elif param_type == "Class":
            # Class is not like the other parameters
            # Find its name
            idx_class_start = head.find("\"")
            idx_class_end = head.find("\"", idx_class_start + 1)
            param_name = head[idx_class_start:idx_class_end]
            parsed[param_name] = parse_xproto(head[idx_class_end+1:])
        elif param_type == "Method" or param_type == "Event":
            parsed[param_name] = parse_list_of_strings(value.strip("\n").strip())
        elif param_type == "Connection":
            parsed[param_name] = value.strip("\n").strip()
        elif param_type == "ProtocolComposer":
                parsed[param_name] = value.strip("\n").strip()
        else:
            parsed[param_name] = parse_xproto(value)
        idx_master = idx_end_value + 1
        if array_data is not None:
            idx_start_array = idx_end_array + 1

    return parsed


def read_vendor_header_img(image):
    meta = ismrmrd.Meta.deserialize(image.attribute_string)
    # logger.info(meta.keys())
    # ['AcquisitionContrast', 'DistortionCorrection', 'EchoTime', 'FrameOfReference', 'IceImageControl', 'IceMiniHead', 'ImageColumnDir', 'ImageHistory', 'ImageRowDir', 'ImageSliceNormDir', 'ImageType', 'Keep_image_geometry', 'ReadPhaseSeqSwap', 'RepetitionTime', 'SequenceDescription', 'SlicePosLightMarker']
    vendor_header = None
    if 'IceMiniHead' in meta:
        vendor_header = base64.b64decode(meta['IceMiniHead']).decode('utf-8')

    if vendor_header is None:
        return None

    head_dict = parse_xproto(vendor_header)
    head_dict = head_dict["XProtocol"][""]["DICOM"]
    # head_dict["XProtocol"][""]["CONTROL"] also exists but does not have much information
    if head_dict.get('SliceNo') == '':
        head_dict['SliceNo'] = 0
    if head_dict.get('TimeAfterStart') == '':
        head_dict['TimeAfterStart'] = 0
    if head_dict.get('ProtocolSliceNumber') == '':
        head_dict['ProtocolSliceNumber'] = 0
    return head_dict


def read_vendor_header_metadata(metadata):
    """ I believe this is the measYaps data structure """

    # A few protocols seem possible:
    # - SiemensBuffer_PROTOCOL_MeasYaps
    # - SiemensBuffer_PROTOCOL_Phoenix
    # - SiemensBuffer_PROTOCOL_Meas

    headers_to_read = []
    header_names = []
    for param in metadata.userParameters.userParameterBase64:
        header_names.append(param.name)

    # Read Meas if both Meas and MeasYaps are both are present
    if "SiemensBuffer_PROTOCOL_Meas" in header_names:
        headers_to_read.append("SiemensBuffer_PROTOCOL_Meas")
    if "SiemensBuffer_PROTOCOL_MeasYaps" in header_names and not "SiemensBuffer_PROTOCOL_Meas" in header_names:
        headers_to_read.append("SiemensBuffer_PROTOCOL_MeasYaps")

    if "SiemensBuffer_PROTOCOL_Phoenix" in header_names:
        headers_to_read.append("SiemensBuffer_PROTOCOL_Phoenix")

    measyaps = None
    dicom = None

    for param in metadata.userParameters.userParameterBase64:
        if param.name not in headers_to_read:
            continue
        if param.name == "SiemensBuffer_PROTOCOL_MeasYaps":
            logger.debug("MeasYaps protocol found, trying to parse")
            measyaps = read_measyaps(param.value)
        if param.name == "SiemensBuffer_PROTOCOL_Phoenix":
            logger.warning("Phoenix protocol found, but not yet supported")
            logger.debug(param.value)
        if param.name == "SiemensBuffer_PROTOCOL_Meas":
            logger.debug("Meas protocol found")
            head_dict = parse_xproto(str(param.value.decode('utf-8')))
            measyaps = head_dict["XProtocol"][""]["MEAS"] | head_dict["XProtocol"][""]["YAPS"]
            dicom = head_dict["XProtocol"][""]["DICOM"]

    return measyaps, dicom

def read_measyaps(vendor_header):
    header_dict = {}
    for i, line in enumerate(vendor_header.decode().split("\n")):
        # Skip comments and termination lines
        if line == "\x00":
            continue
        if line.startswith("###"):
            continue
        line = line.replace(" ", "").replace("\t", "")
        eq_idx = line.find("=")

        logger.debug(line)
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
    echo_spacing = metadata.sequenceParameters.echo_spacing[0] / 1000

    if echo_spacing == 0:
        return None

    return echo_spacing


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
            raise RuntimeError("Multiple contrasts in extract_te()")
        else:
            pass

    return metadata.sequenceParameters.TE[contrast] / 1000


def extract_slice_timing_ice_mini_hdr(metadata, img_metas, volume_images):
    """ img_metas and volume_images must be ordered the same """
    # Todo: Cross check with TimeAfterStart tag in img_metas
    nb_slices = int(metadata.encoding[0].encodingLimits.slice.maximum) + 1
    if len(volume_images) != nb_slices:
        logger.warning("Number of slices in metadata does not correspond to number of images, not extracting slice timing")
        return []

    # Trivial case, if only one slice, no slice timing is necessary
    if nb_slices == 1:
        return []

    # Extract ordering
    mrd_idx_to_order_idx = extract_mrd_index_to_prot_sli_number(volume_images)
    slice_timing = [0 for _ in range(nb_slices)]

    def convert_acq_time_string_to_s_past_midnight(acq_time: str):
        acq_time = acq_time.split(".")
        if len(acq_time) != 2 or len(acq_time[0]) != 6 or len(acq_time[1]) != 6:
            raise ValueError("Acquisition time not in the expected format")
        atime = 3600 * int(acq_time[0][:2]) + 60 * int(acq_time[0][2:4]) + int(acq_time[0][4:]) + (int(acq_time[1]) / 1e6)
        return atime

    first_timestamp = convert_acq_time_string_to_s_past_midnight(img_metas[0]['AcquisitionTime'])
    first_timestamp_idx = 0
    for i, img_meta in enumerate(img_metas):
        if convert_acq_time_string_to_s_past_midnight(img_meta['AcquisitionTime']) < first_timestamp:
            first_timestamp = convert_acq_time_string_to_s_past_midnight(img_meta['AcquisitionTime'])
            first_timestamp_idx = i

    if first_timestamp is None:
        raise RuntimeError("First slice unavailable, can't extract_slice_timing")

    for i, image in enumerate(volume_images):
        hdr = image.getHead()
        atime = convert_acq_time_string_to_s_past_midnight(img_metas[i]['AcquisitionTime'])

        if img_metas[first_timestamp_idx].get('TimeAfterStart') is None:
            slice_timing[mrd_idx_to_order_idx[hdr.slice]] = round(atime - first_timestamp, 4)
        else:
            slice_timing[mrd_idx_to_order_idx[hdr.slice]] = round(atime - first_timestamp + img_metas[first_timestamp_idx]['TimeAfterStart'], 4)

    return slice_timing


def extract_slice_timing(metadata, volume_images):
    """
    Not sure why, but this way of extracting the slice timing is wrong by a factor of 2.5. See
    extract_slice_timing_ice_mini_hdr for the "correct" way to extract the slice timing.

    We currently use the extract_slice_timing_ice_mini_hdr function instead
    """
    # Todo: This is probably in 'ticks', which is 2.5 ms/tick
    logger.warning("Slice timing is different (/2.5) than dcm2niix but seems to be in the right order")
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


def extract_mrd_index_to_prot_sli_number(volume_images):
    img_metas = []
    for image in volume_images:
        img_metas.append(read_vendor_header_img(image))

    mapping = {}
    for i, meta in enumerate(img_metas):
        if meta.get("ProtocolSliceNumber") is not None:
            mapping[volume_images[i].getHead().slice] = meta["ProtocolSliceNumber"]
    return mapping


def extract_prot_sli_number_to_mrd_index(volume_images):
    img_metas = []
    for image in volume_images:
        img_metas.append(read_vendor_header_img(image))

    mapping = {}
    for i, meta in enumerate(img_metas):
        if meta.get("ProtocolSliceNumber") is not None:
            mapping[meta["ProtocolSliceNumber"]] = volume_images[i].getHead().slice
    return mapping


def get_main_dir(dir_vector):
    for i, dir in enumerate(dir_vector):
        dir_vector[i] = float(dir)
    return np.argmax(np.abs(dir_vector))


def extract_phase_encoding_direction(volume_image, vendor_header, dim_info):
    if dim_info == (None, None, None):
        return ""

    # These metadata contain the in-plane rotation information
    # vendor_header.sSliceArray.asSlice[8].dInPlaneRot
    # Vendor_metadata.sAAInitialOffset.SliceInformation.dInPlaneRot
    if vendor_header.get("sAAInitialOffset") is None:
        return ""
    if vendor_header["sAAInitialOffset"].get("SliceInformation") is None:
        return ""

    # TRA
    if get_main_dir(volume_image.meta["ImageSliceNormDir"]) == 2:
        if vendor_header["sAAInitialOffset"]["SliceInformation"].get('dInPlaneRot') is None or \
                np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), 0):
            direction = "-"
        elif np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), math.pi) or \
            np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), -math.pi):
            direction = ""
        elif np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), math.pi / 2):
            direction = ""
        elif np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), -math.pi / 2):
            direction = "-"
        else:
            raise NotImplementedError("In-plane rotation not supported for phase encoding direction extraction")
    # SAG
    elif get_main_dir(volume_image.meta["ImageSliceNormDir"]) == 1:
        if vendor_header["sAAInitialOffset"]["SliceInformation"].get('dInPlaneRot') is None or \
                np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), 0):
            direction = ""
        elif np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), math.pi):
            direction = "-"
        elif np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), math.pi / 2):
            direction = ""
        elif np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), -math.pi / 2):
            direction = "-"
        else:
            raise NotImplementedError("In-plane rotation not supported for phase encoding direction extraction")
    elif get_main_dir(volume_image.meta["ImageSliceNormDir"]) == 0:
        if vendor_header["sAAInitialOffset"]["SliceInformation"].get('dInPlaneRot') is None or \
                np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), 0):
            direction = ""
        elif np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), math.pi):
            direction = "-"
        elif np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), math.pi / 2):
            direction = "-"
        elif np.isclose(float(vendor_header["sAAInitialOffset"]["SliceInformation"]['dInPlaneRot']), -math.pi / 2):
            direction = ""
        else:
            raise NotImplementedError("In-plane rotation not supported for phase encoding direction extraction")
    else:
        raise RuntimeError("Unknown ImageSliceNormDir value")

    mapping = {0: 'i', 1: 'j', 2: 'k'}

    # dim_info[1] is the axis of the phase encoding direction
    return f"{mapping[dim_info[1]]}{direction}"


def get_n_repetitions(volume_images):
    nb_repetitions = 0
    for i in range(len(volume_images)):
        if nb_repetitions < volume_images[i].getHead().repetition:
            nb_repetitions = volume_images[i].getHead().repetition
    nb_repetitions += 1
    return nb_repetitions


def get_n_slices(volume_images, metadata):
    # Extract nb_slices
    if extract_n_encoding_directions(metadata) > 1:
        nb_slices = metadata.encoding[0].encodedSpace.matrixSize.z
    else:
        nb_slices = int(metadata.encoding[0].encodingLimits.slice.maximum) + 1

    nb_repetitions = get_n_repetitions(volume_images)

    # Error check
    if len(volume_images) != nb_slices * nb_repetitions:
        logger.warning(f"Number of images ({len(volume_images)}) does not match the expected number of images")
        if len(volume_images) % nb_repetitions != 0:
            raise RuntimeError("Error while extracting nb_slices from number of images and repetitions")

    return nb_slices


def get_is_3d(metadata):
    return extract_n_encoding_directions(metadata) > 1


def extract_n_encoding_directions(metadata):
    cnt = 0
    if metadata.encoding[0].encodingLimits.kspace_encoding_step_0 is not None:
        if not (metadata.encoding[0].encodingLimits.kspace_encoding_step_0.minimum == 0 and
                metadata.encoding[0].encodingLimits.kspace_encoding_step_0.maximum == 0):
            cnt += 1
    if metadata.encoding[0].encodingLimits.kspace_encoding_step_1 is not None:
        if not (metadata.encoding[0].encodingLimits.kspace_encoding_step_1.minimum == 0 and
                metadata.encoding[0].encodingLimits.kspace_encoding_step_1.maximum == 0):
            cnt += 1
    if metadata.encoding[0].encodingLimits.kspace_encoding_step_2 is not None:
        if not (metadata.encoding[0].encodingLimits.kspace_encoding_step_2.minimum == 0 and
                metadata.encoding[0].encodingLimits.kspace_encoding_step_2.maximum == 0):
            cnt += 1
    return cnt
