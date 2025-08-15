#!/usr/bin/python3
# -*- coding: utf-8 -*

import json
import logging
import math
import os

import ismrmrd
import nibabel as nib
import numpy as np

from mrd2nii.sidecar import create_bids_sidecar, read_vendor_header_img


def mrd2nii_dset(dset: ismrmrd.Dataset, output_dir):
    """Convert MRD file to NIfTI format.

    Args:
        dset (ismrmrd.Dataset): Dataset.
        output_dir (str): Directory to save the output NIfTI file.

    Returns:
        None
    """
    groups = dset.list()
    if 'xml' not in groups:
        raise RuntimeError("No XML header found in the dataset")

    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    metadata = ismrmrd.xsd.CreateFromDocument(xml_header)

    # Read images
    files_output = {}
    for group in groups:

        if group == 'waveforms':
            process_waveforms(dset, output_dir)

        elif group == 'data':
            logging.warning("Raw data is not supported yet")
            # for i_acq in range(0, dset.number_of_acquisitions()):
            #     pass
                # acq = dset.read_acquisition(i_acq)
                # acq.data

        elif group.startswith('image_') or group.startswith('images_'):
            for i_img in range(0, dset.number_of_images(group)):
                image = dset.read_image(group, i_img)

                acq_name = f"{image.getHead().measurement_uid}"
                acq_name += f"_{metadata.measurementInformation.protocolName}"

                if image.getHead().image_type == ismrmrd.IMTYPE_MAGNITUDE:
                    acq_name += "_magnitude"
                elif image.getHead().image_type == ismrmrd.IMTYPE_PHASE:
                    acq_name += "_phase"
                else:
                    raise NotImplementedError(f"Image type {image.getHead().image_type} not implemented")

                acq_name += f"_echo-{image.getHead().contrast}"

                if acq_name not in files_output:
                    files_output[acq_name] = []

                files_output[acq_name].append(image)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert to NIfTI
    for acq_name, volume_images in files_output.items():
        logging.info(f"Converting {acq_name} to NIfTI")

        # Convert to NIfTI
        nii, sidecar = mrd2nii_volume(metadata, volume_images)

        # Save NIfTI file
        fname_nii = os.path.join(output_dir, f"{acq_name}.nii.gz")
        nib.save(nii, fname_nii)
        logging.info(f"Saved NIfTI file to {fname_nii}")

        # Save JSON
        fname_json = os.path.join(output_dir, f"{acq_name}.json")
        with open(fname_json, 'w', encoding='utf-8') as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=4)


def process_waveforms(dset, path_output):
    data_wav = {
        0: {'name': 'ECG',
            'data': {'Time_tics': [], 'trace_0': []}
            },
        1: {'name': 'PULSE',
            'data': {'Time_tics': [], 'trace_0': []}
            },
        2: {'name': 'RESP',
            'data': {'Time_tics': [], 'trace_0': []}
            },
        3: {'name': 'EXT1',
            'data': {'Time_tics': [], 'trace_0': []}
            },
        4: {'name': 'EXT2',
            'data': {'Time_tics': [], 'trace_0': []}
            },
    }

    # The waveforms come in chunks, we read all of them and store them in a dictionary
    for i_waveform in range(0, dset.number_of_waveforms()):
        waveform = dset.read_waveform(i_waveform)
        wav_header = waveform.getHead()
        if wav_header.waveform_id not in data_wav:
            logging.warning(f"Waveform ID {wav_header.waveform_id} not recognized, skipping")
            continue

        if wav_header.number_of_samples == 0:
            continue

        # A time_tic is 2.5 ms, so we need to divide by 2.5ms per time_tic
        time_tic_per_sample = wav_header.sample_time_us / 1000 / 2.5
        time_tics = np.arange(wav_header.time_stamp, wav_header.time_stamp + (wav_header.number_of_samples * time_tic_per_sample), time_tic_per_sample).astype(int)

        data_wav[wav_header.waveform_id]['data']['Time_tics'].extend(time_tics.tolist())

        for i_channel in range(wav_header.channels):
            if data_wav[wav_header.waveform_id]['data'].get(f'trace_{i_channel}') is None:
                data_wav[wav_header.waveform_id]['data'][f'trace_{i_channel}'] = []
            data_wav[wav_header.waveform_id]['data'][f'trace_{i_channel}'].extend(waveform.data[i_channel, :].tolist())

    # Sort the data so that it is time ordered
    for wav_id in data_wav:
        if len(data_wav[wav_id]['data']['Time_tics']) == 0:
            continue

        n_timepoints = len(data_wav[wav_id]['data']['Time_tics'])

        sorted_indices = np.argsort(data_wav[wav_id]['data']['Time_tics'])
        for key in data_wav[wav_id]['data']:
            if len(data_wav[wav_id]['data'][key]) != n_timepoints:
                raise ValueError(f"Waveform ID {wav_id} has inconsistent data length for key '{key}'.")
            data_wav[wav_id]['data'][key] = np.array(data_wav[wav_id]['data'][key])[sorted_indices].tolist()

    # Create output directory
    os.makedirs(path_output, exist_ok=True)

    # Save resp
    trace_ids = [0]
    trace_names = ['RESP']
    fname_output = os.path.join(path_output, f"waveform_RESP.log")
    save_waveform_log_file(2, trace_ids, trace_names, data_wav, fname_output)
    # Todo: Save PhysioTriggerTimes
    # Save Ext1
    trace_ids = [0]
    trace_names = ['EXT1']
    fname_output = os.path.join(path_output, f"waveform_EXT1.log")
    save_waveform_log_file(3, trace_ids, trace_names, data_wav, fname_output)
    # Todo: Save ECG
    # Todo: Save PULSE
    # Todo: Save Ext2


def save_waveform_log_file(logfile_id, trace_ids, trace_names, data_wav, fname_output):
    wav = data_wav[logfile_id]
    with open(fname_output, 'w', encoding='utf-8') as f:
        f.write(f"Time_tics")

        for trace_name in trace_names:
            f.write(f" {trace_name}")
        f.write("\n")

        for i_timepoints in range(len(wav['data']['Time_tics'])):
            f.write(f"{wav['data']['Time_tics'][i_timepoints]}")
            for i_id in range(len(trace_ids)):
                f.write(f" {wav['data'][f'trace_{trace_ids[i_id]}'][i_timepoints]}")
            f.write("\n")


def mrd2nii_volume(metadata, volume_images, skip_sidecar=False):
    # logging.info(f"Header: {header}")
    # Header: version: 1
    # data_type: 5
    # flags: 15
    # measurement_uid: 99
    # matrix_size: 64, 64, 1
    # field_of_view: 300.0, 300.0, 5.0
    # channels: 1
    # position: 0.0, 0.0, -7.5 (Center voxel in LPS coordinates)
    # read_dir: -1.0, 0.0, 0.0
    # phase_dir: 0.0, 1.0, 0.0
    # slice_dir: 0.0, 0.0, 1.0
    # patient_table_position: 0.0, 0.0, -1403.0
    # average: 0
    # slice: 3
    # contrast: 0
    # phase: 0
    # repetition: 4
    # set: 0
    # acquisition_time_stamp: 23375428
    # physiology_time_stamp: 0, 0, 0
    # image_type: 1
    # image_index: 0
    # image_series_index: 0
    # user_int: 0, 0, 0, 0, 0, 0, 0, 0
    # user_float: 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # attribute_string_len: 4580

    # Todo: single slice acquisitions are not supported yet

    logging.info(volume_images[0].getHead())

    # Extract ordering
    nb_slices = int(metadata.encoding[0].encodingLimits.slice.maximum) + 1

    if nb_slices <= 1:
        raise NotImplementedError("Single slice acquisitions are not supported yet")

    if metadata.encoding[0].reconSpace.matrixSize.z != 1:
        raise NotImplementedError("Only  2D acquisitions are supported")

    mrd_idx_to_order_idx = {}
    order_idx_to_mrd_idx = {}

    for param in metadata.userParameters.userParameterLong:
        if param.name.startswith("RelativeSliceNumber_"):
            i_slice = int(param.name.split("_")[-1]) - 1
            order_idx_to_mrd_idx[int(param.value)] = i_slice
            mrd_idx_to_order_idx[i_slice] = int(param.value)

    logging.info(f"mrd_idx_to_order_idx: {mrd_idx_to_order_idx}")
    logging.info(f"order_idx_to_mrd_idx: {order_idx_to_mrd_idx}")

    # Find image on one edge of the FOV
    # Looks like the most inferior slice
    sli_idx = order_idx_to_mrd_idx[0]
    idxbeg_in_volume_images = -1
    for i, image in enumerate(volume_images):
        header = image.getHead()
        if header.slice == sli_idx:
            idxbeg_in_volume_images = i
            break

    # Find image on the other edge of the FOV
    # Looks like the most superior slice
    sli_idx = order_idx_to_mrd_idx[nb_slices - 1]
    idxend_in_volume_images = -1
    for i, image in enumerate(volume_images):
        header = image.getHead()
        if header.slice == sli_idx:
            idxend_in_volume_images = i
            break

    if idxend_in_volume_images < 0 or idxbeg_in_volume_images < 0:
        raise RuntimeError("Can't find start or end idx in provided images")

    matrix = [metadata.encoding[0].reconSpace.matrixSize.x, metadata.encoding[0].reconSpace.matrixSize.y, nb_slices]
    fov = [metadata.encoding[0].reconSpace.fieldOfView_mm.x, metadata.encoding[0].reconSpace.fieldOfView_mm.y,
           metadata.encoding[0].reconSpace.fieldOfView_mm.z]

    # fov in z is only the FOV for the item. In this case, that's only one slice
    # We take the middle of the slice positioning to calculate the FOV in the slice direction
    # Since "position" is the middle of the pixel, we need to add an additional slice to account for the missing half on both sides
    fovz_missing_edges = math.sqrt(((volume_images[idxend_in_volume_images].getHead().position[0] -
                                     volume_images[idxbeg_in_volume_images].getHead().position[0]) ** 2) +
                                   ((volume_images[idxend_in_volume_images].getHead().position[1] -
                                     volume_images[idxbeg_in_volume_images].getHead().position[1]) ** 2) +
                                   ((volume_images[idxend_in_volume_images].getHead().position[2] -
                                     volume_images[idxbeg_in_volume_images].getHead().position[2]) ** 2))
    logging.info(f"fovz_missing_edges: {fovz_missing_edges}")
    pix_dim = [
        fov[0] / matrix[0],
        fov[1] / matrix[1],
        fovz_missing_edges / (matrix[2] - 1)]

    # Add the missing pixel now that we know the pixdim (including slice gap)
    fov[2] = fovz_missing_edges + pix_dim[2]

    rotm = np.zeros((3, 3))
    rotm[:, 0] = [volume_images[idxbeg_in_volume_images].meta['ImageRowDir'][0],
                  volume_images[idxbeg_in_volume_images].meta['ImageRowDir'][1],
                  volume_images[idxbeg_in_volume_images].meta['ImageRowDir'][2]]
    rotm[:, 1] = [volume_images[idxbeg_in_volume_images].meta['ImageColumnDir'][0],
                  volume_images[idxbeg_in_volume_images].meta['ImageColumnDir'][1],
                  volume_images[idxbeg_in_volume_images].meta['ImageColumnDir'][2]]
    rotm[:, 2] = [volume_images[idxbeg_in_volume_images].meta['ImageSliceNormDir'][0],
                  volume_images[idxbeg_in_volume_images].meta['ImageSliceNormDir'][1],
                  volume_images[idxbeg_in_volume_images].meta['ImageSliceNormDir'][2]]

    affine = np.zeros((4, 4))
    affine[:3, 0] = pix_dim[0] * rotm[:, 0]
    affine[:3, 1] = pix_dim[1] * rotm[:, 1]
    affine[:3, 2] = pix_dim[2] * rotm[:, 2]

    # 'SlicePosLightMarker' seems to be more accurate
    if volume_images[0].meta.get('SlicePosLightMarker') is not None:
        im_beg_pos = [float(i) for i in volume_images[idxbeg_in_volume_images].meta.get('SlicePosLightMarker')]
        im_end_pos = [float(i) for i in volume_images[idxend_in_volume_images].meta.get('SlicePosLightMarker')]
    else:
        im_beg_pos = volume_images[idxbeg_in_volume_images].getHead().position
        im_end_pos = volume_images[idxend_in_volume_images].getHead().position

    mid_voxel_coord = [
        (im_beg_pos[0] + im_end_pos[0]) / 2,
        (im_beg_pos[1] + im_end_pos[1]) / 2,
        (im_beg_pos[2] + im_end_pos[2]) / 2
    ]

    mid_voxel_index = np.array(matrix) / 2
    # Some adjustment through experimental testing (maybe due to where the (0,0,0) is defined?)
    # I would have expected needing (-0.5, -0.5, -0.5) everywhere (middle of the corner voxel)
    mid_voxel_index += np.array((0, 1, 0.5))

    logging.info(f"matrix size: {matrix}")
    logging.info(f"mid_voxel_coord: {list(mid_voxel_coord)}")
    # (matrix / 2) @ rotm + translation = mid_voxel_coord

    mid_voxel_index -= np.array([0, matrix[1], matrix[2]])
    translation = mid_voxel_coord - (affine[:3, :3] @ mid_voxel_index)
    logging.info(f"translation: {list(translation)}")

    affine[:3, 3] = translation
    affine[3, 3] = 1

    # Not entirely sure what is going on, it works experimentally
    affine[:2, :] *= -1
    affine[:, 1:3] *= -1
    logging.info(f"Translation idxbeg: {[i for i in volume_images[idxbeg_in_volume_images].getHead().position]}")
    logging.info(f"Translation idxend: {[i for i in volume_images[idxend_in_volume_images].getHead().position]}")

    logging.info(f"pix_dim: {pix_dim}")
    logging.info(f"rotm: {rotm}")
    logging.info(f"matrix size: {matrix}")
    logging.info(f"FOV: {fov}")
    logging.info(f"affine: {affine}")

    # Reconstruct images
    a_volume_image = []
    nb_repetitions = 0
    for i in range(len(volume_images)):
        if nb_repetitions < volume_images[i].getHead().repetition:
            nb_repetitions = volume_images[i].getHead().repetition
        if volume_images[i].getHead().repetition == 0:
            a_volume_image.append(volume_images[i])
    nb_repetitions += 1

    img_metas = []
    for image in volume_images:
        img_metas.append(read_vendor_header_img(image))

    data = np.zeros((matrix[0], matrix[1], nb_slices, nb_repetitions))
    for i_vol, volume in enumerate(volume_images):
        slice = volume.getHead().slice
        repetition = volume.getHead().repetition
        logging.info(f"repetition: {repetition}")

        if img_metas[i_vol]["ProtocolSliceNumber"] != mrd_idx_to_order_idx[slice]:
            raise RuntimeError("ProtocolSliceNumber does not match slice number")

        if img_metas[i_vol]["AcquisitionNumber"] - 1 != repetition:
            raise RuntimeError("Repetition # does not match repetition number")

        # logging.info(f"shape: {volume.data.shape}")
        datatmp = np.flip(volume.data[0, 0, :, :], 0)
        datatmp = np.transpose(datatmp, (1, 0))
        data[:, :, mrd_idx_to_order_idx[slice], repetition] = datatmp

    if data.shape[-1] == 1:
        data = np.squeeze(data, axis=-1)

    nii = nib.Nifti1Image(data, affine=affine)
    if len(nii.shape) > 3:
        nii.header.set_xyzt_units("mm", "sec")
    else:
        nii.header.set_xyzt_units("mm")

    nii.header.set_qform(affine, code=1)
    nii.header.set_sform(affine, code=1)

    if not skip_sidecar:
        sidecar = create_bids_sidecar(metadata, a_volume_image)
    else:
        sidecar = None

    return nii, sidecar


def mrd2nii_stack(metadata, image, include_slice_gap=True):

    header = image.getHead()
    logging.info(header)

    # Extract ordering
    nb_slices = header.matrix_size[-1]

    if metadata.encoding[0].reconSpace.matrixSize.z != 1:
        raise NotImplementedError("Only  2D acquisitions are supported")

    matrix = [metadata.encoding[0].reconSpace.matrixSize.x, metadata.encoding[0].reconSpace.matrixSize.y, nb_slices]

    fov = [metadata.encoding[0].reconSpace.fieldOfView_mm.x, metadata.encoding[0].reconSpace.fieldOfView_mm.y,
           metadata.encoding[0].reconSpace.fieldOfView_mm.z]

    vendor_header = read_vendor_header_img(image)
    slice_gap = vendor_header.get('SpacingBetweenSlices')

    if include_slice_gap:
        vendor_header = read_vendor_header_img(image)
        fov[2] = slice_gap

    pix_dim = [
        fov[0] / matrix[0],
        fov[1] / matrix[1],
        fov[2] / matrix[2]]

    rotm = np.zeros((3, 3))
    rotm[:, 0] = [image.meta['ImageRowDir'][0],
                  image.meta['ImageRowDir'][1],
                  image.meta['ImageRowDir'][2]]
    rotm[:, 1] = [image.meta['ImageColumnDir'][0],
                  image.meta['ImageColumnDir'][1],
                  image.meta['ImageColumnDir'][2]]
    rotm[:, 2] = [image.meta['ImageSliceNormDir'][0],
                  image.meta['ImageSliceNormDir'][1],
                  image.meta['ImageSliceNormDir'][2]]

    affine = np.zeros((4, 4))
    affine[:3, 0] = pix_dim[0] * rotm[:, 0]
    affine[:3, 1] = pix_dim[1] * rotm[:, 1]
    affine[:3, 2] = pix_dim[2] * rotm[:, 2]

    # 'SlicePosLightMarker' seems to be more accurate
    if image.meta.get('SlicePosLightMarker') is not None:
        mid_voxel_coord = [float(i) for i in image.meta.get('SlicePosLightMarker')]
    else:
        mid_voxel_coord = list(np.array(image.position))

    mid_voxel_index = np.array(matrix) / 2
    # Some adjustment through experimental testing (maybe due to where the (0,0,0) is defined?)
    # I would have expected needing (-0.5, -0.5, -0.5) everywhere (middle of the corner voxel)
    mid_voxel_index += np.array((0, 1, 0.5))
    # mid_voxel_index += np.array((0, 0, +0.5))

    logging.info(f"matrix size: {matrix}")
    logging.info(f"mid_voxel_coord: {list(mid_voxel_coord)}")
    # (matrix / 2) @ rotm + translation = mid_voxel_coord

    mid_voxel_index -= np.array([0, matrix[1], matrix[2]])
    translation = mid_voxel_coord - (affine[:3, :3] @ mid_voxel_index)
    logging.info(f"translation: {list(translation)}")

    affine[:3, 3] = translation
    affine[3, 3] = 1

    # Not entirely sure what is going on, it works experimentally
    affine[:2, :] *= -1
    affine[:, 1:3] *= -1

    logging.info(f"pix_dim: {pix_dim}")
    logging.info(f"rotm: {rotm}")
    logging.info(f"matrix size: {matrix}")
    logging.info(f"FOV: {fov}")
    logging.info(f"affine: {affine}")

    # Reconstruct images
    data = np.zeros((matrix[0], matrix[1], nb_slices))

    datatmp = np.flip(image.data[0, 0, :, :], 0)
    datatmp = np.transpose(datatmp, (1, 0))
    data[:, :, 0] = datatmp

    if data.shape[-1] == 1:
        data = np.squeeze(data, axis=-1)

    nii = nib.Nifti1Image(data, affine=affine)
    if len(nii.shape) > 3:
        nii.header.set_xyzt_units("mm", "sec")
    else:
        nii.header.set_xyzt_units("mm")

    nii.header.set_qform(affine, code=1)
    nii.header.set_sform(affine, code=1)

    return nii