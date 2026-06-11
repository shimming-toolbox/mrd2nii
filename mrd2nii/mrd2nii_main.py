#!/usr/bin/python3
# -*- coding: utf-8 -*

import json
import logging
import math
import os

import ismrmrd
import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np
import numpy.linalg as npl

from mrd2nii.sidecar import (create_bids_sidecar, read_vendor_header_img, extract_prot_sli_number_to_mrd_index,
                             extract_mrd_index_to_prot_sli_number, get_main_dir)


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
            chunks = extract_in_chunks(dset, group)
            for i_chunk, chunk in enumerate(chunks):
                for i_img in range(len(chunk)):
                    image = chunk[i_img]
                    acq_name = f"{image.getHead().measurement_uid}"
                    # Can't use: read_vendor_header_img(image)['AcquisitionNumber']
                    acq_name += f"_{metadata.measurementInformation.protocolName}"
                    if len(chunks) > 1:
                        acq_name += f"_chunk-{i_chunk + 1}"
                    if image.getHead().image_type == ismrmrd.IMTYPE_MAGNITUDE:
                        acq_name += "_magnitude"
                    elif image.getHead().image_type == ismrmrd.IMTYPE_PHASE:
                        acq_name += "_phase"
                    else:
                        raise NotImplementedError(f"Image type {image.getHead().image_type} not implemented")

                    acq_name += f"_echo-{image.getHead().contrast + 1}"

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


def extract_in_chunks(dset, group):
    """Extract images in chunks.
    A chunk is defined as a set of image that creates a volume when combined together.
    """
    chunks = []
    rotms = []
    for i_img in range(0, dset.number_of_images(group)):
        image = dset.read_image(group, i_img)
        rotm = extract_rot_matrix(image)
        # Check if we have a chunk with the same rotation matrix
        # Todo: This is not bullet proof, if chunks have the same rotation matrix but different locations
        #  then this will fail to extract the chunks correctly.
        #  I have not found a reliable chunk parameter
        found = False
        for i_chunk in range(len(rotms)):
            if np.allclose(rotm, rotms[i_chunk]):
                chunks[i_chunk].append(image)
                found = True
                break
        # If we didn't find a matching chunk, create a new one
        if not found:
            rotms.append(rotm)
            chunks.append([image])
    return chunks


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
            logging.debug(f"Waveform ID {wav_header.waveform_id} not recognized, skipping")
            continue

        if wav_header.number_of_samples == 0:
            continue

        # A time_tic is 2.5 ms, so we need to divide by 2.5ms per time_tic
        time_tic_per_sample = wav_header.sample_time_us / 1000 / 2.5
        time_tics = np.arange(wav_header.time_stamp,
                              wav_header.time_stamp + (wav_header.number_of_samples * time_tic_per_sample),
                              time_tic_per_sample).astype(int)

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
    logging.debug(volume_images[0].getHead())

    # Make sure all slices have the same rotation matrix
    rotm = None
    for i, volume_image in enumerate(volume_images):
        tmp = extract_rot_matrix(volume_image)
        if rotm is None:
            rotm = tmp
        else:
            if not np.allclose(tmp, rotm):
                raise NotImplementedError("Slices have different rotations.")

    # Sort the images in a dictionary with repetition number as key
    images_by_rep_number = {}
    for i, volume_image in enumerate(volume_images):
        rep = volume_image.getHead().repetition
        if rep not in images_by_rep_number:
            images_by_rep_number[rep] = []
        images_by_rep_number[rep].append(volume_image)

    if max(images_by_rep_number.keys()) != len(images_by_rep_number) - 1:
        raise ValueError("Repetition numbers should be consecutive and start from 0.")

    nii = None
    for rep in sorted(images_by_rep_number.keys()):
        nii_volume = None
        for i, volume_image in enumerate(images_by_rep_number[rep]):
            nii_stack = mrd2nii_stack(metadata, volume_image, include_slice_gap=True)
            nii_volume = merge_stacks_into_volume(nii_volume, nii_stack)
        nii = merge_repetitions(nii, nii_volume)

    if not skip_sidecar:
        # Grab all images for a single volume (repetition 0)
        a_volume_image = []
        for i in range(len(volume_images)):
            if volume_images[i].getHead().repetition == 0:
                a_volume_image.append(volume_images[i])

        sidecar = create_bids_sidecar(metadata, a_volume_image, nii.header.get_dim_info())
        if sidecar.get('SliceTiming') is not None:
            # Reorder slice timing if we needed to flip it when creating the Nifti
            if get_main_dir(a_volume_image[0].meta['ImageSliceNormDir']) == 0:
                sidecar['SliceTiming'] = sidecar['SliceTiming'][::-1]
    else:
        sidecar = None

    return nii, sidecar


def mrd2nii_stack(metadata, image, include_slice_gap=True):
    header = image.getHead()
    logging.debug(header)

    # Extract ordering
    nb_slices = header.matrix_size[-1]
    if nb_slices != 1:
        raise NotImplementedError("Non single slice acquisitions are not supported yet")

    mrd_dims_to_nii_dims = np.argsort([get_main_dir(image.getHead().read_dir[:]),
                                       get_main_dir(image.getHead().phase_dir[:]),
                                       get_main_dir(image.getHead().slice_dir[:])])
    nii_dims_to_mrd_dims = np.argsort(mrd_dims_to_nii_dims)

    matrix = [metadata.encoding[0].reconSpace.matrixSize.x,
              metadata.encoding[0].reconSpace.matrixSize.y,
              nb_slices]
    matrix = list(np.array(matrix)[mrd_dims_to_nii_dims])

    fov = [metadata.encoding[0].reconSpace.fieldOfView_mm.x,
           metadata.encoding[0].reconSpace.fieldOfView_mm.y,
           metadata.encoding[0].reconSpace.fieldOfView_mm.z / metadata.encoding[0].reconSpace.matrixSize.z]
    fov = list(np.array(fov)[mrd_dims_to_nii_dims])

    if include_slice_gap:
        vendor_header = read_vendor_header_img(image)

        slice_gap = vendor_header.get('SpacingBetweenSlices')
        if slice_gap is not None:
            fov[nii_dims_to_mrd_dims[2]] = slice_gap

    pix_dim = [
        fov[0] / matrix[0],
        fov[1] / matrix[1],
        fov[2] / matrix[2]]

    rotm = extract_rot_matrix(image)

    affine = np.zeros((4, 4))
    affine[:3, :3] = pix_dim * rotm

    # 'SlicePosLightMarker' seems to be more accurate than image.position
    if image.meta.get('SlicePosLightMarker') is not None:
        mid_voxel_coord = [float(i) for i in image.meta.get('SlicePosLightMarker')]
    else:
        mid_voxel_coord = list(np.array(image.position))

    mid_voxel_index = np.array(matrix) / 2
    # Some adjustment through experimental testing (maybe due to where the (0,0,0) is defined?)
    # I would have expected needing (-0.5, -0.5, -0.5) everywhere (middle of the corner voxel)
    if get_main_dir(image.meta['ImageSliceNormDir']) == 2:
        mid_voxel_index += np.array((0, 1, 0.5))
    elif get_main_dir(image.meta['ImageSliceNormDir']) == 1:
        mid_voxel_index += np.array((0, 0.5, 1))
    elif get_main_dir(image.meta['ImageSliceNormDir']) == 0:
        mid_voxel_index += np.array((-0.5, 1, 1))
    else:
        raise RuntimeError("Slice direction not recognized")

    logging.debug(f"matrix size: {matrix}")
    logging.debug(f"mid_voxel_coord: {list(mid_voxel_coord)}")

    mid_voxel_index -= np.array([0, matrix[1], matrix[2]])
    translation = mid_voxel_coord - (affine[:3, :3] @ mid_voxel_index)
    logging.debug(f"translation: {list(translation)}")

    affine[:3, 3] = translation
    affine[3, 3] = 1

    # Not entirely sure what is going on, it works experimentally
    affine[:2, :] *= -1
    affine[:, 1:3] *= -1

    logging.debug(f"pix_dim: {pix_dim}")
    logging.debug(f"rotm: {rotm}")
    logging.debug(f"matrix size: {matrix}")
    logging.debug(f"FOV: {fov}")
    logging.debug(f"affine: {affine}")

    # Reconstruct images
    data = np.zeros((matrix[0], matrix[1], matrix[2]))

    # logging.debug(f"shape: {volume.data.shape}")
    if get_main_dir(image.meta['ImageSliceNormDir']) == 2:
        datatmp = np.flip(image.data[0, 0, :, :], 0)
        datatmp = np.transpose(datatmp, (1, 0))
        data[:, :, 0] = datatmp
    elif get_main_dir(image.meta['ImageSliceNormDir']) == 1:
        datatmp = np.flip(image.data[0, 0, :, :], 0)
        datatmp = np.transpose(datatmp, (1, 0))
        data[:, 0, :] = datatmp
    elif get_main_dir(image.meta['ImageSliceNormDir']) == 0:
        datatmp = np.flip(image.data[0, 0, :, :], 1)
        datatmp = np.flip(datatmp, 0)
        datatmp = np.transpose(datatmp, (1, 0))
        data[0, :, :] = datatmp
    else:
        raise RuntimeError("Slice direction not recognized")

    nii_tmp = nib.Nifti1Image(data, affine=affine)
    nii_tmp.header.set_dim_info(*mrd_dims_to_nii_dims)
    if len(nii_tmp.shape) > 3:
        nii_tmp.header.set_xyzt_units("mm", "sec")
    else:
        nii_tmp.header.set_xyzt_units("mm")

    nii_tmp.header.set_qform(affine, code=1)
    nii_tmp.header.set_sform(affine, code=1)

    ornt_in = nib.orientations.axcodes2ornt(nib.orientations.aff2axcodes(nii_tmp.affine))
    if get_main_dir(image.meta['ImageSliceNormDir']) == 2:
        ornt_out = nib.orientations.axcodes2ornt(('L', 'A', 'S'))
        ornt = nib.orientations.ornt_transform(ornt_in, ornt_out)
        nii = nii_tmp.as_reoriented(ornt)
        # Looks like there is a bug in nibabel and dim info is not set correctly, we patch it here
        new_dim_info = [None, None, None]
        for i in range(3):
            new_dim_info[int(ornt[i, 0])] = nii_tmp.header.get_dim_info()[i]
        nii.header.set_dim_info(*new_dim_info)
    elif get_main_dir(image.meta['ImageSliceNormDir']) == 1:
        ornt_out = nib.orientations.axcodes2ornt(('L', 'S', 'P'))
        ornt = nib.orientations.ornt_transform(ornt_in, ornt_out)
        nii = nii_tmp.as_reoriented(ornt)
        # Looks like there is a bug in nibabel and dim info is not set correctly, we patch it here
        new_dim_info = [None, None, None]
        for i in range(3):
            new_dim_info[int(ornt[i, 0])] = nii_tmp.header.get_dim_info()[i]
        nii.header.set_dim_info(*new_dim_info)
    elif get_main_dir(image.meta['ImageSliceNormDir']) == 0:
        ornt_out = nib.orientations.axcodes2ornt(('P', 'S', 'R'))
        ornt = nib.orientations.ornt_transform(ornt_in, ornt_out)
        nii = nii_tmp.as_reoriented(ornt)
        # Looks like there is a bug in nibabel and dim info is not set correctly, we patch it here
        new_dim_info = [None, None, None]
        for i in range(3):
            new_dim_info[int(ornt[i, 0])] = nii_tmp.header.get_dim_info()[i]
        nii.header.set_dim_info(*new_dim_info)
    else:
        raise RuntimeError("Slice direction not recognized")

    return nii


def extract_rot_matrix(volume_image):
    # Todo: If volume_image.meta['ImageRowDir']) has 2 values that are maximum (exactly 45 deg rotated I believe), this could fail
    rotm = np.zeros((3, 3))
    rotm[:, get_main_dir(volume_image.meta['ImageRowDir'])] = [volume_image.meta['ImageRowDir'][0],
                                                               volume_image.meta['ImageRowDir'][1],
                                                               volume_image.meta['ImageRowDir'][2]]
    rotm[:, get_main_dir(volume_image.meta['ImageColumnDir'])] = [volume_image.meta['ImageColumnDir'][0],
                                                                  volume_image.meta['ImageColumnDir'][1],
                                                                  volume_image.meta['ImageColumnDir'][2]]
    rotm[:, get_main_dir(volume_image.meta['ImageSliceNormDir'])] = [volume_image.meta['ImageSliceNormDir'][0],
                                                                     volume_image.meta['ImageSliceNormDir'][1],
                                                                     volume_image.meta['ImageSliceNormDir'][2]]
    return rotm


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


def compress_mapping(mapping, compress_keys=True, compress_values=True):
    """Compress mapping to remove gaps in the order indices."""
    # e.g.: {0: 6, 4: 5, 5: 7} -> {0: 1, 1: 0, 2: 2}
    if compress_keys:
        tmp_mapping = {}
        for new_idx, old_idx in enumerate(sorted(mapping.keys())):
            tmp_mapping[new_idx] = mapping[old_idx]
    else:
        tmp_mapping = mapping

    if compress_values:
        compressed_mapping = {}
        sorted_values = sorted(tmp_mapping.values())
        for key, value in tmp_mapping.items():
            compressed_mapping[key] = sorted_values.index(value)
    else:
        compressed_mapping = tmp_mapping

    return compressed_mapping


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
        logging.warning(f"Number of images ({len(volume_images)}) does not match the expected number of images")
        if len(volume_images) % nb_repetitions != 0:
            raise RuntimeError("Error while extracting nb_slices from number of images and repetitions")

    return nb_slices

def get_is_3d(metadata):
    return extract_n_encoding_directions(metadata) > 1


def merge_stacks_into_volume(nii1, nii2):
    # if slices are overlapping, overwrite the slices from nii1 by nii2
    if nii1 is None:
        return nii2

    if nii1.header.get_dim_info() != nii2.header.get_dim_info():
        raise RuntimeError("Can't merge stacks with different dim info")

    if nii1.shape[:2] != nii2.shape[:2]:
        raise RuntimeError("Can't merge stacks with different shapes")

    # If inputs are 2D arrays, convert them to 3D. The zoom needs to be set correctly because it was 1 since it was 2D
    if nii1.ndim == 2:
        nii1 = nib.Nifti1Image(nii1.get_fdata()[..., np.newaxis], affine=nii1.affine, header=nii1.header)
        nii1.header.set_zooms(npl.norm(nii1.affine[:3, :3], axis=0))
    if nii2.ndim == 2:
        nii2 = nib.Nifti1Image(nii2.get_fdata()[..., np.newaxis], affine=nii2.affine, header=nii2.header)
        nii2.header.set_zooms(npl.norm(nii2.affine[:3, :3], axis=0))

    if not np.allclose(nii1.header.get_zooms(), nii2.header.get_zooms()):
        raise RuntimeError("Can't merge stacks with different pixel dimensions")

    pix_dim = nii1.header.get_zooms()
    # Find location of [0,0,:] (ie first voxel of each slice)
    slice_loc1 = []
    for i in range(nii1.shape[2]):
        slice_loc1.append(apply_affine(nii1.affine, [0, 0, i]))  # Location of the first voxel of the slice
    slice_loc2 = []
    for i in range(nii2.shape[2]):
        slice_loc2.append(apply_affine(nii2.affine, [0, 0, i]))  # Location of the first voxel of the slice
    all_slice_loc = slice_loc1 + slice_loc2

    master_n_slices = -1
    for i in range(len(all_slice_loc)):
        for j in range(len(all_slice_loc)):
            # Find the distance between the two slice locations then divide by slice dimension
            # We add 1 since we want the number of slices
            n_slices = np.sqrt((all_slice_loc[i][0] - all_slice_loc[j][0]) ** 2 +
                               (all_slice_loc[i][1] - all_slice_loc[j][1]) ** 2 +
                               (all_slice_loc[i][2] - all_slice_loc[j][2]) ** 2) / pix_dim[2] + 1

            if master_n_slices < n_slices:
                master_n_slices = round(n_slices)

    logging.debug(f"master_n_slices: {master_n_slices}")

    # Find which affine is the master affine
    # ie the one that when used with the other nii does not return negative indices
    # OR both affines are the same...
    inv_aff2 = npl.inv(nii2.affine)
    k_indx_1 = []
    for i in range(nii1.shape[2]):
        k_indx_1.append(round(apply_affine(inv_aff2, slice_loc1[i])[2]))
    inv_aff1 = npl.inv(nii1.affine)
    k_indx_2 = []
    for i in range(nii2.shape[2]):
        k_indx_2.append(round(apply_affine(inv_aff1, slice_loc2[i])[2]))

    master_affine = None
    if np.any(np.array(k_indx_1) < 0):
        master_affine = nii1.affine
        k_indx_1 = list(range(nii1.shape[2]))
    if np.any(np.array(k_indx_2) < 0):
        master_affine = nii2.affine
        k_indx_2 = list(range(nii2.shape[2]))
    if np.allclose(nii1.affine, nii2.affine):
        master_affine = nii1.affine
    if master_affine is None:
        raise RuntimeError("Can't find master affine")
    if not np.all(np.array(k_indx_1) >= 0) and not np.all(np.array(k_indx_2) >= 0):
        raise RuntimeError("Both affines return negative indices, can't find master affine")

    # Initialize a master array with an array encompassing both stacks (ie with enough slices to fit both stacks)
    master_shape = (nii1.shape[0], nii1.shape[1], master_n_slices)
    master_data = np.full(master_shape, np.nan)  # Using NaN helps track empty gaps

    # Fill the master array
    for i in range(nii1.shape[2]):
        master_data[..., k_indx_1[i]] = nii1.dataobj[:, :, i]
    for i in range(nii2.shape[2]):
        master_data[..., k_indx_2[i]] = nii2.dataobj[:, :, i]

    master_data = np.nan_to_num(master_data, nan=0.0)
    nii_merged = nib.Nifti1Image(master_data, affine=master_affine, header=nii1.header)

    return nii_merged


def merge_repetitions(nii, nii_volume):
    if nii is None:
        return nii_volume

    if nii.ndim == 3:
        data_nii = nii.get_fdata()[..., np.newaxis]
    else:
        data_nii = nii.get_fdata()
    data = np.concatenate((data_nii, nii_volume.get_fdata()[..., np.newaxis]), axis=3)
    nii_merged = nib.Nifti1Image(data, nii.affine, header=nii.header)

    return nii_merged