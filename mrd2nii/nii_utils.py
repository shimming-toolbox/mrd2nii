#!/usr/bin/python3
# -*- coding: utf-8 -*

import copy
import nibabel as nib
import numpy as np


def orient_nii_to(nii: nib.Nifti1Image, target_dim_info=(0, 1, 2)) -> nib.Nifti1Image:

    if np.any(np.array(target_dim_info)) < 0 or np.any(np.array(target_dim_info) > 2):
        raise ValueError("Axis must be 0, 1, or 2")

    affine = nii.affine
    dim_info = list(nii.header.get_dim_info())
    data = np.asanyarray(nii.dataobj)

    new_dim_info = copy.deepcopy(dim_info)
    for i_dim, dim in enumerate(target_dim_info):
        if dim_info[i_dim] != dim:
            index_in = dim_info.index(dim)
            index_out = i_dim
            # Swap axis in the array
            data = np.swapaxes(data, index_in, index_out)
            # Affine must change
            affine[:3, [index_in, index_out]] = affine[:3, [index_out, index_in]]
            # Update dim_info
            new_dim_info[index_in] = dim_info[index_out]
            new_dim_info[index_out] = dim_info[index_in]
            dim_info = copy.deepcopy(new_dim_info)

    nii.header.set_dim_info(*dim_info)
    nii_oriented = nib.Nifti1Image(data, affine, header=nii.header)

    return nii_oriented


# def flip_nii_axis(nii: nib.Nifti1Image, axis: int) -> nib.Nifti1Image:
#     if axis < 0 or axis > 2:
#         raise ValueError("Axis must be 0, 1, or 2")
#
#     affine = nii.affine
#     dim_info = list(nii.header.get_dim_info())
#     data = np.asanyarray(nii.dataobj)
#
#     # Flip the data along the specified axis
#     data = np.flip(data, axis=axis)
#
#     # Update the affine to reflect the flip
#     affine[axis, 3] = affine[axis, 3] + affine[axis, axis] * (data.shape[axis] - 1)
#     affine[axis, axis] = -affine[axis, axis]
#
#     nii_flipped = nib.Nifti1Image(data, affine, header=nii.header)
#
#     return nii_flipped


def orient_nii_to_voxel_ras(nii, target_ax_code_in=('R', 'A', 'S')):
    ax_code_in = nib.orientations.aff2axcodes(nii.affine)
    ornt_in = nib.orientations.axcodes2ornt(ax_code_in)
    nii_out = nii.as_reoriented(ornt_in)
    return nii_out
