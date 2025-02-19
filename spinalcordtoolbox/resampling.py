#########################################################################################
#
# Resample data using nibabel.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Sara Dupont
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: remove resample_file (not needed)

from __future__ import division, absolute_import

import logging
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to


import sct_utils as sct

logger = logging.getLogger(__name__)


def resample_nib(img, new_size=None, new_size_type=None, img_dest=None, interpolation='linear'):
    """
    Resample a nibabel image object based on a specified resampling factor.
    Can deal with 2d, 3d or 4d image objects.

    :param img: nibabel Image.
    :param new_size: list of float: Resampling factor, final dimension or resolution, depending on new_size_type.
    :param new_size_type: {'vox', 'factor', 'mm'}: Feature used for resampling. Examples:
      new_size=[128, 128, 90], new_size_type='vox' --> Resampling to a dimension of 128x128x90 voxels
      new_size=[2, 2, 2], new_size_type='factor' --> 2x isotropic upsampling
      new_size=[1, 1, 5], new_size_type='mm' --> Resampling to a resolution of 1x1x5 mm
    :param img_dest: Destination nibabel Image to resample the input image to. In this case, new_size and new_size_type
    are ignored
    :param interpolation: {'nn', 'linear', 'spline'}. The interpolation type
    :return: The resampled nibabel Image.
    """

    # set interpolation method
    dict_interp = {'nn': 0, 'linear': 1, 'spline': 2}

    if img_dest is None:
        # Get dimensions of data
        p = img.header.get_zooms()
        shape = img.header.get_data_shape()

        if img.ndim == 4:
            new_size += ['1']  # needed because the code below is general, i.e., does not assume 3d input and uses img.shape

        # compute new shape based on specific resampling method
        if new_size_type == 'vox':
            shape_r = tuple([int(new_size[i]) for i in range(img.ndim)])
        elif new_size_type == 'factor':
            if len(new_size) == 1:
                # isotropic resampling
                new_size = tuple([new_size[0] for i in range(img.ndim)])
            # compute new shape as: shape_r = shape * f
            shape_r = tuple([int(np.round(shape[i] * float(new_size[i]))) for i in range(img.ndim)])
        elif new_size_type == 'mm':
            if len(new_size) == 1:
                # isotropic resampling
                new_size = tuple([new_size[0] for i in range(img.ndim)])
            # compute new shape as: shape_r = shape * (p_r / p)
            shape_r = tuple([int(np.round(shape[i] * float(p[i]) / float(new_size[i]))) for i in range(img.ndim)])
        else:
            logger.error('new_size_type is not recognized.')

        # Generate 3d affine transformation: R
        affine = img.affine[:4, :4]
        affine[3, :] = np.array([0, 0, 0, 1])  # satisfy to nifti convention. Otherwise it grabs the temporal
        logger.debug('Affine matrix: \n' + str(affine))
        R = np.eye(4)
        for i in range(3):
            R[i, i] = img.shape[i] / float(shape_r[i])
        affine_r = np.dot(affine, R)
        reference = (shape_r, affine_r)
    else:
        # If reference is provided
        reference = img_dest

    if img.ndim == 3:
        img_r = resample_from_to(
            img, to_vox_map=reference, order=dict_interp[interpolation], mode='constant', cval=0.0, out_class=None)

    elif img.ndim == 4:
        # TODO: Cover img_dest with 4D volumes
        # Import here instead of top of the file because this is an isolated case and nibabel takes time to import
        data4d = np.zeros(shape_r)
        # Loop across 4th dimension and resample each 3d volume
        for it in range(img.shape[3]):
            # Create dummy 3d nibabel image
            nii_tmp = nib.nifti1.Nifti1Image(img.get_data()[..., it], affine)
            img3d_r = resample_from_to(
                nii_tmp, to_vox_map=(shape_r[:-1], affine_r), order=dict_interp[interpolation], mode='constant',
                cval=0.0, out_class=None)
            data4d[..., it] = img3d_r.get_data()
        # Create 4d nibabel Image
        img_r = nib.nifti1.Nifti1Image(data4d, affine_r)

    return img_r


def resample_file(fname_data, fname_out, new_size, new_size_type, interpolation, verbose, fname_ref=None):
    """This function will resample the specified input
    image file to the target size.
    Can deal with 2d, 3d or 4d image objects.
    :param fname_data: The input image filename.
    :param fname_out: The output image filename.
    :param new_size: The target size, i.e. 0.25x0.25
    :param new_size_type: Unit of resample (mm, vox, factor)
    :param interpolation: The interpolation type
    :param verbose: verbosity level
    :param fname_ref: Reference image to resample input image to
    """
    # Load data
    logger.info('load data...')
    nii = nib.load(fname_data)
    if fname_ref is not None:
        nii_ref = nib.load(fname_ref)
    else:
        nii_ref = None

    nii_r = resample_nib(nii, new_size.split('x'), new_size_type, img_dest=nii_ref, interpolation=interpolation)

    # build output file name
    if fname_out == '':
        fname_out = sct.add_suffix(fname_data, '_r')
    else:
        fname_out = fname_out

    # save data
    nib.save(nii_r, fname_out)

    # to view results
    sct.display_viewer_syntax([fname_out], verbose=verbose)

    return nii_r

