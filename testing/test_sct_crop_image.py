#!/usr/bin/env python
#########################################################################################
#
# Test function for sct_crop_image
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Cohen-Adad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from __future__ import absolute_import

from spinalcordtoolbox.image import Image


def init(param_test):
    """
    Initialize class: param_test
    """
    # initialization
    default_args = ['-i t2/t2.nii.gz -o cropped_normal.nii.gz -dim 1 -start 10 -end 50']
    # assign default params
    if not param_test.args:
        param_test.args = default_args
    return param_test


def test_integrity(param_test):
    """
    Test integrity of function
    """
    # check if cropping was correct
    nx, ny, nz, nt, px, py, pz, pt = Image('cropped_normal.nii.gz').dim
    if ny != 41:
        param_test.status = 99
        param_test.output += '--> FAILED'
    else:
        param_test.output += '--> PASSED'
    return param_test
