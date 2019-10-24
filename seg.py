import os, sys, logging
import glob
import nibabel as nib
import numpy as np
import argparse

from spinalcordtoolbox.image import Image, change_type, zeros_like
from spinalcordtoolbox.deepseg_sc.core import find_centerline, crop_image_around_centerline
from spinalcordtoolbox.deepseg_sc.core import post_processing_slice_wise, apply_intensity_normalization
from spinalcordtoolbox.deepseg_sc.cnn_models import nn_architecture_seg
import sct_utils as sct
from spinalcordtoolbox import resampling


def preprocess_image(image, contrast_type='t1', ctr_algo='svm', ctr_file=None, brain_bool=True,
								 kernel_size='2d', remove_temp_files=1, verbose=1):
	""" Resamples, reorients to RPI, and applies OptiC cropping to an Image and returns the result as an sct Image.
	Inputs:
		image - Image to be cropped
	Returns:
		im_nii - resampled Image
		im_norm_in - resampled, cropped, and normalized Imagect
		X_CROP_LST, Y_CROP_LST, Z_CROP_LST - coordinates for cropping original image
	"""
	
	im = image.copy()
	
	# create temporary folder with intermediate results
	tmp_folder = sct.TempFolder(verbose=verbose)
	tmp_folder_path = tmp_folder.get_path()
	if ctr_algo == 'file':  # if the ctr_file is provided
		tmp_folder.copy_from(ctr_file)
		file_ctr = os.path.basename(ctr_file)
	else:
		file_ctr = None
	tmp_folder.chdir()

	# re-orient image to RPI if necessary...
	original_orientation = im.orientation
	fname_orient = 'image_in_RPI.nii'
	im.change_orientation('RPI').save(fname_orient)

	input_resolution = im.dim[4:7]

	# resamples image to 0.5x0.5 resolution and finds the spinal cord centerline - execute OptiC binary
	fname_res, centerline_filename, im_labels = find_centerline(algo=ctr_algo,
													 image_fname=fname_orient,
													 contrast_type=contrast_type,
													 brain_bool=brain_bool,
													 folder_output=tmp_folder_path,
													 remove_temp_files=remove_temp_files,
													 centerline_fname=file_ctr)
	# could save the ctr_nii later if desired
	im_nii, ctr_nii = Image(fname_res), Image(centerline_filename)
	
	# crop image around the spinal cord centerline
	crop_size = 96 if (kernel_size == '3d' and contrast_type == 't2s') else 64
	X_CROP_LST, Y_CROP_LST, Z_CROP_LST, im_crop_nii = crop_image_around_centerline(im_in=im_nii,
																				   ctr_in=ctr_nii,
																				   crop_size=crop_size)
	# normalize the intensity of the images
	im_norm_in = apply_intensity_normalization(im_in=im_crop_nii)
	return im_nii, im_norm_in, X_CROP_LST, Y_CROP_LST, Z_CROP_LST

def segment_2d_slices(image, seg_model, binary_seg=True, threshold=0.5):
	"""Applies seg_model on 2d slices of a cropped Image.
	
	Inputs:
		image - Image to be segmented
		seg_model - 2d segmentation model
		binary - whether the segmentation is binary or partial
		threshold - threshold for binary segmentation
	Returns:
		seg_crop - output segmentation as an Image
	"""
	cropped_seg = zeros_like(image)
	cropped_seg_data = np.zeros(image.data.shape)

	data_norm = image.data
	x_cOm, y_cOm = None, None #??
	for z in range(data_norm.shape[2]):
		pred_seg = seg_model.predict(np.expand_dims(np.expand_dims(data_norm[:, :, z], -1), 0),
									 batch_size=BATCH_SIZE)[0, :, :, 0]
		if binary_seg:
			pred_seg_th = (pred_seg > threshold).astype(int)
			pred_seg_pp = post_processing_slice_wise(pred_seg_th, x_cOm, y_cOm)
		else:
			pred_seg_pp = pred_seg
		cropped_seg_data[:, :, z] = pred_seg_pp
	cropped_seg.data = cropped_seg_data
	return cropped_seg

def uncrop_image(ref_in, data_crop, X_CROP_LST, Y_CROP_LST, Z_CROP_LST):
	""" Reconstructs the segmentation from cropped seg_data and returns as an sct Image.
	
	Inputs:
		ref_in - original reference Image with correct dimensions
		data_crop - cropped segmentation data
		X_CROP_LST, Y_CROP_LST, Z_CROP_LST - coordinates for cropping original image
	Returns:
		seg_uncrop - uncropped Image
	"""
	seg_uncrop = zeros_like(ref_in, dtype=np.float32)
	crop_size_x, crop_size_y = data_crop.shape[:2]
	for i_z, zz in enumerate(Z_CROP_LST):
		pred_seg = data_crop[:, :, zz]
		x_start, y_start = int(X_CROP_LST[i_z]), int(Y_CROP_LST[i_z])
		x_end = x_start + crop_size_x if x_start + crop_size_x < seg_uncrop.dim[0] else seg_uncrop.dim[0]
		y_end = y_start + crop_size_y if y_start + crop_size_y < seg_uncrop.dim[1] else seg_uncrop.dim[1]
		seg_uncrop.data[x_start:x_end, y_start:y_end, zz] = pred_seg[0:x_end - x_start, 0:y_end - y_start]
	return seg_uncrop

if __name__ == "__main__":
	parser = argparse.ArgumentParser('This script has 3 inputs: An input directory of niftis, an output directory, and the path to a weights file.')
	parser.add_argument("input_dir")
	parser.add_argument("output_dir")
	parser.add_argument("seg_model_fname")
	parser.add_argument("contrast_type")
	parser.add_argument("binary_seg")
	args = parser.parse_args()

	input_dir = args.input_dir
	output_dir = args.output_dir
	seg_model_fname = args.seg_model_fname
	contrast_type = args.contrast_type
	binary_seg = True if args.binary_seg == 'True' else False

 	# build model
	BATCH_SIZE = 4
	input_size = (64,64)

	seg_model = nn_architecture_seg(height=input_size[0],
										width=input_size[1],
										depth=2,
										features=32,
										batchnorm=True,
										dropout=0.0)
	seg_model.load_weights(seg_model_fname)

	# segment image
	for name in os.listdir(input_dir):
		if not '.DS_Store' in name:
			fname_image = input_dir + name
			parent, stem, ext = sct.extract_fname(fname_image)

			# import image
			image = Image(fname_image)
			image.save(output_dir+ stem + '_im' + ext)

			# crop image
			resampled_image, cropped_image, X_CROP_LST, Y_CROP_LST, Z_CROP_LST = preprocess_image(image, contrast_type=contrast_type)
			resampled_image.save(output_dir+ stem + '_res_im' + ext)
			if contrast_type == 't2':
				cropped_image.data = 255.0 - cropped_image.data
			cropped_image.save(output_dir+ stem + '_im_crop' + ext)

			# segment 
			cropped_seg = segment_2d_slices(cropped_image, seg_model,binary_seg=binary_seg)
			cropped_seg.save(output_dir+ stem + '_seg_crop' + ext)

			# uncrop segmentation
			uncropped_seg = uncrop_image(resampled_image, cropped_seg.data, X_CROP_LST, Y_CROP_LST, Z_CROP_LST)
			uncropped_seg.save(output_dir+ stem + '_seg' + ext)

