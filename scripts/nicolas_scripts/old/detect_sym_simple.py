
import sys, os
import numpy as np
from scipy import misc
import copy
import sct_utils as sct


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    image_directory = args[0]
    list_image_fname = os.listdir(image_directory)

    for image_fname in list_image_fname:

        image = misc.imread(image_directory + "/" + image_fname).astype(int)

        if len(image.shape) > 2:
            image = np.mean(image[:, :, 0:2], axis=2)

        image_axes = np.zeros((image.shape[0], image.shape[1], 3))
        image_axes[:, :, 0] = copy.copy(image)
        image_axes[:, :, 1] = copy.copy(image)
        image_axes[:, :, 2] = copy.copy(image)

        seg_image = (image > np.mean(np.concatenate(image))).astype(int)

        angle_hog, conf_score, centermass = find_angle(image, seg_image, 0.0001, 0.0001, "hog", angle_range=90, return_centermass=True, save_figure_path=image_directory + "/fig_sym_" + image_fname)
        if angle_hog is None:
            angle_hog = 0
        angle_pca, _, centermass = find_angle(image, seg_image, 1, 1, "pca", angle_range=90, return_centermass=True)
        image_axes[:, :, 1] = generate_2Dimage_line(image, centermass[0], centermass[1], 2*pi - angle_hog + pi/2)
        image_axes[:, :, 2] = generate_2Dimage_line(image, centermass[0], centermass[1], 2*pi - angle_pca)

        misc.imsave(image_directory + "/sym_" + image_fname.split(".")[0] + ".png", image_axes, "png")

if __name__ == '__main__':

    sct.init_sct()
    main()
