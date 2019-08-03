
import sys, os
import numpy as np
import imageio
import copy
import sct_utils as sct
from msct_register import find_angle_hog
from nicolas_scripts.functions_sym_rot import generate_2Dimage_line


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    image_directory = args[0]
    list_image_fname = os.listdir(image_directory)

    for image_fname in list_image_fname:

        image = imageio.imread(image_directory + "/" + image_fname)

        if len(image.shape) > 2:
            image = np.mean(image[:, :, 0:2], axis=2)

        image_axes = np.zeros((image.shape[0], image.shape[1], 3))
        image_axes[:, :, 1] = copy.copy(image)
        image_axes[:, :, 2] = copy.copy(image)

        seg_image = (image > np.mean(np.concatenate(image)))
        # TODO do seg of the image

        angle, conf_score = find_angle_hog(image, (image.shape[0]//2, image.shape[1]//2), 0.0001, 0.0001, angle_range=90)
        image_axes[:, :, 0] = generate_2Dimage_line(copy.copy(image), image.shape[0]//2, image.shape[1]//2, angle)

        imageio.imsave(image_directory + "/sym_" + image_fname.split(".")[0] + ".png", image_axes, "png")


if __name__ == '__main__':

    sct.init_sct()
    main()
