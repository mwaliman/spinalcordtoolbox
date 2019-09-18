
import sys, os
import numpy as np
import imageio
import copy
import sct_utils as sct
from msct_register import find_angle_hog, compute_pca
from nicolas_scripts.functions_sym_rot import generate_2Dimage_line
import matplotlib.pyplot as plt


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    image_path = args[0]
    segmentation_path = args[1]
    if segmentation_path != "None":
        segmentation = imageio.imread(segmentation_path)
    else:
        segmentation = None
    image = imageio.imread(image_path)

    if len(image.shape) > 2:
        image = np.mean(image[:, :, 0:2], axis=2)
        if segmentation is not None:
            segmentation = np.array(np.mean(segmentation[:, :, 0:2], axis=2) > 50, dtype="float64")

    image_axes = np.zeros((image.shape[0], image.shape[1], 3))
    image_axes[:, :, 1] = copy.copy(image)
    image_axes[:, :, 2] = copy.copy(image)

    #find centermass
    if segmentation is not None:
        _, _, centermass = compute_pca(segmentation)
    else:
        centermass = (image.shape[0]//2, image.shape[1]//2)

    angle, conf_score = find_angle_hog(image, centermass, 0.00000000001, 0.0000000001, angle_range=90)
    image_axes[:, :, 0] = generate_2Dimage_line(copy.copy(image), image.shape[0]//2, image.shape[1]//2, np.pi/2 - angle)

    imageio.imsave("/home/nicolas/sym_image.png", image_axes, "png")


if __name__ == '__main__':

    sct.init_sct()
    main()


# to have in msct register to plot things

# import matplotlib.pyplot as plt
# from matplotlib.colors import hsv_to_rgb
# plt.figure(figsize=(20, 10))
# plt.subplot(221)
# plt.imshow(np.max(image)- image, cmap='Greys')
# plt.title("image")
# plt.xlabel("angle in degrees")
# plt.subplot(222)
# plt.imshow(orient, cmap="hsv")
# plt.colorbar()
# plt.title("gradient orientation")
# plt.subplot(223)
# plt.imshow(weighting_map.astype("float64"), cmap="jet")
# plt.title("final weighting map")
# plt.colorbar()
# plt.subplot(224)
# plt.imshow(1 - grad_mag.astype("float64"), cmap="Greys")
# plt.title("gradient magnitude")
# plt.colorbar()
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(20, 10))
# plt.subplot(231)
# plt.plot(repr_hist*180/np.pi, grad_orient_histo)
# plt.title("gradient orientation histogram")
# plt.xlabel("angle in degrees")
# plt.subplot(232)
# plt.plot(repr_hist*180/np.pi, grad_orient_histo_smooth)
# plt.title("gradient orientation histogram smoothed")
# plt.xlabel("angle in degrees")
# plt.subplot(233)
# plt.plot(repr_hist*90/np.pi, grad_orient_histo_conv)
# plt.title("circular convolution of the GOH smoothed")
# plt.xlabel("angle in degrees")
# plt.subplot(234)
# plt.imshow(255 - image, cmap="Greys")
# plt.title("image")
# plt.subplot(235)
# plt.imshow(seg_weighted_mask)
# plt.title("segmentation gaussian weighted map")
# plt.colorbar()
