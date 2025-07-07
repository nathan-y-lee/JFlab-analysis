from import_data import import_data
from img_utils import crop_with_buffer, plot_img
from jiaoyang_analysis.config import data_dir
import numpy as np
from skimage import filters, measure, morphology

img, roi = import_data('selecttest3', data_dir)
zproj = np.max(img, axis=0) # Z-projection with maximum intensity

mask = roi > 0 # convert to binary mask
zproj_roi = np.where(mask, zproj, 0) # apply mask to image
zproj_roi_cropped, coords = crop_with_buffer(zproj_roi, buffer=200)
zproj_cropped = zproj[:, coords[0]:coords[1], coords[2]:coords[3]] # crop the same region from the original slice

microglia_cropped = zproj_cropped[0, :, :]
lysosome_cropped = zproj_cropped[2, :, :]
microglia_roi_cropped = zproj_roi_cropped[0, :, :]
lysosome_roi_cropped = zproj_roi_cropped[2, :, :]

# Detect microglia

block_size = 51 # size of the local neighborhood; must be odd
offset = -0.15 # adjusts sensitivity: greater = stricter
local_thresh = filters.threshold_local(microglia_cropped, block_size=block_size, offset=offset)
binary = microglia_roi_cropped > local_thresh

binary = morphology.remove_small_objects(binary, min_size=200)  # remove tiny noise
binary = morphology.remove_small_holes(binary, area_threshold=300)

# Assign labels to connected components
labels = measure.label(binary)
props = measure.regionprops(labels)
print("Detected", len(props), "objects")

plot_img([microglia_cropped, microglia_roi_cropped, binary])

