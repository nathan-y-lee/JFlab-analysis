from import_data import import_data
from img_utils import crop_with_buffer, plot_img, detect_objects
from jiaoyang_analysis.config import data_dir
import numpy as np
from skimage import filters, measure, morphology

img, roi = import_data('selecttest3', data_dir)
zproj = np.max(img, axis=0) # z-projection with maximum intensity

mask = roi > 0 # convert to binary mask
zproj_roi = np.where(mask, zproj, 0) # apply mask to image
zproj_roi_cropped, coords = crop_with_buffer(zproj_roi, buffer=200)
zproj_cropped = zproj[:, coords[0]:coords[1], coords[2]:coords[3]] # crop the same region from the original slice

mg_cropped = zproj_cropped[0, :, :] # microglia channel (IBA1)
lys_cropped = zproj_cropped[2, :, :] # lysosome channel (CB68)
mg_roi_cropped = zproj_roi_cropped[0, :, :]
lys_roi_cropped = zproj_roi_cropped[2, :, :]

# Detect microglia
mg_binary, mg_labels, mg_props = detect_objects(mg_cropped, mg_roi_cropped, block_size=51, offset=-0.15, min_size=200, area_threshold=300)
plot_img([mg_cropped, mg_roi_cropped, mg_binary])

# Detect lysosomes
lys_binary, lys_labels, lys_props = detect_objects(lys_cropped, lys_roi_cropped, block_size=15, offset=-10.0, min_size=10, area_threshold=50)
plot_img([lys_cropped, lys_roi_cropped, lys_binary])

# Detect overlapping microglia and lysosomes
overlap_binary = mg_binary & lys_binary
overlap_labels, overlap_props = measure.label(overlap_binary, return_num=True)
print("Detected", overlap_props, "overlapping objects")
plot_img([mg_labels])

overlap_mg_labels = np.unique(mg_labels[lys_binary])
overlap_mg_binary = ~np.isin(mg_labels, overlap_mg_labels)
plot_img([mg_binary, overlap_mg_binary])