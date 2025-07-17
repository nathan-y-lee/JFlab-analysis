from img_utils import crop_with_buffer, plot_img, detect_objects
import numpy as np
import pandas as pd

def analyze_image(filename, img, roi):
    """
    Analyze a single image and its ROI, returning a DataFrame row with results.
    """
    zproj = np.max(img, axis=0) # z-projection with maximum intensity
    mask = roi > 0 # convert to binary mask
    zproj_roi = np.where(mask, zproj, 0) # apply mask to image
    zproj_roi_cropped, coords = crop_with_buffer(zproj_roi, buffer=200)
    zproj_cropped = zproj[:, coords[0]:coords[1], coords[2]:coords[3]] # crop the same region from the original slice

    mg_cropped = zproj_cropped[0, :, :] # microglia channel (IBA1)
    lys_cropped = zproj_cropped[2, :, :] # lysosome channel (CB68)
    mg_roi_cropped = zproj_roi_cropped[0, :, :]
    lys_roi_cropped = zproj_roi_cropped[2, :, :]

    # Detect microglia and lysosomes
    mg_binary, mg_labels, mg_props = detect_objects(mg_cropped, mg_roi_cropped, block_size=51, offset=-0.15, min_size=200, area_threshold=300)
    lys_binary, lys_labels, lys_props = detect_objects(lys_cropped, lys_roi_cropped, block_size=15, offset=-15.0, min_size=10, area_threshold=50)

    # Detect overlapping microglia and lysosomes
    overlap_lys_labels = np.unique(lys_labels[mg_binary]) # find overlapping lysosomes
    overlap_lys_binary = ~np.isin(lys_labels, overlap_lys_labels)
    overlap_mg_labels = np.unique(mg_labels[lys_binary]) # find overlapping microglia
    overlap_mg_binary = ~np.isin(mg_labels, overlap_mg_labels)
    print(f"Overlapping lysosomes: {len(overlap_lys_labels)}\n"
        f"Overlapping microglia: {len(overlap_mg_labels)}")

    plot_img([mg_binary, lys_binary, overlap_mg_binary, overlap_lys_binary])

    result = pd.DataFrame([{
        'Image No.': filename,
        'ROI Area (pixels)': np.count_nonzero(roi),
        'Microglia Area' : np.count_nonzero(mg_binary),
        'Lysosome Area': np.count_nonzero(lys_binary),
        'Overlap Area' : np.count_nonzero(mg_binary & lys_binary),
        'Microglia Count': len(mg_props),
        'Lysosome Count': len(lys_props),
        'Overlapping Microglia Count': len(overlap_mg_labels),
        'Overlapping Lysosome Count': len(overlap_lys_labels),
        'Microglia Intensity': np.sum(mg_roi_cropped * mg_binary),
        'Lysosome Intensity': np.sum(lys_roi_cropped * lys_binary),
        'Overlapping Microglia Intensity': np.sum(overlap_mg_binary * mg_binary),
        'Overlapping Lysosome Intensity': np.sum(overlap_lys_binary * lys_binary)
    }])

    return result

def simple_analyze_image(filename, img, roi):
    """
    A simplified version of analyze_image identical to ImageJ
    """
    zproj = np.max(img, axis=0) # z-projection with maximum intensity
    mask = roi > 0 # convert to binary mask
    zproj_roi = np.where(mask, zproj, 0) # apply mask to image

    mg = zproj_roi[0, :, :] # microglia channel (IBA1)
    lys = zproj_roi[2, :, :] # lysosome channel (CB68)

    threshold_mg = 60
    threshold_lys = 104

    mg_thresh = np.where(mg > threshold_mg, mg, 0)
    lys_thresh = np.where(lys > threshold_lys, lys, 0)

    overlap_mask = (mg_thresh > 0) & (lys_thresh > 0)
    lys_overlap_thresh = lys_thresh[overlap_mask]

    result = pd.DataFrame([{
        'Image No.': filename,
        'ROI Area (pixels)': np.count_nonzero(roi),
        'Microglia Area': np.count_nonzero(mg_thresh),
        'Lysosome Area': np.count_nonzero(lys_thresh),
        'Microglia Intensity': np.sum(mg_thresh),
        'Lysosome Intensity': np.sum(lys_thresh),
        'Lysosome In Microglia Intensity': np.sum(lys_overlap_thresh),
    }])

    return result