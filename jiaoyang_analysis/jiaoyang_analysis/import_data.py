import tifffile

def import_data(filename, data_dir):
    """
    Import .tif img and ROI from the data_dir directory.
    """
    # Load the image data
    img = tifffile.imread(data_dir / f'{filename}.tif')

    # Load the ROI mask
    roi = tifffile.imread(data_dir / f'{filename}roi.tif')

    return img, roi