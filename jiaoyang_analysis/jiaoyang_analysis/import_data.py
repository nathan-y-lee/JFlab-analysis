import tifffile

def import_data(filepath, roi_prefix):
    img = tifffile.imread(f'{filepath}.tif')
    roi = tifffile.imread(f'{roi_prefix}{filepath}.tif')

    return img, roi

def batch_import_data(data_dir, roi_prefix):
    """
    Imports all .tif images and their corresponding ROI masks from the given directory.
    
    ROI files must be named with the format: roi_prefix + filename.tif
    For example, for 'roi_dg_granule', ROI of 'image1.tif' should be 'roi_dg_granuleimage1.tif'.

    Returns:
        A list of tuples: (filename (str), image (ndarray), roi (ndarray))
    """
    data = []
    # Iterate over all .tif files that are NOT ROI files
    for roi_file in data_dir.glob("*.tif"):
        if roi_file.name.startswith(roi_prefix):
            img_file = data_dir / f"{roi_file.name.removeprefix(roi_prefix)}"
            
            if roi_file.exists():
                img = tifffile.imread(img_file)
                roi = tifffile.imread(roi_file)
                data.append((img_file.name, img, roi))
                print(f"Loaded: {img_file.name} with ROI {roi_file.name}")
            else:
                print(f"[Warning] ROI not found for {roi_file.name}, skipping.")

    return data