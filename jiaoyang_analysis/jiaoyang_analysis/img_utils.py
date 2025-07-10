import numpy as np

def detect_objects(img, roi, block_size, offset, min_size, area_threshold):
    """
    Detect objects in a 2D image using local thresholding and morphological operations.
    """
    from skimage import filters, measure, morphology
    
    local_thresh = filters.threshold_local(img, block_size=block_size, offset=offset)
    binary = roi > local_thresh

    # Remove small objects and holes
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    binary = morphology.remove_small_holes(binary, area_threshold=area_threshold)

    # Assign labels to connected components
    labels = measure.label(binary)
    props = measure.regionprops(labels)

    print("Detected", len(props), "objects")

    return binary, labels, props

def crop_with_buffer(img, buffer):
    """
    Crop a 3D ROI (C, W, H) to the bounding box of non-zero pixels with a specified buffer.
    """
    y, x = np.nonzero(img[0, :, :])
    y_min = max(y.min() - buffer, 0)
    y_max = min(y.max() + buffer, img.shape[1])
    x_min = max(x.min() - buffer, 0)
    x_max = min(x.max() + buffer, img.shape[2])

    return img[:, y_min:y_max, x_min:x_max], (y_min, y_max, x_min, x_max)

def plot_img(img_list=[]):
    """
    Plot a list of images in a grid.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(img_list)
    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n))  # one row, n columns

    if n == 1:
        axes = [axes]  # make it iterable if only one plot

    for i, img in enumerate(img_list):
        sns.heatmap(img, square=False, cmap='viridis', ax=axes[i], cbar=True)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()