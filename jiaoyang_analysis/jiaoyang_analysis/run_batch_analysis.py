from jiaoyang_analysis.analysis import analyze_image
from jiaoyang_analysis.config import data_dir, save_dir
from jiaoyang_analysis.import_data import batch_import_data
import pandas as pd

roi_prefix = 'test_'  # Adjust this prefix based on your ROI naming convention
data_package = batch_import_data(data_dir, roi_prefix)

results = []

for img_package in data_package:
    filename, img, roi = img_package
    result = analyze_image(filename, img, roi)
    results.append(result)

print(f"Total images processed: {len(results)}\nSaving results to {save_dir / 'analysis_results.csv'}")
df = pd.DataFrame(pd.concat(results, ignore_index=True))
df.to_csv(save_dir / 'analysis_results.csv', index=False)
