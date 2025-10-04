Pollution Control by Identifying Potential Land for Afforestation

Last updated: 23 Jul, 2025

A Python project that analyzes satellite imagery to identify potential land parcels suitable for afforestation and estimates the number of trees that can be planted. The pipeline uses image segmentation (Mean Shift and simple heuristics) and basic geospatial calculations to convert segmented pixels into area and tree-count suggestions.

Key features

Fetch satellite images (Google Maps Static API) for a target lat/lon.

Segment the image into land-object classes (buildings, roads, water, vegetation, barren land) using Mean Shift image segmentation and post-processing.

Identify and export suitable planting areas as masks / polygons (GeoJSON).

Estimate number of trees based on area and configurable average canopy size.

Export visual results (segmented image, suitability mask, heatmap) and a CSV summary.

Tech stack

Python 3.8+

NumPy, Pillow (PIL), OpenCV (cv2)

scikit-learn (optional for clustering)

requests (for fetching images)

matplotlib (visualization)

(optional) geopandas / shapely for GeoJSON export

Quick start
1. Prerequisites

Python 3.8 or higher

Google Maps API key with Static Maps access (billing may apply)

pip for installing packages

2. Install

Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows
pip install --upgrade pip
pip install numpy pillow opencv-python scikit-learn matplotlib requests
# optional: pip install geopandas shapely


You can put the above packages in requirements.txt:

numpy
pillow
opencv-python
scikit-learn
matplotlib
requests
geopandas  # optional
shapely    # optional

3. Configure API key

Export your Google Maps API key (replace with your method if on Windows):

export GOOGLE_MAPS_API_KEY="YOUR_API_KEY"

Example scripts & usage
Fetch a satellite image

fetch_satellite.py (example snippet)

# fetch_satellite.py
import os, requests

def fetch_satellite(lat, lon, zoom=18, size='640x640', filename='sat.png'):
    key = os.getenv('GOOGLE_MAPS_API_KEY')
    url = (f"https://maps.googleapis.com/maps/api/staticmap?"
           f"center={lat},{lon}&zoom={zoom}&size={size}&maptype=satellite&key={key}")
    r = requests.get(url)
    r.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(r.content)
    return filename

if __name__ == "__main__":
    fetch_satellite(17.3850,78.4867, zoom=18, size='640x640', filename='sample_sat.png')


Note: The Google Static Maps URL above is the standard pattern; make sure your key has the required API & billing set up.

Segment the map (Mean Shift)

segment_map.py (high-level)

# segment_map.py
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sample_sat.png')
# apply Mean Shift filtering (smooths while preserving edges)
sp = 21  # spatial window radius
sr = 51  # color window radius
shifted = cv2.pyrMeanShiftFiltering(img, sp, sr)

# convert to grayscale and threshold / cluster
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

# morphological ops to clean regions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

# visualize
plt.imshow(cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB))
plt.title('Mean Shift result')
plt.show()

# Save masks / results
cv2.imwrite('shifted.png', shifted)
cv2.imwrite('suitable_mask.png', clean)


Tip: For more robust semantic segmentation (trees vs built vs water) use pretrained deep models (DeepLab, U-Net) with transfer learning.

Estimate number of trees from area

analyze_afforestation.py (core idea)

import numpy as np

def estimate_trees(suitable_mask, pixel_area_m2, avg_canopy_m2=9.0):
    # suitable_mask: binary numpy array (1 for suitable pixels)
    num_pixels = np.sum(suitable_mask > 0)
    total_area_m2 = num_pixels * pixel_area_m2
    estimated_trees = int(total_area_m2 / avg_canopy_m2)
    return {'area_m2': total_area_m2, 'estimated_trees': estimated_trees}


How to get pixel_area_m2: depends on map zoom, image size and latitude. For a rough estimate, you can compute ground resolution (meters/pixel) using map metadata / known scale at given zoom or use a ground truth measurement (e.g., measure a known object in the image).

Output

shifted.png — mean-shift filtered image

suitable_mask.png — binary mask of candidate planting pixels

suitable_polygons.geojson — candidate polygons (if geopandas/shapely used)

results.csv — summary with area (m²) and estimated trees

Limitations & caveats

RGB satellite images limit ability to perfectly separate vegetation types (no NDVI without multispectral data).

Accuracy depends on image resolution and zoom; fine-grained identification requires high-resolution images.

Pixel-to-area conversion must be computed carefully for accurate tree counts.

Regulatory & ethical: Always verify land ownership, protected status, and obtain permissions before afforestation. This tool only suggests candidate areas — do not act without local approval.

Possible improvements / next steps

Use multispectral data (Sentinel-2/Planet Labs) and NDVI for better vegetation detection.

Train a semantic segmentation model (U-Net / DeepLab) on labeled satellite datasets to detect built-up vs bare vs greenery accurately.

Integrate with Google Earth Engine for large-area processing.

Add exclusion layers (roads, buildings, protected zones) using OpenStreetMap / local cadastral data.

Create an interactive map UI (Folium / Leaflet) to let users review and adjust candidate polygons.

License & author

Author: Cheguri Venkatesham — https://github.com/Venkatesh290

License: MIT (or choose your preferred license)

