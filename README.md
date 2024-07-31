# mask2bbox
[![PyPI](https://img.shields.io/pypi/v/mask2bbox?style=flat-square)](https://pypi.org/project/mask2bbox/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mask2bbox?style=flat-square)](https://pypi.org/project/mask2bbox/)
[![PyPI - License](https://img.shields.io/pypi/l/mask2bbox?style=flat-square)](https://pypi.org/project/mask2bbox/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/mask2bbox?style=flat-square)](https://pypi.org/project/mask2bbox/)
[![main](https://github.com/saezlab/liana-py/actions/workflows/main.yml/badge.svg)](https://github.com/schapirolabor/mask2bbox/actions)

For a given mask, gets the coordinates of bounding box of each element of the mask. It will also allow for more operations in the future.

## Installation

```bash
pip install mask2bbox
```

## CLI

```bash
isolate-cells -h
isolate-cells-from-file -h
get-average-iou -h
```

## Usage

```python
import numpy as np
from mask2bbox import BBoxes

# Create a BBoxes object
all_boxes = BBoxes.from_mask("path/to/mask.png")

# Expand the bounding boxes
all_boxes = all_boxes.expand(n=10)

# Remove the bounding boxes that are located on the edge of the image
all_boxes = all_boxes.remove_from_edge()

# Get the sides of all the bounding boxes
sides = all_boxes.get("sides")

# Filter the bounding boxes by the sides
filtered_boxes = all_boxes.filter("sides", np.greater_equal, (35, 35))

# Get the IoU matrix of all the bounding boxes
iou = filtered_boxes.iou_matrix()

# Save the overlapping pairs to
filtered_boxes.save_overlapping_pairs("path/to/save/overlapping_pairs.csv")

# Save the IOU matrix to a csv file
filtered_boxes.save_iou_matrix("path/to/save/iou_matrix.csv")   

# Plot the bounding boxes on the mask image
filtered_boxes.draw(to="image", method="matplotlib", show="False", save="path/to/save/image.png")

# Save your bounding boxes
filtered_boxes.save_csv("path/to/bounding_boxes.csv")

# Get resize factors to resize the bounding boxes to a given size
resize_factors = filtered_boxes.de(desired_ratio=0.7, size=(256, 256))

# Extract the bounding boxes as images
filtered_boxes.extract(resize_factors, size=(256, 256), output="path/to/save/images")
```

## License

mask2bbox offers a dual licensing mode the [GNU Affero General Public License v3.0](LICENSE) - see [LICENSE](LICENSE) and [ESSENTIAL_LICENSE_CONDITIONS.txt](ESSENTIAL_LICENSE_CONDITIONS.txt)
