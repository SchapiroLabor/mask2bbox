# Import libraries
import argparse
from pathlib import Path

# Import third-party libraries
import numpy as np

# Import local libraries
from mask2bbox._bboxes import BBoxes
from mask2bbox.logger import set_logger
from mask2bbox.version import __version__

# Get arguments
def get_arguments():
    # Start with the description
    description = "Calculates the Intersection over the Union (IoU) for all the bounding boxes of a mask file."

    # Add parser
    parser = argparse.ArgumentParser(description=description)

    # Add a group of arguments for input
    required = parser.add_argument_group(
        title="Input",
        description="Input arguments for the script.")

    required.add_argument("-m", "--mask", dest="mask",
                          action="store", type=str, required=True,
                          help="Path to the mask file.")
    required.add_argument("-i", "--image", dest="image",
                          action="store", type=str, required=True,
                          help="Path to the image file.")

    # Add a group of arguments for the single cell crops
    single_cell = parser.add_argument_group(
        title="Single Cell",
        description="Arguments that modify the behaviour of the single cell crops.")
    single_cell.add_argument("-e", "--expand", dest="expand",
                             action="store", type=int, required=False, default=0,
                             help="Number of pixels to expand the bounding boxes.")
    single_cell.add_argument("-s", "--size", dest="size",
                             action="store", type=int, required=False, default=(256, 256),
                             help="Final image size for the single cell crops.")
    single_cell.add_argument("-rf", "--resize-factor", dest="resize_factor",
                             action="store", type=float, required=False,
                             default=1.0, help="Resize factor for the single cell crops.")

    # Add a group of arguments for output
    output = parser.add_argument_group(
        title="Output",
        description="Output arguments for the script.")

    output.add_argument("-o", "--output", dest="output",
                        action="store", type=str, required=True,
                        help="Folder path to which the single cell crops will be saved.")

    # Add a group of arguments for logging
    logging = parser.add_argument_group(
        title="Logging",
        description="Arguments for logging.")

    logging.add_argument("-l", "--log", dest="log",
                         action="store", type=str, required=False, default="info",
                         choices=["info", "debug"],
                         help="Level of logging to use default = 'info")

    # Add a group of arguments for version
    version = parser.add_argument_group(
        title="Version",
        description="Arguments for version.")

    version.add_argument("-v", "--version", dest="version",
                        action="version", version=f"%(prog)s {__version__}")

    # Parse the arguments
    arg = parser.parse_args()

    # Standardize paths
    arg.mask = Path(arg.mask).resolve()
    arg.image = Path(arg.image).resolve()
    arg.output = Path(arg.output).resolve()

    # Return the parser
    return arg


# Main function
def main():
    # Get arguments
    args = get_arguments()

    # Set logger
    lg = set_logger(log_level=args.log)
    lg.info(f"Reading bounding boxes from               = {args.mask}")
    lg.info(f"Reading image data from                   = {args.image}")
    lg.info(f"Resizing factor                           = {args.output}")
    lg.info(f"Expanding bounding boxes by               = {args.expand}")

    # Create BBoxes object from mask
    lg.debug("Creating BBoxes object")
    boxes = BBoxes.from_mask(mask=args.mask, image=args.image)
    lg.info(f"Number of bounding boxes                  = {len(boxes)}")

    # Expand the bounding boxes
    lg.debug("Expanding bounding boxes")
    boxes = boxes.expand(args.expand)

    # Get the resizing factor from the original bounding boxes
    lg.debug("Calculating iou")


    # Get the overlapping pair
    pairs, values = boxes.get_overlapping_pairs()
    iou_matrix = boxes.iou_matrix

    results = {
        "number_of_cells": len(boxes),
        "number_of_pairs": len(np.triu_indices(len(np.diag(iou_matrix)),1)[0]),
        "number_of_overlapping_pairs": len(values),
        "number_of_overlapping_cells": len(np.unique(pairs)),
        "max_iou": np.max(iou_matrix),

    }

    # Get me the indexes of values greater than a threshold value
    threshold = 0.3
    lg.info(f"Number of cells with an overlap bigger than {threshold} = {len(np.unique(pairs[np.where(values > threshold)]))}")

    threshold = 0.5
    lg.info(f"Number of cells with an overlap bigger than {threshold} = {len(np.unique(pairs[np.where(values > threshold)]))}")

    # Get the average iou over the upper triangular matrix
    lg.info(f"Average iou across all pairs = {np.mean(iou_matrix[np.triu_indices(len(np.diag(iou_matrix)),1)]):.8f}")

    # GEt the average iou over the upper triangular matrix for iou values greater than 0
    lg.info(f"Average iou across all pairs greater than 0 = {np.mean(iou_matrix[iou_matrix > 0]):.8f}")

    # Get the median iou over the upper triangular matrix
    lg.info(f"Median iou across all pairs = {np.median(iou_matrix[np.triu_indices(len(np.diag(iou_matrix)),1)]):.8f}")

    # Get the median iou over the upper triangular matrix for iou values greater than 0
    lg.info(f"Median iou across all pairs greater than 0 = {np.median(iou_matrix[iou_matrix > 0]):.8f}")


# Main function
if __name__ == "__main__":
    main()