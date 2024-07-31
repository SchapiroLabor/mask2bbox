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
    description = "Isolates single cells from a mask file using a list of CellIDs/MaskIDs."

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
    required.add_argument("-f", "--file", dest="file",
                          action="store", type=str, required=True,
                          help="Path to the file with the list of CellIDs to isolate.")
    required.add_argument("-c", "--cell-id", dest="cell_id",
                          action="store", type=int, required=False, default="CellID",
                          help="Column name in the -f/--file with the CellIDs/MaskIDs. Default = 'CellID'.")

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
    arg.file = Path(arg.file).resolve()
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

    # Read label file and get the cell IDs
    lg.debug("Reading the label file")
    cell_ids = np.loadtxt(args.file, delimiter=",", skiprows=1, usecols=(args.cell_id,), dtype=int)
    lg.info(f"Number of cell IDs to isolate             = {len(cell_ids)}")

    # Isolate the cells
    lg.debug("Isolating cells")

    # Create BBoxes object from mask
    lg.debug("Creating BBoxes object")
    boxes = BBoxes.from_mask(mask=args.mask, image=args.image)
    lg.info(f"Number of bounding boxes                  = {len(boxes)}")

    # Get the resizing factor from the original bounding boxes
    lg.debug("Calculating resizing factor")
    rf = boxes.calculate_resizing_factor(args.resize_factor, args.size)

    # Expand the bounding boxes by a given number of pixels (if any)
    lg.debug("Expanding bounding boxes")
    mask_boxes = boxes.expand(args.expand)

    # Save single cell bounding boxes
    lg.debug(f"Saving single cell bounding boxes        = {args.output}")

    # Sub-setting the bounding boxes for the cell IDs
    filtered_boxes = boxes.subset(cell_ids)
    lg.debug(f"Subset bounding boxes                    = {len(filtered_boxes)}")

    # Extract and save the single cell crops
    filtered_boxes.extract(resize_factors=rf[filtered_boxes.idx()],
                            size=args.size,
                            rescale_intensity=True,
                            output=args.output)


# Main function
if __name__ == "__main__":
    main()