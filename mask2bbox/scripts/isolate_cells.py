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
    description = ("From a mask and an image file obtains the single nuclei crops and saves them to an output folder "
                   "as .png single cell images.")

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
                          action="store", type=str, required=False, default=None,
                          help="Path to the image file.")

    # Add a group of arguments with the tool filters
    filters = parser.add_argument_group(
        title="Options",
        description="Arguments for filtering bounding boxes.")

    filters.add_argument("-e", "--expand", dest="expand",
                         action="store", type=int, required=False, default=0,
                         help="Number of pixels to expand the bounding boxes. [default = 0]")
    filters.add_argument("-fv", "--filter-value", dest="filter_value",
                         action="store", type=float, required=False,
                         default=0.0, nargs="+", help="Filter bounding boxes with a given value value. [default = 0.0]")
    filters.add_argument("-fo", "--filter-operator", dest="filter_operator",
                         action="store", type=str, required=False,
                         choices=["less_equal", "greater_equal", "equal", "not_equal"], default="greater_equal",
                         help="Filter operator. Options =  [less_equal, greater_equal, equal, not_equal] "
                              "[default = greater_equal]")
    filters.add_argument("-ft", "--filter-type", dest="filter_type",
                         action="store", type=str, required=False,
                         choices=["area", "ratio", "center", "sides"], default="area",
                         help="Filter type. Options =  [area, ratio, center, sides][default = area]")
    filters.add_argument("-fe", "--filter-edge", dest="filter_edge",
                         action="store_true", required=False,
                         default=False, help="Filter bounding boxes on the edge of the image. [default = False]")

    # Add a group of arguments for re-sizing
    single_cell = parser.add_argument_group(
        title="Single Cell",
        description="Arguments that modify the behaviour for the single cell crops.")

    single_cell.add_argument("-s", "--size", dest="size",
                             action="store", type=int, required=False, default=256,
                             help="Final image size for the single cell crops. [default = 256]")
    single_cell.add_argument("-rf", "--resize-factor", dest="resize_factor",
                             action="store", type=float, required=False,
                             default=1.0, help="Resize factor for the single cell crops. [default = 1.0]")

    # Add a group of arguments for output
    output = parser.add_argument_group(
        title="Output",
        description="Output arguments for the script.")

    output.add_argument("-o", "--output", dest="output",
                        action="store", type=str, required=False, default=None,
                        help="Path to the output file with the bounding boxes. [default = None]")
    output.add_argument("-sm", "--sample", dest="sample",
                        action="store", type=int, required=False, default=0,
                        help="Working the same to `-osc` but randomly select n=5 cells to explore parameters. "
                             "[default = 0]")

    tool = parser.add_argument_group(
        title="Tool",
        description="Tool arguments for the script.")

    tool.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    tool.add_argument("-log", "--log-level", dest="log_level",
                      action="store", default="info", type=str, required=False,
                      choices=["info", "debug"], help="Log level for the logger.")

    # Parse arguments
    arg = parser.parse_args()

    # Choose filter type
    if arg.filter_operator == "less_equal":
        arg.filter_operator = np.less_equal
    elif arg.filter_operator == "greater_equal":
        arg.filter_operator = np.greater_equal
    elif arg.filter_operator == "equal":
        arg.filter_operator = np.equal
    elif arg.filter_operator == "not_equal":
        arg.filter_operator = np.not_equal
    else:
        raise ValueError(f"Filter type {arg.filter_operator} not recognized.")

    # If filter type is sides then the filter value must be a tuple
    # if arg.filter_type == "sides":
    #     arg.filter_value = (arg.filter_value, arg.filter_value)

    # Convert size to tuple
    arg.size = (arg.size, arg.size)

    # Standardize paths
    arg.mask = Path(arg.mask).resolve()
    arg.image = Path(arg.image).resolve() if arg.image is not None else None
    arg.output = Path(arg.output).resolve() if arg.output is not None else None

    # Return arguments
    return arg


def main():
    # Get arguments
    args = get_arguments()

    # Set logger
    lg = set_logger(log_level=args.log_level)
    lg.info(f"Reading bounding boxes from               = {args.mask}")
    lg.info(f"Reading image data from                   = {args.image}")
    lg.info(f"Resizing factor                           = {args.output}")
    lg.info(f"Expanding bounding boxes by               = {args.expand}")
    lg.info(f"Filtering bounding boxes on edge          = {args.filter_edge}")
    lg.info(
        f"Filtering bounding boxes by               = {args.filter_type} {args.filter_operator} {args.filter_value}")

    # Create a BBoxes object and get the bounding boxes
    lg.debug("Creating BBoxes object")
    mask_boxes = BBoxes.from_mask(args.mask, args.image)
    lg.info(f"Initial number of bounding boxes detected = {len(mask_boxes)}")

    # Get the resizing factor from the original bounding boxes
    lg.debug("Calculating resizing factor")
    rf = mask_boxes.calculate_resizing_factor(args.resize_factor, args.size)

    # Expand the bounding boxes by a given number of pixels (if any)
    lg.debug("Expanding bounding boxes")
    mask_boxes = mask_boxes.expand(args.expand)

    # Filter edge
    if args.filter_edge:
        lg.debug("Filtering bounding boxes on the edge")
        mask_boxes = mask_boxes.filter_edge()

    # Filter bounding boxes by a given type, operator and value
    lg.debug("Filtering bounding boxes")
    mask_boxes = mask_boxes.filter(args.filter_type, args.filter_operator, args.filter_value)
    lg.info(f"Bounding boxes after filtering            = {len(mask_boxes)}")

    # Save single cell bounding boxes
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        lg.debug(f"Saving single cell bounding boxes        = {args.output}")

        # Sample images if needed
        if args.sample > 0:
            lg.debug(f"Saving sample bounding boxes             = {args.sample}")
            mask_boxes = mask_boxes.sample(args.sample)

        # Extract and save the single cell crops
        mask_boxes.extract(resize_factors=rf[mask_boxes.idx()],
                           size=args.size,
                           rescale_intensity=True,
                           output=args.output / args.mask.name.split(".")[0])


# Run main
if __name__ == "__main__":
    main()
