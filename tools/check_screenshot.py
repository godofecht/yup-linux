#!/usr/bin/env python3

# ==============================================================================
#
#  This file is part of the YUP library.
#  Copyright (c) 2026 - kunitoki@gmail.com
#
#  YUP is an open source library subject to open-source licensing.
#
#  The code included in this file is provided under the terms of the ISC license
#  http://www.isc.org/downloads/software-support-policy/isc-license. Permission
#  to use, copy, modify, and/or distribute this software for any purpose with or
#  without fee is hereby granted provided that the above copyright notice and
#  this permission notice appear in all copies.
#
#  YUP IS PROVIDED "AS IS" WITHOUT ANY WARRANTY, AND ALL WARRANTIES, WHETHER
#  EXPRESSED OR IMPLIED, INCLUDING MERCHANTABILITY AND FITNESS FOR PURPOSE, ARE
#  DISCLAIMED.
#
# ==============================================================================

"""
Sanity-check a captured screenshot before it is committed.

A GPU capture can fail in two ways that still produce a valid PNG file:

  Noise    The presented surface was never written, so the capture holds
           uninitialised GPU memory. It decodes fine and looks like static.

  Blank    Nothing rendered at all, so the capture is a single flat colour.

Both are caught by measuring how much of the image is locally flat: the
fraction of pixels identical to all four of their neighbours. Real UI is
mostly flat fills and so scores high; uninitialised memory scores near zero.

Measured on the YUP graphics example at 1024x768:

    uninitialised memory     7.9% flat,  50275 unique colours
    correct render          80.8% flat,    633 unique colours

Exit code is 0 when the image passes and 1 when it does not.

Usage:
    check_screenshot.py IMAGE [--min-flat PERCENT] [--min-colors N]
"""

import argparse
import sys

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:
    sys.stderr.write(f"check_screenshot: missing dependency: {exc}\n")
    sys.stderr.write("check_screenshot: install with 'pip install pillow numpy'\n")
    sys.exit(2)


DEFAULT_MIN_FLAT = 25.0
DEFAULT_MIN_COLORS = 5


def measure(path):
    """Returns (width, height, flat_percent, unique_colors) for an image."""
    with Image.open(path) as handle:
        pixels = np.asarray(handle.convert("RGB"))

    height, width, _ = pixels.shape
    if width < 3 or height < 3:
        raise ValueError(f"image is too small to measure: {width}x{height}")

    packed = (
        (pixels[:, :, 0].astype(np.uint32) << 16)
        | (pixels[:, :, 1].astype(np.uint32) << 8)
        | pixels[:, :, 2].astype(np.uint32)
    )

    core = packed[1:-1, 1:-1]
    flat = (
        (core == packed[1:-1, 0:-2])
        & (core == packed[1:-1, 2:])
        & (core == packed[0:-2, 1:-1])
        & (core == packed[2:, 1:-1])
    )

    return width, height, float(flat.mean()) * 100.0, int(np.unique(packed).size)


def main():
    parser = argparse.ArgumentParser(
        description="Reject a screenshot that is uninitialised memory or blank."
    )
    parser.add_argument("image", help="path to the PNG to check")
    parser.add_argument(
        "--min-flat",
        type=float,
        default=DEFAULT_MIN_FLAT,
        metavar="PERCENT",
        help=f"minimum percentage of locally flat pixels (default: {DEFAULT_MIN_FLAT})",
    )
    parser.add_argument(
        "--min-colors",
        type=int,
        default=DEFAULT_MIN_COLORS,
        metavar="N",
        help=f"minimum number of distinct colours (default: {DEFAULT_MIN_COLORS})",
    )
    args = parser.parse_args()

    try:
        width, height, flat, colors = measure(args.image)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"check_screenshot: cannot read {args.image}: {exc}\n")
        return 1

    print(f"image      {args.image}")
    print(f"size       {width}x{height}")
    print(f"flat       {flat:.2f}%  (minimum {args.min_flat:.2f}%)")
    print(f"colours    {colors}  (minimum {args.min_colors})")

    if flat < args.min_flat:
        print()
        print("FAIL: the capture is mostly noise.")
        print("Only a small part of it is locally flat, which is what uninitialised")
        print("GPU memory looks like. The frame was probably never presented, or the")
        print("copy to the window was clipped.")
        return 1

    if colors < args.min_colors:
        print()
        print("FAIL: the capture is blank.")
        print(f"The whole image is {colors} colour(s), so nothing was drawn into it.")
        return 1

    print()
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
