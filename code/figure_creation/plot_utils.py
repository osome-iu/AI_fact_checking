"""
Plotting convenience functions.
"""
import matplotlib.colors as mcolors
import numpy as np


def darken_hexcode(hexcode, factor=0.05):
    """
    Make a darker version of a hexcode color.

    Parameters:
    -----------
    hexcode : str
        A string representation of a hexcode color.
    factor : float, optional (default=0.05)
        A factor indicating how much darker the color should be.
        It is a float value ranging from 0 to 1.

    Returns:
    --------
    str
        A new hexcode color that is darker than the original one.

    Example:
    --------
    >>> darken_hexcode('#FF0000', 0.2)
    '#CC0000'
    """
    rgb = np.array(mcolors.hex2color(hexcode))

    # Darken by factor
    darker_rgb = tuple(rgb - (rgb) * factor)

    # Convert color back to hexcode
    new_hexcode = mcolors.to_hex(darker_rgb)
    return new_hexcode
