"""
Effect size function saved here.
"""
from numpy import std, mean, sqrt


def cohen_d(x, y):
    """
    Calculate cohen's d based on sampled x and y data.
    See: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(
        ((nx - 1) * std(x, ddof=1) ** 2 + (ny - 1) * std(y, ddof=1) ** 2) / dof
    )
