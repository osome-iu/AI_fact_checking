"""
Convenience bonferonni correction function.
"""


def bonferroni_correction(p_value, n_comparisons):
    """
    Perform Bonferroni correction on a given p-value for multiple comparisons.
    Parameters:
    -----------
    - p_value (float): The original p-value.
    - n_comparisons (int): The number of comparisons being made.
    Returns:
    -----------
    - float: The Bonferroni corrected p-value.
    """
    bonferroni_corrected_p_value = p_value * n_comparisons
    # Ensure the corrected p-value is not greater than 1
    bonferroni_corrected_p_value = min(1.0, bonferroni_corrected_p_value)
    return bonferroni_corrected_p_value
