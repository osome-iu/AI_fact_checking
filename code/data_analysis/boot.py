"""
Bootstrapping functions are saved here.
"""
import numpy as np

np.random.seed(22)


# Define the bootstrap function
def bootstrap_ci(array, confidence=0.95, n_samples=5_000, d_only=True):
    """
    Calculate a confidence interval for sample data via bootstrapping

    Parameters:
    -----------
    - array (of ints/floats): the sample data
    - confidence (float): the desired level of confidence (default=0.95)
    - n_samples (float): number of bootstrap samples
    - distance_only (bool): If True, return only half the distance between the
        upper and lower bounds

    Returns:
    -----------
    - if d_only = False (tuple): The lower and upper bounds of the confidence interval.
    - if d_only = True (float, Default): Distance from the mean based on SEM 95% CI
    """
    # Ignore NaN values
    array = [val for val in array if not np.isnan(val)]
    n = len(array)
    bootstrapped_means = []
    for _ in range(n_samples):
        bootstrap_sample = np.random.choice(array, n)
        bootstrapped_means.append(np.mean(bootstrap_sample))
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(bootstrapped_means, alpha * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 - alpha) * 100)
    if d_only:
        h = (upper_bound - lower_bound) / 2
        return h
    return (lower_bound, upper_bound)


def bootstrap_wci(obs, weights, confidence=0.95, n_samples=5_000, d_only=True):
    """
    Calculate a weighted confidence interval for sample data via bootstrapping.
    This works by following the below procedure:
        1. Sample both observed data point and it's weight
        2. Calculate the weighted mean using both
        3. Calculate the confidence interval from the distribution of means

    Parameters:
    -----------
    - obs (float): the sample data
    - weights (float): the weights for `obs`
    - confidence (float): the desired level of confidence (default=0.95)
    - n_samples (float): number of bootstrap samples
    - distance_only (bool): If True, return only half the distance between the
        upper and lower bounds

    Returns:
    -----------
    - if d_only = False (tuple): The lower and upper bounds of the confidence interval.
    - if d_only = True (float, Default): Distance from the mean based on SEM 95% CI
    """
    paired_data = np.array([obs, weights]).T
    n = len(paired_data)
    bootstrapped_means = []
    for _ in range(n_samples):
        sampled_indices = np.random.choice(range(n), n)
        bootstrap_sample = paired_data[sampled_indices]
        a = bootstrap_sample[:, 0]
        weights = bootstrap_sample[:, 1]
        bootstrapped_means.append(np.average(a=a, weights=weights))
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(bootstrapped_means, alpha * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 - alpha) * 100)
    if d_only:
        h = (upper_bound - lower_bound) / 2
        return h
    return (lower_bound, upper_bound)


def mean_diff_bootstrap_ci(
    array1, array2, confidence=0.95, n_samples=5_000, d_only=True
):
    """
    Calculate a confidence interval for a MannWhitneyU comparison, i.e., bounds
    for the avg. median difference.

    Parameters:
    -----------
    - array1 (of ints/floats): the sample data from group 1
    - array2 (of ints/floats): the sample data from group 2
    - confidence (float): the desired level of confidence (default=0.95)
    - n_samples (float): number of bootstrap samples
    - distance_only (bool): If True, return only half the distance between the
        upper and lower bounds

    Returns:
    -----------
    - if d_only = False (tuple): The lower and upper bounds of the confidence interval.
    - if d_only = True (float, Default): Distance from the mean based on SEM 95% CI
    """
    # Ignore NaN values
    array1 = [val for val in array1 if not np.isnan(val)]
    array2 = [val for val in array2 if not np.isnan(val)]
    n1 = len(array1)
    n2 = len(array2)
    bootstrapped_diffs = []
    for _ in range(n_samples):
        bootstrap_sample1 = np.random.choice(array1, n1)
        bootstrap_sample2 = np.random.choice(array2, n2)
        bootstrapped_diffs.append(
            np.mean(bootstrap_sample1) - np.mean(bootstrap_sample2)
        )

    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(bootstrapped_diffs, alpha * 100)
    upper_bound = np.percentile(bootstrapped_diffs, (1 - alpha) * 100)
    if d_only:
        h = (upper_bound - lower_bound) / 2
        return h
    return (lower_bound, upper_bound)
