"""
Convenience functions to calculate metrics are saved here.
"""


def df_precision(df, label):
    """
    Calculate the precision of predicting `label` within a df.

    precision = TP_label / (TP_label + FP_label)

    Precision measures the model's ability to identify instances of a particular class correctly.
    I.e., when making a prediction of label, how often is it correct.

    Parameters
    ----------
    - df (pandas df): the dataframe to operate on
    - label (str): the label
    """
    true_positives = sum((df["veracity"] == label) & (df["judgment"] == label))
    false_positives = sum((df["veracity"] != label) & (df["judgment"] == label))

    denominator = true_positives + false_positives
    return true_positives / denominator if denominator != 0 else 0


def df_recall(df, label):
    """
    Calculate the recall of `label` within a df.

    precision = TP_label / (TP_label + FN_label)

    Recall is the fraction of instances in a class that the model correctly classified
    out of all instances in that class. I.e., the proportion of instances that were predicted correctly.

    Parameters
    ----------
    - df (pandas df): the dataframe to operate on
    - label (str): the label
    """

    true_positives = sum((df["veracity"] == label) & (df["judgment"] == label))
    false_negatives = sum((df["veracity"] == label) & (df["judgment"] != label))

    denominator = true_positives + false_negatives
    return true_positives / denominator if denominator != 0 else 0
