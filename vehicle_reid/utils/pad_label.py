def pad_label(label: int, dataset: str) -> str:
    """Convert an integer label of a dataset class to a padded string.

    For example, 1 becomes "001" for VeRi-776.

    Parameters
    ----------
    label : int
        The label to convert.
    dataset : str
        The dataset the label belongs to, for proper padding depending on size.
    """
    match dataset:
        case "veri":
            # less than 1000 classes, pad to 3 digits
            return f"{label:03}"
        case "vric":
            # pad to 4 digits
            return f"{label:04}"
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")
