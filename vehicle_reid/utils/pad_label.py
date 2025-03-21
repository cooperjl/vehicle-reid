def pad_label(label: int, dataset: str) -> str:
    match dataset:
        case "veri":
            # less than 1000 classes, pad to 3 digits
            return f"{label:03}"
        case "vric":
            # pad to 4 digits
            return f"{label:04}"
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

