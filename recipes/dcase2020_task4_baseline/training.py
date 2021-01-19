def get_batchsizes_and_masks(no_synthetic, batch_size):
    """
        Getting the batch szes and labels mask depending on the use of synthetic data or not.

    Args:
        no_synthetic: bool, True if synthetic data are not used, False if synthetic data are used
        batch_size: int, batch size

    Return:
        weak_mask: slice function used to get only information regarding weak label data
        strong_mask: slice function used to get only information regarding strong label data
        batch_sizes: list of batch sizes
    """

    if not no_synthetic:
        batch_sizes = [
            batch_size // 4,
            batch_size // 2,
            batch_size // 4,
        ]
        strong_mask = slice((3 * batch_size) // 4, batch_size)
    else:
        batch_sizes = [batch_size // 4, 3 * batch_size // 4]
        strong_mask = None

    # assume weak data is always the first one
    weak_mask = slice(batch_sizes[0])

    return weak_mask, strong_mask, batch_sizes
