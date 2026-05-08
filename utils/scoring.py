def score_sequence(seq):

    if not seq:
        return 0.0

    hydrophobic = set("AILMFWYV")

    hydrophobic_count = sum(
        1 for c in seq if c in hydrophobic
    )

    return round(hydrophobic_count / len(seq), 3)