def score_sequence(seq):
    hydrophobic = "AILMFWYV"
    return sum(1 for c in seq if c in hydrophobic) / len(seq)