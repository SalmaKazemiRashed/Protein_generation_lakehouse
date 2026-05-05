def interpolate(seq1, seq2, alpha=0.5):
    return "".join(
        s1 if i < alpha * len(seq1) else s2
        for i, (s1, s2) in enumerate(zip(seq1, seq2))
    )