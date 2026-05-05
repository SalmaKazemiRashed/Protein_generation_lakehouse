import random

def add_noise(seq):
    return "".join(c if random.random() > 0.1 else "X" for c in seq)

def denoise(seq):
    return seq.replace("X", "A")