import random

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def generate_sequence(length=50):
    return "".join(random.choice(AMINO_ACIDS) for _ in range(length))

def mutate_sequence(seq):
    i = random.randint(0, len(seq)-1)
    return seq[:i] + random.choice(AMINO_ACIDS) + seq[i+1:]