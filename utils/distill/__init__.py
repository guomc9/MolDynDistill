from .mtt import MTT

def get_distill_algorithm(algorithm: str):
    algorithm = algorithm.lower()
    if algorithm == 'mtt':
        return MTT()