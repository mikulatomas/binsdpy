import math

from .utils import operational_taxonomic_units, BinaryFeatureVector


def hamming(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    _, b, c, _ = operational_taxonomic_units(x, y)

    return b + c


def euclid(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    _, b, c, _ = operational_taxonomic_units(x, y)

    return math.sqrt(b + c)


def squared_euclid(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    _, b, c, _ = operational_taxonomic_units(x, y)

    return math.sqrt(math.pow(b + c, 2))


def canberra(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    _, b, c, _ = operational_taxonomic_units(x, y)

    return b + c


def manhattan(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    _, b, c, _ = operational_taxonomic_units(x, y)

    return b + c


def mean_manhattan(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (b + c) / (a + b + c + d)


def cityblock(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    _, b, c, _ = operational_taxonomic_units(x, y)

    return b + c


def minkowski(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    _, b, c, _ = operational_taxonomic_units(x, y)

    return b + c


def vari(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (b + c) / (4 * (a + b + c + d))


def size_difference(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return math.pow(b + c, 2) / math.pow(a + b + c + d, 2)


def shape_difference(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    return (n * (b + c) - math.pow(b + c, 2)) / math.pow(n, 2)


def pattern_difference(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (4 * b * c) / math.pow(a + b + c + d, 2)


def lance_williams(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (b + c) / (2 * a + b + c)


def bray_curtis(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (b + c) / (2 * a + b + c)


def hellinger(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return 2 * math.sqrt((1 - (a / (math.sqrt((a + b) * (a + c))))))


def chord(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return math.sqrt(2 * (1 - (a / (math.sqrt((a + b) * (a + c))))))



