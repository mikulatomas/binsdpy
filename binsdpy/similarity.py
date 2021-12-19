import math

from .utils import operational_taxonomic_units, BinaryFeatureVector


def jaccard(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / (a + b + c)


def dice(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (2 * a) / (2 * a + b + c)


def czekanowski(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (2 * a) / (2 * a + b + c)


def jaccard_3w(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (3 * a) / (3 * a + b + c)


def nei_li(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (2 * a) / (2 * a + b + c)


def sokal_sneath_1(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / (a + 2 * b + 2 * c)


def sokal_michener(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + d) / (a + b + c + d)


def sokal_sneath_2(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (2 * (a + d)) / (2 * a + b + c + 2 * d)


def roger_tanimoto(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + d) / (a + 2 * (b + c) + d)


def faith(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + 0.5 * d) / (a + b + c + d)


def gower_legendre(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + d) / (a + 0.5 * (b + c) + d)


def itersection(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, _, _, _ = operational_taxonomic_units(x, y)

    return a


def inner_product(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, _, _, d = operational_taxonomic_units(x, y)

    return a + d


def russell_rao(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return a / (a + b + c + d)


def cosine(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return a / math.sqrt((a + b) * (a + c))


def gilbert_wells(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    return math.log10(a) - math.log10(n) - math.log10((a + b) / n) - math.log10((a + c) / n)


def ochiai1(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / math.sqrt((a + b) * (a + c))


def forbesi(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    return (n * a) / ((a + b) * (a + c))


def fossum(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    return (n * math.pow(a - 0.5, 2)) / ((a + b) * (a + c))


def sorgenfrei(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return math.pow(a, 2) / ((a + b) * (a + c))


def mountford(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / (0.5 * (a * b + a * c) + b * c)


def otsuka(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / math.pow((a + b) * (a + c), 0.5)


def mcconnaughey(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (math.pow(a, 2) - b * c) / ((a + b) * (a + c))


def tarwid(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    return (n * a - (a + b) * (a + c)) / (n * a + (a + b) * (a + c))


def kulczynski2(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return ((a / 2) * (2 * a + b + c)) / ((a + b) * (a + c))


def driver_kroeber(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (a / 2) * ((1 / (a + b) + (1 / (a + c))))


def johnson(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / (a + b) + a / (a + c)


def dennis(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    return (a * d - b * c) / math.sqrt(n * (a + b) * (a + c))


def simpson(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / min(a + b, a + c)


def braun_banquet(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / max(a + b, a + c)


def fager_mcgowan(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / math.sqrt((a + b) * (a + c)) - max(a + b, a + c) / 2


def forbes2(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    return (n * a - (a + b) * (a + c)) / (n * min(a + b, a + c) - (a + b) * (a + c))


def sokal_sneath4(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a / (a + b) + a / (a + c) + d / (b + d) + d / (b + d)) / 4
