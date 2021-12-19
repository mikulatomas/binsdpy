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


def sokal_sneath1(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / (a + 2 * b + 2 * c)


def sokal_michener(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + d) / (a + b + c + d)


def sokal_sneath2(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
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

    return math.log(a) - math.log(n) - math.log((a + b) / n) - math.log((a + c) / n)


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


def gower(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + d) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))


def pearson1(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    return (n * math.pow(a * d - b * c, 2)) / ((a + b) * (a + c) * (b + d) * (c + d))


def pearson2(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    x_2 = pearson1(x, y)

    return math.sqrt(x_2 / (n + x_2))


def pearson_heron1(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a * d - b * c) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))


def pearson3(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    p = pearson_heron1(x, y)

    return math.sqrt(p / (n + p))


def pearson_heron2(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return math.cos(
        (math.pi * math.sqrt(b * c)) / (math.sqrt(a * d) + math.sqrt(b * c))
    )


def sokal_sneath3(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + d) / (b + c)


def sokal_sneath5(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a * d) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))


def cole(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (math.sqrt(2) * (a * d - b * c)) / math.sqrt(
        math.pow(a * d - b * c, 2) - (a + b) * (a + c) * (b + d) * (c + d)
    )


def stiles(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    return math.log10(
        (n * math.pow(abs(a * d - b * c) - (n / 2), 2))
        / (a + b)
        * (a + c)
        * (b + d)
        * (c + d)
    )


def yuleq(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a * d - b * c) / (a * d + b * c)


def yulew(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (math.sqrt(a * d) - math.sqrt(b * c)) / (math.sqrt(a * d) + math.sqrt(b * c))


def kulczynski1(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / (b + c)


def tanimoto(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / ((a + b) + (a + c) - a)


def disperson(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a * d - b * c) / math.pow(a + b + c + d, 2)


def hamann(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return ((a + d) - (b + c)) / (a + b + c + d)


def michael(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return 4 * (a * d - b * c) / (math.pow(a + d, 2) + math.pow(b + c, 2))


def goodman_kruskal(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    sigma = max(a, b) + max(c, d) + max(a, c) + max(b, d)
    sigma_ = max(a + c, b + d) + max(a + b, c + d)

    return (sigma - sigma_) / (2 * n - sigma_)


def anderberg(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    sigma = max(a, b) + max(c, d) + max(a, c) + max(b, d)
    sigma_ = max(a + c, b + d) + max(a + b, c + d)

    return (sigma - sigma_) / (2 * n)


def baroni_urbani_buser1(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (math.sqrt(a * d) + a) / (math.sqrt(a * d) + a + b + c)


def baroni_urbani_buser2(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (math.sqrt(a * d) + a - (b + c)) / (math.sqrt(a * d) + a + b + c)


def peirce(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a * b + b * c) / (a * b + 2 * b * c + c * d)


def eyraud(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    n = a + b + c + d

    return (math.pow(n, 2) + (n * a - (a + b) * (a + c))) / ((a + b) * (a + c) * (b + d) * (c + d))


def tarantula(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a * (c + d)) / (c * (a + b))


def ample(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    a, b, c, d = operational_taxonomic_units(x, y)

    return abs((a * (c + d)) / (c * (a + b)))



    