import math

from .utils import operational_taxonomic_units, BinaryFeatureVector


def jaccard(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Jaccard similarity
    
    Jaccard, P. (1908).
    Nouvelles recherches sur la distribution florale.
    Bull. Soc. Vaud. Sci. Nat., 44, 223-270.

    Same as:
        Tanimoto coefficient

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / (a + b + c)


def dice(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Sørensen–Dice similarity
    
    Sorensen, T. A. (1948).
    A method of establishing groups of equal amplitude in plant sociology
    based on similarity of species content and its application to analyses
    of the vegetation on Danish commons.
    Biol. Skar., 5, 1-34.

    Dice, L. R. (1945).
    Measures of the amount of ecologic association between species.
    Ecology, 26(3), 297-302
    
    Same as:
        Czkanowski similarity
        Nei-Li similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (2 * a) / (2 * a + b + c)


def czekanowski(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Czkanowski similarity

    Same as:
        Sørensen–Dice coefficient
        Nei-Li similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (2 * a) / (2 * a + b + c)


def jaccard_3w(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """3W Jaccard similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (3 * a) / (3 * a + b + c)


def nei_li(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Nei-Li similarity

    Same as:
        Sørensen–Dice coefficient
        Czkanowski similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y)

    return (2 * a) / (2 * a + b + c)


def sokal_sneath1(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Sokal-Sneath similarity (v1)
    
    Sneath, P. H., & Sokal, R. R. (1973).
    Numerical taxonomy.
    The principles and practice of numerical classification.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / (a + 2 * b + 2 * c)


def smc(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Simple Matching Coefficient (SMC) similarity
    
    Sokal, R. R. (1958).
    A statistical method for evaluating systematic relationships. 
    Univ. Kansas, Sci. Bull., 38, 1409-1438.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + d) / (a + b + c + d)


def sokal_sneath2(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Sokal-Sneath similarity (v2)
    
    Sneath, P. H., & Sokal, R. R. (1973).
    Numerical taxonomy.
    The principles and practice of numerical classification.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y)

    return (2 * (a + d)) / (2 * a + b + c + 2 * d)


def rogers_tanimoto(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Roges-Tanimoto similarity

    Rogers, D. J., & Tanimoto, T. T. (1960).
    A computer program for classifying plants.
    Science, 132(3434), 1115-1118.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + d) / (a + 2 * (b + c) + d)


def faith(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Faith similarity

    Faith, D. P. (1983).
    Asymmetric binary similarity measures.
    Oecologia, 57(3), 287-290.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + 0.5 * d) / (a + b + c + d)


def gower_legendre(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Gower-Legendre similarity

    Gower, J. C., & Legendre, P. (1986).
    Metric and Euclidean properties of dissimilarity coefficients.
    Journal of classification, 3(1), 5-48.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + d) / (a + 0.5 * (b + c) + d)


def itersection(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Intersection similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, _, _, _ = operational_taxonomic_units(x, y)

    return a


def inner_product(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Inner product similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, _, _, d = operational_taxonomic_units(x, y)

    return a + d


def russell_rao(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Russel-Rao similarity

    Rao, C. R. (1948).
    The utilization of multiple measurements in problems of biological classification.
    Journal of the Royal Statistical Society. Series B (Methodological), 10(2), 159-203.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y)

    return a / (a + b + c + d)


def cosine(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Cosine similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y)

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
    """Sokal-Sneath similarity (v4)
    
    Sneath, P. H., & Sokal, R. R. (1973).
    Numerical taxonomy.
    The principles and practice of numerical classification.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a / (a + b) + a / (a + c) + d / (b + d) + d / (b + d)) / 4


def gower(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Gower similarity
    
    Gower, J. C. (1971).
    A general coefficient of similarity and some of its properties.
    Biometrics, 857-871.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
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
    """Sokal-Sneath similarity (v3)
    
    Sneath, P. H., & Sokal, R. R. (1973).
    Numerical taxonomy.
    The principles and practice of numerical classification.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y)

    return (a + d) / (b + c)


def sokal_sneath5(x: BinaryFeatureVector, y: BinaryFeatureVector) -> float:
    """Sokal-Sneath similarity (v5)
    
    Sneath, P. H., & Sokal, R. R. (1973).
    Numerical taxonomy.
    The principles and practice of numerical classification.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
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
    """Tanimoto coefficient
    
    Tanimoto, T. T. (1968).
    An elementary mathematical theory of classification and prediction, IBM Report (November, 1958),
    cited in: G. Salton, Automatic Information Organization and Retrieval.

    Same as:
        Jaccard index

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y)

    return a / (a + b + c)


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



    