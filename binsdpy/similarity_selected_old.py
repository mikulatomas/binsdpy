import math

from binsdpy.utils import (
    operational_taxonomic_units,
    BinaryFeatureVector,
    BinaryFeatureVectorEmpty,
)

def jaccard(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Jaccard similarity

    Same as:
        Tanimoto coefficient

    Jaccard, P. (1908).
    Nouvelles recherches sur la distribution florale.
    Bull. Soc. Vaud. Sci. Nat., 44, 223-270.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / (a + b + c)


def sorenson_dice(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Sorenson-Dice similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (2 * a) / (2 * a + b + c)


def czekanowski(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Czekanowski similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (2 * a) / (2 * a + b + c)


def nei_li(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Czekanowski similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (2 * a) / (2 * a + b + c)


def sw_jaccard(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """SW Jaccard similarity

    Jaccard, P. (1908).
    Nouvelles recherches sur la distribution florale.
    Bull. Soc. Vaud. Sci. Nat., 44, 223-270.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (3 * a) / (3 * a + b + c)


def sokal_sneath1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
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
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / (a + 2 * b + 2 * c)


def sokal_michener(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Sokal-Michener similarity (also called simple matching coefficient)

    Sokal, R. R. (1958).
    A statistical method for evaluating systematic relationships.
    Univ. Kansas, Sci. Bull., 38, 1409-1438.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d) / (a + b + c + d)


def sokal_sneath2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
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
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (2 * (a + d)) / (2 * (a + d) + b + c)


def rogers_tanimoto(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
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
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d) / (a + 2 * (b + c) + d)


def faith(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
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
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + 0.5 * d) / (a + b + c + d)


def gower_legendre(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
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
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d) / (a + 0.5 * (b + c) + d)


def intersection(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Intersection similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, _, _, _ = operational_taxonomic_units(x, y, mask)

    return a


def innerproduct(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Inner product similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, _, _, d = operational_taxonomic_units(x, y, mask)

    return a + d


def russell_rao(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Russel-Rao similarity

    Russell, P. F., & Rao, T. R. (1940).
    On habitat and association of species of anopheline larvae in south-eastern Madras.
    Journal of the Malaria Institute of India, 3(1).

    Rao, C. R. (1948).
    The utilization of multiple measurements in problems of biological classification.
    Journal of the Royal Statistical Society. Series B (Methodological), 10(2), 159-203.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return a / (a + b + c + d)


def cosine(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Cosine similarity

    Ochiai, A. (1957).
    Zoogeographic studies on the soleoid fishes found in Japan and its neighbouring regions.
    Bulletin of Japanese Society of Scientific Fisheries, 22, 526-530.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """

    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / math.sqrt((a + b) * (a + c))


def ochiai_1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Ochiai similarity

    Ochiai, A. (1957).
    Zoogeographic studies on the soleoid fishes found in Japan and its neighbouring regions.
    Bulletin of Japanese Society of Scientific Fisheries, 22, 526-530.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """

    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / math.sqrt((a + b) * (a + c))


def otsuka(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Otsuka similarity

    Ochiai, A. (1957).
    Zoogeographic studies on the soleoid fishes found in Japan and its neighbouring regions.
    Bulletin of Japanese Society of Scientific Fisheries, 22, 526-530.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """

    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / (((a + b) * (a + c)) ** 0.5)


def gilbert_wells(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Gilbert-Wells similarity

    Gilbert, G. K. (1884).
    Finley's tornado predictions. American Meteorological Journal.
    A Monthly Review of Meteorology and Allied Branches of Study (1884-1896), 1(5), 166.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return math.log10(a) - math.log10(n) - math.log10((a + b) / n) - math.log10((a + c) / n)


def forbes1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Forbesi similarity (v1)

    Forbes, S. A. (1907).
    On the local distribution of certain Illinois fishes: an essay in statistical ecology (Vol. 7).
    Illinois State Laboratory of Natural History.

    Forbes, S. A. (1925).
    Method of determining and measuring the associative relations of species.
    Science, 61(1585), 518-524.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (n * a) / ((a + b) * (a + c))


def fossum(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Fossum similarity

    Holliday, J. D., Hu, C. Y., & Willett, P. (2002).
    Grouping of coefficients for the calculation of inter-molecular similarity and dissimilarity using 2D fragment bit-strings.
    Combinatorial chemistry & high throughput screening, 5(2), 155-166.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (n * (a - 0.5) ** 2) / ((a + b) * (a + c))


def sorgenfrei(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Sorgenfrei similarity

    Sorgenfrei, T. (1958).
    Molluscan Assemblages from the Marine Middle Miocene of South Jutland and their Environments. Vol. II.
    Danmarks Geologiske Undersøgelse II. Række, 79, 356-503.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (a * a) / ((a + b) * (a + c))


def mountford(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Mountford similarity

    Mountford, M. D. (1962).
    An index of similarity and its application to classificatory problem.
    Progress in soil zoology"(ed. Murphy, PW), 43-50.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / (0.5 * (a * b + a * c) + b * c)


def mcconnaughey(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """McConnaughey similarity

    McConnaughey, B. H. (1964).
    The determination and analysis of plankton communities.
    Lembaga Penelitian Laut.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (a * a - b * c) / ((a + b) * (a + c))


def tarwid(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Tarwid similarity

    Tarwid, K. (1960).
    Szacowanie zbieznosci nisz ekologicznych gatunkow droga oceny prawdopodobienstwa spotykania sie ich w polowach.
    Ecol Polska B (6), 115-130.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (n * a - (a + b) * (a + c)) / (n * a + (a + b) * (a + c))


def kulczynski2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Kulczynski similarity (v2)

    Stanisław Kulczynśki. (1927).
    Die pflanzenassoziationen der pieninen.
    Bulletin International de l'Academie Polonaise des Sciences et des Lettres, Classe des Sciences Mathematiques et Naturelles, B (Sciences Naturelles), pages 57–203.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return ((a / 2) * (2 * a + b + c)) / ((a + b) * (a + c))


def driver_kroeber(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Driver and Kroeber

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (a / 2) * ((1 / (a + b)) + (1 / (a + c)))


def johnson(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Johnson similarity

    Johnson, S. C. (1967).
    Hierarchical clustering schemes.
    Psychometrika, 32(3), 241-254.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / (a + b) + a / (a + c)


def dennis(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Dennis similarity

    Dennis, S. F. (1965).
    The Construction of a Thesaurus Automatically From.
    In Statistical Association Methods for Mechanized Documentation: Symposium Proceedings (Vol. 269, p. 61).
    US Government Printing Office.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d - b * c) / math.sqrt((a + b + c + d) * (a + b) * (a + c))


def simpson(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Simpson similarity

    Simpson, E. H. (1949).
    Measurement of diversity.
    Nature, 163(4148), 688-688.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / min(a + b, a + c)


def braun_blanquet(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Braun-Banquet similarity

    Braun-Blanquet, J. (1932).
    Plant sociology. The study of plant communities. Plant sociology.
    The study of plant communities. First ed.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / max(a + b, a + c)


def fager_mcgowan(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Fager-McGowan similarity

    Fager, E. W. (1957).
    Determination and analysis of recurrent groups.
    Ecology, 38(4), 586-595.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / math.sqrt((a + b) * (a + c)) - max(a + b, a + c) / 2


def forbes2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Forbesi similarity (v2)

    Forbes, S. A. (1907).
    On the local distribution of certain Illinois fishes: an essay in statistical ecology (Vol. 7).
    Illinois State Laboratory of Natural History.

    Forbes, S. A. (1925).
    Method of determining and measuring the associative relations of species.
    Science, 61(1585), 518-524.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (n * a - (a + b) * (a + c)) / (n * min(a + b, a + c) - (a + b) * (a + c))


def sokal_sneath4(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
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
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a / (a + b) + a / (a + c) + d / (b + d) + d / (b + d)) / 4


def gower(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
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
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))


def pearson1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Pearson Chi-Squared similarity

    Pearson, K., & Heron, D. (1913).
    On theories of association.
    Biometrika, 9(1/2), 159-315.

    Pearson, K. X. (1900).
    On the criterion that a given system of deviations from the probable in the case 538 of a correlated system of variables is such that it can be reasonably supposed to have arisen 539 from random sampling.
    London, Edinburgh, Dublin Philos. Mag. J. Sci, 540, 50.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (n * ((a * d - b * c) ** 2)) / ((a + b) * (a + c) * (b + d) * (c + d))


def pearson2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Pearson similarity (v2)

    Pearson, K., & Heron, D. (1913).
    On theories of association.
    Biometrika, 9(1/2), 159-315.

    Pearson, K. X. (1900).
    On the criterion that a given system of deviations from the probable in the case 538 of a correlated system of variables is such that it can be reasonably supposed to have arisen 539 from random sampling.
    London, Edinburgh, Dublin Philos. Mag. J. Sci, 540, 50.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    x_2 = pearson1(x, y, mask)

    return math.sqrt(x_2 / (a + b + c + d + x_2))


def pearson_heron1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Pearson similarity (v3)

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d - b * c) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))


def pearson3(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Pearson similarity (v3)

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    p = pearson_heron1(x, y, mask)

    return math.sqrt(p / (a + b + c + d + p))


def pearson_heron2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Pearson similarity (v3)

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    float(math.cos((math.pi * math.sqrt(b * c)) / (math.sqrt(a * d) + math.sqrt(b * c))))


def sokal_sneath3(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
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
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d) / (b + c)


def sokal_sneath5(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
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
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))


def cole(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Cole similarity

    Cole, L. C. (1957).
    The measurement of partial interspecific association.
    Ecology, 38(2), 226-233.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (math.sqrt(2) * (a * d - b * c)) / math.sqrt((a * d - b * c) ** 2 - (a + b) * (a + c) * (b + d) * (c + d))


def stiles(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Stiles similarity

    Stiles, H. E. (1961).
    The association factor in information retrieval.
    Journal of the ACM (JACM), 8(2), 271-279.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    t = abs(a * d - b * c) - 0.5 * n

    return math.log10((n * t * t) / ((a + b) * (a + c) * (b + d) * (c + d)))


def ochiai2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
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
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))


def yuleq(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Yule's Q similarity

    Yule, G. U. (1912).
    On the methods of measuring association between two attributes.
    Journal of the Royal Statistical Society, 75(6), 579-652.

    Yule, G. U. (1900).
    On the association of attributes in statistics.
    Philosophical Transactions of the Royal Society. Series A, 194, 257-319.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d - b * c) / (a * d + b * c)


def yulew(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Yule's W similarity

    Yule, G. U. (1912).
    On the methods of measuring association between two attributes.
    Journal of the Royal Statistical Society, 75(6), 579-652.

    Yule, G. U. (1900).
    On the association of attributes in statistics.
    Philosophical Transactions of the Royal Society. Series A, 194, 257-319.


    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (math.sqrt(a * d) - math.sqrt(b * c)) / (math.sqrt(a * d) + math.sqrt(b * c))


def kulczynski1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Kulczynski similarity (v1)

    Stanisław Kulczynśki. (1927).
    Die pflanzenassoziationen der pieninen.
    Bulletin International de l'Academie Polonaise des Sciences et des Lettres, Classe des Sciences Mathematiques et Naturelles, B (Sciences Naturelles), pages 57–203.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / (b + c)


def tanimoto(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Tanimoto similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / ((a + b) + (a + c) - a)


def disperson(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Disperson similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (a * d - b * c) / (n * n)


def hamman(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Hamman similarity

    Hamann, U. (1961).
    Merkmalsbestand und verwandtschaftsbeziehungen der farinosae: ein beitrag zum system der monokotyledonen.
    Willdenowia, 639-768.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d - b - c) / (a + b + c + d)


def michael(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Michael similarity

    Michael, E. L. (1920).
    Marine ecology and the coefficient of association: a plea in behalf of quantitative biology.
    Journal of Ecology, 8(1), 54-59.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return 4 * (a * d - b * c) / ((a + d) ** 2 + (b + c) ** 2)


def goodman_kruskal(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Goodman-Kruskal similarity (v1)

    Goodman, L. A., & Kruskal, W. H. (1979).
    Measures of association for cross classifications.
    Measures of association for cross classifications, 2-34.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    p1 = max(a, b) + max(c, d) + max(a, c) + max(b, d)
    p2 = max(a + c, b + d) + max(a + b, c + d)

    return (p1 - p2) / (2 * n - p2)


def anderberg(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Anderberg similarity

    Anderberg, M. R. (2014).
    Cluster analysis for applications: probability and mathematical statistics: a series of monographs and textbooks (Vol. 19).
    Academic press.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    p1 = max(a, b) + max(c, d) + max(a, c) + max(b, d)
    p2 = max(a + c, b + d) + max(a + b, c + d)

    return (p1 - p2) / (2 * n)


def baroni_urbani_buser1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Baroni-Urbani similarity (v1)

    Baroni-Urbani, C., & Buser, M. W. (1976).
    Similarity of binary data.
    Systematic Zoology, 25(3), 251-259.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (math.sqrt(a * d) + a) / (math.sqrt(a * d) + a + b + c)


def baroni_urbani_buser2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Baroni-Urbani similarity (v1)

    Baroni-Urbani, C., & Buser, M. W. (1976).
    Similarity of binary data.
    Systematic Zoology, 25(3), 251-259.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (math.sqrt(a * d) + a - b + c) / (math.sqrt(a * d) + a + b + c)


def peirce(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Peirce similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return ((a * b) + (b * c)) / (a * b + 2 * b * c + c * d)


def eyraud(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Eyraud similarity

    Eyraud, H. (1936).
    Les principes de la mesure des correlations.
    Ann. Univ. Lyon, III. Ser., Sect. A, 1(30-47), 111.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (n * n * (n * a - (a + b) * (a + c))) / (
        (a + b) * (a + c) * (b + d) * (c + d)
    )


def tarantula(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Tarantula similarity

    Jones, J. A., & Harrold, M. J. (2005, November).
    Empirical evaluation of the tarantula automatic fault-localization technique.
    In Proceedings of the 20th IEEE/ACM international Conference on Automated software engineering (pp. 273-282).

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * (c + d)) / (c * (a + b))


def ample(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Ample similarity

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """

    return abs(tarantula(x, y, mask))


# Dalsi paper
# A comparison of 71 binary similarity coefficients: The effect of base rates

