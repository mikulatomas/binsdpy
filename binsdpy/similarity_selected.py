import math
import functools

from binsdpy.utils import (
    operational_taxonomic_units,
    BinaryFeatureVector,
    BinaryFeatureVectorEmpty,
)


def austin_colwell(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Austin-Colwell [AC]

    Austin, Brian, and Rita R. Colwell. "Evaluation of some coefficients for use in numerical taxonomy of microorganisms." International Journal of Systematic and Evolutionary Microbiology 27, no. 3 (1977): 204-210.

    - pod nazvem angular transformation of SMC

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return 2 / math.pi * math.asin(math.sqrt((a + d) / (a + b + c + d)))


def anderberg(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Anderberg [And]

    Anderberg, Michael R. "Cluster Analysis for Applications (New York and London, Academic Press)." (1973).

    - p.80, 86

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


def braun_blanquet(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Braun-Banquet [BB]

    Braun-Blanquet, Josias. "Plant sociology. The study of plant communities." Plant sociology. The study of plant communities. First ed. (1932).

    - stranu jsem nenasel

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / max(a + b, a + c)


def baroni_urbani_buser1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Baproni-Urbani-Buser [BUB]

    Baroni-Urbani, Cesare, and Mauro W. Buser. "Similarity of binary data." Systematic Zoology 25, no. 3 (1976): 251-259.

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
    """Baroni-Urbani-Buser 2 [BU2]

    Baroni-Urbani, Cesare, and Mauro W. Buser. "Similarity of binary data." Systematic Zoology 25, no. 3 (1976): 251-259.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (math.sqrt(a * d) + a - b - c) / (math.sqrt(a * d) + a + b + c)


def cohen(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Cohen [Coh]

    Cohen, Jacob. "A coefficient of agreement for nominal scales." Educational and psychological measurement 20, no. 1 (1960): 37-46.

    - pro 2x2 situaci neni primo uveden vzorec
    - je mozne ze vzorec neodpovida prehledovym clankum

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (2 * (a * d - b * c)) / ((a + b) * (b + d) + (a + c) * (c + d))


def cole(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Cole [Col]

    Cole, LaMont C. "The measurement of partial interspecific association." Ecology 38, no. 2 (1957): 226-233.

    - presny vzorec pro 2x2 jsem nenasel

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    # return (math.sqrt(2) * (a * d - b * c)) / math.sqrt(
    #     (a * d - b * c) ** 2 - (a + b) * (a + c) * (b + d) * (c + d)
    # )

    if (a * d) >= (b * c):
        return (a * d - b * c) / ((a + b) * (b + d))
    elif (a * d) < (b * c) and a <= d:
        return (a * d - b * c) / ((a + b) * (a + c))
    else:
        return (a * d - b * c) / ((b + d) * (c + d))


def cole1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Cole (Cole I) [Co1]

    Cole, LaMont C. "The measurement of partial interspecific association." Ecology 38, no. 2 (1957): 226-233.

    - presny vzorec pro 2x2 jsem nenasel

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d - b * c) / ((a + c) * (c + d))


def cole2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Cole (Cole II) [Co2]

    Cole, LaMont C. "The measurement of partial interspecific association." Ecology 38, no. 2 (1957): 226-233.

    - presny vzorec pro 2x2 jsem nenasel

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d - b * c) / ((a + b) * (b + d))


def consonni_todeschini1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Consonni-Todeschini 1 [CT1]

    Consonni, Viviana, and Roberto Todeschini. "New similarity coefficients for binary data." Match-Communications in Mathematical and Computer Chemistry 68, no. 2 (2012): 581.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return math.log(1 + a + d) / math.log(1 + a + b + c + d)


def consonni_todeschini2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Consonni-Todeschini 2 [CT2]

    Consonni, Viviana, and Roberto Todeschini. "New similarity coefficients for binary data." Match-Communications in Mathematical and Computer Chemistry 68, no. 2 (2012): 581.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (math.log(1 + n) - math.log(1 + b + c)) / math.log(1 + n)


def consonni_todeschini3(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Consonni-Todeschini 3 [CT3]

    Consonni, Viviana, and Roberto Todeschini. "New similarity coefficients for binary data." Match-Communications in Mathematical and Computer Chemistry 68, no. 2 (2012): 581.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return math.log(1 + a) / math.log(1 + a + b + c + d)


def consonni_todeschini4(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Consonni-Todeschini 4 [CT4]

    Consonni, Viviana, and Roberto Todeschini. "New similarity coefficients for binary data." Match-Communications in Mathematical and Computer Chemistry 68, no. 2 (2012): 581.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return math.log(1 + a) / math.log(1 + a + b + c)


def consonni_todeschini5(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Consonni-Todeschini 5 [CT5]

    Consonni, Viviana, and Roberto Todeschini. "New similarity coefficients for binary data." Match-Communications in Mathematical and Computer Chemistry 68, no. 2 (2012): 581.

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (math.log(1 + a * d) - math.log(1 + b * c)) / math.log(1 + (n * n) / 4)


def dennis(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Dennis [Den]

    ???

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d - b * c) / math.sqrt((a + b + c + d) * (a + b) * (a + c))


def dice1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Dice 1 [Di1]

    Dice, Lee R. "Measures of the amount of ecologic association between species." Ecology 26, no. 3 (1945): 297-302.

    - konkretni vzorec jsem nenasel

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, _, _ = operational_taxonomic_units(x, y, mask)

    return a / (a + b)


def dice2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Dice 2 [Di2]

    Dice, Lee R. "Measures of the amount of ecologic association between species." Ecology 26, no. 3 (1945): 297-302.

    - konkretni vzorec jsem nenasel

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, _, c, _ = operational_taxonomic_units(x, y, mask)

    return a / (a + c)


def dispersion(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """dispersion [dis]

    ???

    - pouze v review clancich zatim

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (a * d - b * c) / (n * n)


def driver_kroeber(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Driver-Kroeber (Ochiai, cosine) [DK]

    Driver, Harold Edson, and Alfred Louis Kroeber. Quantitative expression of cultural relationships. Vol. 31, no. 4. Berkeley: University of California Press, 1932.

    - paper jsem nenasel

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """

    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / math.sqrt((a + b) * (a + c))


def eyraud(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Eyraud [Eyr]

    Eyraud, Henri. "Les principes de la mesure des correlations." Ann. Univ. Lyon, III. Ser., Sect. A 1, no. 30-47 (1936): 111.

    - paper jsem nenasel

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


def fager_mcgowan(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Fager-McGowan [FM]

    Fager, Edward W. "Determination and analysis of recurrent groups." Ecology 38, no. 4 (1957): 586-595.

    - presny vzorec nevidim

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    # return a / math.sqrt((a + b) * (a + c)) - max(a + b, a + c) / 2

    return a / math.sqrt((a + b) * (a + c)) - 1 / (2 * math.sqrt(max(a + b, a + c)))

def faith(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Faith [Fai]

    Faith, Daniel P. "Asymmetric binary similarity measures." Oecologia 57 (1983): 287-290.

    - vzorec presne v clanku

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + 0.5 * d) / (a + b + c + d)


def forbes1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Forbes [For]

    Forbes, Stephen A. On the local distribution of certain Illinois fishes: an essay in statistical ecology. Vol. 7. Illinois State Laboratory of Natural History, 1907.

    - presny vzorec nevidim

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (n * a) / ((a + b) * (a + c))


def forbes2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Forbes 2 [Fo2]

    Forbes, Stephen A. "Method of determining and measuring the associative relations of species." Science 61, no. 1585 (1925): 518-524.

    - udajne, clanek jsem nenalezl

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (n * a - (a + b) * (a + c)) / (n * min(a + b, a + c) - (a + b) * (a + c))


def fossum(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Fossum [Fos]

    Fossum, Earl G., and Gilbert Kaskey. Optimization and standardization of information retrieval language and systems. SPERRY RAND CORP PHILADELPHIA PA UNIVAC DIV, 1966.

    - vzorec jsem presne nenasel

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (n * (a - 0.5) ** 2) / ((a + b) * (a + c))


def gilbert_wells(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Gilbert-Wells [GW]

    Gilbert, N., and Terry CE Wells. "Analysis of quadrat data." The Journal of Ecology (1966): 675-685.

    -

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)
    a, b, c, d = int(a), int(b), int(c), int(d)

    n = a + b + c + d

    return math.log(
        (n**3) / (2 * math.pi * (a + b) * (a + c) * (b + d) * (c + d))
        + 2
        * math.log(
            (
                math.factorial(n)
                * math.factorial(a)
                * math.factorial(b)
                * math.factorial(c)
                * math.factorial(d)
            )
            / (
                math.factorial(a + b)
                * math.factorial(a + c)
                * math.factorial(b + d)
                * math.factorial(c + d)
            )
        )
    )
    # return math.log(a) - math.log(n) - math.log((a + b) / n) - math.log((a + c) / n)


def gleason(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Gleason (Dice, S rensen, Czekanowski) [Gle]

    Gleason, Henry Allan. "Some applications of the quadrat method." Bulletin of the Torrey Botanical Club 47, no. 1 (1920): 21-33.

    - vzorec nevidim

    Dice, Lee R. "Measures of the amount of ecologic association between species." Ecology 26, no. 3 (1945): 297-302.

    - vzorec nevidim

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (2 * a) / (2 * a + b + c)


def goodman_kruskal1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Goodman-Kruskal I [GK1]

    Goodman, Leo A., William H. Kruskal, Leo A. Goodman, and William H. Kruskal. Measures of association for cross classifications. Springer New York, 1979.

    - kniha, stranu jsem zatim nenasel

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


def goodman_kruskal2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Goodman-Kruskal II [GK2]

    Goodman, Leo A., William H. Kruskal, Leo A. Goodman, and William H. Kruskal. Measures of association for cross classifications. Springer New York, 1979.

    - kniha, stranu jsem zatim nenasel

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (2 * min(a, d) - b - c) / (2 * min(a, d) + b + c)


def gower(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Gower [Gow]

    Gower, John C., and Pierre Legendre. "Metric and Euclidean properties of dissimilarity coefficients." Journal of classification 3 (1986): 5-48.

    - v clanku se vyskytuje vzorec kde je a * d misto a + d

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))


def hamman(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Hamman [Ham]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d - b - c) / (a + b + c + d)


def harris_lahey(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Harris-Lahey [HL]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return ((a * (2 * d + b + c)) / (2 * (a + b + c))) + (
        (d * (2 * a + b + c) / (2 * (b + c + d)))
    )


def hawkins_dotson(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Hawkins-Dotson [HD]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return 0.5 * (((a / (a + b + c)) + (d / (d + b + c))))


def intersection(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Intersection [Int]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, _, _, _ = operational_taxonomic_units(x, y, mask)

    return a


def inner_product(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """inner product [Ip]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, _, _, d = operational_taxonomic_units(x, y, mask)

    return a + d


def jaccard(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Jaccard (Jaccard-Tanimoto) [Jac]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / (a + b + c)


def sw_jaccard(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Jaccard (3W-Jaccard) [Jac3]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (3 * a) / (3 * a + b + c)


def kulczynski1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Kulczynski 1 [Ku1]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / (b + c)


def kulczynski2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Kulczynski 2 (Kulczynski II, Driver-Kroeber) [Ku2]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return 0.5 * ((a / (a + b)) + (a / (a + c)))


def johnson(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Johnson [Joh]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / (a + b) + a / (a + c)


def van_der_maarel(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """van der Maarel [Maa]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (2 * a - b - c) / (2 * a + b + c)


def maxwell_pilliner(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Maxwell-Pilliner [MP]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (2 * (a * d - b * c)) / ((a + b) * (c + d) + (a + c) * (b + d))


def mcconnaughey(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """McConnaughey [McC]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (a * a - b * c) / ((a + b) * (a + c))


def michael(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Michael [Mic]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return 4 * (a * d - b * c) / ((a + d) ** 2 + (b + c) ** 2)


def mountford(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Mountford [Mou]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (2 * a) / (a * b + a * c + 2 * b * c)


def pearson1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Pearson 1 [Pe1]

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
    """Pearson 2 [Pe2]

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
    """Pearson-Heron (Phi) [PH1]

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
    """Pearson 3 [Pe3]

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
    """Pearson-Heron 2 [PH2]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return float(
        math.cos((math.pi * math.sqrt(b * c)) / (math.sqrt(a * d) + math.sqrt(b * c)))
    )


def peirce1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Peirce 1 [Pei1]
    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d - b * c) / ((a + b) * (c + d))


def peirce2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Peirce 2 [Pei2]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d - b * c) / ((a + c) * (b + d))


def peirce3(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Peirce 3 [Pei3]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * b + b * c) / (a * b + 2 * b * c + c * d)


def rogot_goldberg(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Rogot-Goldberg [RG]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a / (2 * a + b + c)) + (d / (2 * d + b + c))


def russell_rao(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Russel-Rao [RR]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return a / (a + b + c + d)


def rogers_tanimoto(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Roges-Tanimoto [RT]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d) / (a + 2 * (b + c) + d)


def scott(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Scott [Sco]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (4 * a * d - (b + c) ** 2) / ((2 * a + b + c) * (2 * d + b + c))


def simpson(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Simpson [Sim]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / min(a + b, a + c)


def smc(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """simple matching coe cient (Sokal-Michener) [SMC]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d) / (a + b + c + d)


def sokal_sneath1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Sokal-Sneath 1 [SS1]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / (a + 2 * b + 2 * c)


def sokal_sneath2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Sokal-Sneath 2 [SS2]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (2 * (a + d)) / (2 * (a + d) + b + c)


def gower_legendre(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Gower-Legendre [GL]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d) / (a + 0.5 * (b + c) + d)


def sokal_sneath3(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Sokal-Sneath 3 [SS3]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return 0.25 * (a / (a + b) + a / (a + c) + d / (b + d) + d / (c + d))


def sokal_sneath4(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Sokal-Sneath 4 [SS4]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return a / math.sqrt((a + b) * (a + c)) * d / math.sqrt((b + d) * (c + d))


def sokal_sneath3a(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Sokal-Sneath 3a [SS3a]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a + d) / (b + c)


def sokal_sneath4a(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Sokal-Sneath 4a (Ochiai) [SS4a]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))


def sorgenfrei(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Sorgenfrei [Sor]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return (a * a) / ((a + b) * (a + c))


def stiles(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Stiles [Sti]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    # t = abs(a * d - b * c) - 0.5 * n
    t = abs(a * n - b * c) - 0.5 * n

    # return math.log10((n * t * t) / ((a + b) * (a + c) * (b + d) * (c + d)))

    return math.log10((n * t * t) / (b * c * (n - b) * (n - c)))


def tanimoto(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Tanimoto [Tan]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, _ = operational_taxonomic_units(x, y, mask)

    return a / ((a + b) + (a + c) - a)


def tarantula(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Tarantula [Tar]

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
    """Ample [Amp]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """

    return abs(tarantula(x, y, mask))


def tarwid(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Tarwid [Ewd]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    n = a + b + c + d

    return (n * a - (a + b) * (a + c)) / (n * a + (a + b) * (a + c))


def yule1(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Yule (Yule Q) [Yu1]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (a * d - b * c) / (a * d + b * c)


def yule2(
    x: BinaryFeatureVector,
    y: BinaryFeatureVector,
    mask: BinaryFeatureVectorEmpty = None,
) -> float:
    """Yule (Yule W) [Yu2]

    Args:
        x (BinaryFeatureVector): binary feature vector
        y (BinaryFeatureVector): binary feature vector

    Returns:
        float: similarity of given vectors
    """
    a, b, c, d = operational_taxonomic_units(x, y, mask)

    return (math.sqrt(a * d) - math.sqrt(b * c)) / (math.sqrt(a * d) + math.sqrt(b * c))
