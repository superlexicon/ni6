"""
String matching utilities for fuzzy name comparison.

Provides Jaro-Winkler similarity - optimized for name matching.
"""


def _jaro_similarity(s1: str, s2: str) -> float:
    """
    Calculate the Jaro similarity between two strings.

    The Jaro similarity metric is designed for string comparison in record linkage
    and works well for name matching. It gives more weight to prefix matches.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0 and 1
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Length of strings
    len1, len2 = len(s1), len(s2)

    # Match distance (characters within this distance match)
    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    # Find matches
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    # Calculate Jaro similarity
    jaro_score = (
        (matches / len1) +
        (matches / len2) +
        ((matches - transpositions / 2) / matches)
    ) / 3

    return jaro_score


def jaro_winkler_similarity(s1: str, s2: str, prefix_scale: float = 0.1) -> float:
    """
    Calculate the Jaro-Winkler similarity between two strings.

    Jaro-Winkler is an extension of Jaro that gives extra weight to prefix matches.
    This makes it particularly effective for name matching.

    Args:
        s1: First string
        s2: Second string
        prefix_scale: Weight to give to prefix matches (default 0.1)

    Returns:
        Similarity score between 0 and 1
    """
    jaro = _jaro_similarity(s1, s2)

    # Find prefix length (up to 4 characters)
    prefix_len = 0
    max_prefix = min(4, min(len(s1), len(s2)))
    for i in range(max_prefix):
        if s1[i].lower() == s2[i].lower():
            prefix_len += 1
        else:
            break

    # Apply Winkler modification
    winkler_score = jaro + (prefix_len * prefix_scale * (1 - jaro))
    return min(1.0, winkler_score)


def clean_name_for_storage(name: str) -> str:
    """
    Clean a name by removing titles, honorifics, and patronymic markers
    before storing in the database.

    Removes:
    - English titles: Mr., Mrs., Ms., Miss, Dr., Prof., Rev., Hon., Sr., Jr.
    - Indian honorifics: Smt., Shri., Kum., Sri, Shrimati
    - Patronymic markers: S/O, D/O, A/L, B/O (removes the "/" and marker, keeps both names)

    Examples:
        "Mr. JOHN DOE" -> "JOHN DOE"
        "MANOGARAN S/O THANABALAN" -> "MANOGARAN THANABALAN"
        "RAVI D/O KRISHNA" -> "RAVI KRISHNA"
        "Sri PRIYA A/L KUMAR" -> "PRIYA KUMAR"

    Args:
        name: Raw name string

    Returns:
        Cleaned name, or None if input is None/empty
    """
    if not name:
        return None

    import re
    cleaned_name = name.strip()

    # List of titles/honorifics to remove (with and without dots)
    titles = [
        # English titles
        'MR\\.', 'MRS\\.', 'MS\\.', 'MISS\\.', 'DR\\.', 'PROF\\.', 'REV\\.',
        'HON\\.', 'SR\\.', 'JR\\.',
        # Indian honorifics
        'SMT\\.', 'SHRI\\.', 'KUM\\.', 'SRI\\.', 'SHRIMATI\\.',
        # Without dots
        'MR', 'MRS', 'MS', 'MISS', 'DR', 'PROF', 'REV', 'HON', 'SR', 'JR',
        'SMT', 'SHRI', 'KUM', 'SRI', 'SHRIMATI',
    ]

    # Remove titles from the beginning of the name
    # Sort by length (longest first) to avoid partial matches (e.g., "SRI" vs "SR")
    titles = sorted(titles, key=len, reverse=True)
    for title in titles:
        pattern = rf'^{title}\s*\.?\s*'
        cleaned_name = re.sub(pattern, '', cleaned_name, flags=re.IGNORECASE)

    # Remove "null" prefix (from JSON null values concatenated with name)
    # "null GAURAV RUSTAGI" -> "GAURAV RUSTAGI"
    cleaned_name = re.sub(r'^null\s+', '', cleaned_name, flags=re.IGNORECASE)

    # Handle patronymic markers: S/O, D/O, A/L, B/O
    # Remove the "/" and marker text, keep both names
    # "MANOGARAN S/O THANABALAN" -> "MANOGARAN THANABALAN"
    patronymic_patterns = [
        r'\s+S/O\s+',    # Replace " S/O " with " "
        r'\s+D/O\s+',
        r'\s+A/L\s+',
        r'\s+B/O\s+',
        r'\s+S/O\.\s*',   # With dots: " S/O. " -> " "
        r'\s+D/O\.\s*',
        r'\s+A/L\.\s*',
        r'\s+B/O\.\s*',
    ]
    for pattern in patronymic_patterns:
        cleaned_name = re.sub(pattern, ' ', cleaned_name, flags=re.IGNORECASE)

    # Clean up extra whitespace
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()

    return cleaned_name if cleaned_name else None


def normalize_for_matching(name: str) -> str:
    """
    Normalize a name for fuzzy matching.

    Removes common punctuation, extra spaces, and converts to lowercase.

    Args:
        name: Name to normalize

    Returns:
        Normalized name
    """
    if not name:
        return ""
    # Remove common punctuation and extra whitespace
    # Include forward slash to normalize S/O, D/O, A/L patterns
    normalized = name.lower()
    for char in ',.-_\'"`/':
        normalized = normalized.replace(char, ' ')
    # Collapse multiple spaces
    return ' '.join(normalized.split())


def fuzzy_match_names(name1: str, name2: str) -> float:
    """
    Perform fuzzy matching between two names using Jaro-Winkler.

    Args:
        name1: First name
        name2: Second name

    Returns:
        Similarity score between 0 and 1
    """
    norm1 = normalize_for_matching(name1)
    norm2 = normalize_for_matching(name2)
    return jaro_winkler_similarity(norm1, norm2)


def get_match_details(name1: str, name2: str) -> dict:
    """
    Get detailed comparison between two names.

    Args:
        name1: First name
        name2: Second name

    Returns:
        Dictionary with similarity metrics
    """
    norm1 = normalize_for_matching(name1)
    norm2 = normalize_for_matching(name2)

    return {
        'normalized_name1': norm1,
        'normalized_name2': norm2,
        'jaro_winkler_similarity': round(jaro_winkler_similarity(norm1, norm2), 4),
    }
