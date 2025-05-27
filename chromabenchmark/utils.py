# External imports
import re
import difflib

def find_query_despite_whitespace(document, query):
    """Finds the query despite whitespace."""
     # Normalize spaces and newlines in the query
    normalized_query = re.sub(r'\s+', ' ', query).strip()

    # Create a regex pattern from the normalized query to match any whitespace characters between words
    pattern = r'\s*'.join(re.escape(word) for word in normalized_query.split())

    # Compile the regex to ignore case and search for it in the document
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)

    if match:
        return document[match.start(): match.end()], match.start(), match.end()
    else:
        return None

def get_similarity_ratio(str1, str2):
    """
    Calculate similarity ratio between two strings using difflib.
    Returns a value between 0 and 100 (similar to fuzzywuzzy's scoring).
    """
    # Normalize and tokenize strings for better comparison
    tokens1 = sorted(str1.lower().split())
    tokens2 = sorted(str2.lower().split())

    # Use SequenceMatcher to get similarity ratio
    matcher = difflib.SequenceMatcher(None, ' '.join(tokens1), ' '.join(tokens2))
    return matcher.ratio() * 100

def find_best_match(target, choices, threshold=98):
    """
    Find the best matching string from a list of choices.
    Returns tuple of (best_match, score) or None if no match above threshold.
    """
    best_match = None
    best_score = 0

    for choice in choices:
        if not choice.strip():  # Skip empty strings
            continue

        score = get_similarity_ratio(target, choice)
        if score > best_score:
            best_score = score
            best_match = choice

    if best_score >= threshold:
        return (best_match, best_score)
    else:
        return None


def rigorous_document_search(document: str, target: str):
    """
    This function performs a rigorous search of a target string within a document.
    It handles issues related to whitespace, changes in grammar, and other minor text alterations.
    The function first checks for an exact match of the target in the document.
    If no exact match is found, it performs a raw search that accounts for variations in whitespace.
    If the raw search also fails, it splits the document into sentences and uses fuzzy matching
    to find the sentence that best matches the target.

    Args:
        document (str): The document in which to search for the target.
        target (str): The string to search for within the document.
    Returns:
        tuple: A tuple containing the best match found in the document, its start index, and its end index.
        If no match is found, returns None.
    """
    if target.endswith('.'):
        target = target[:-1]

    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index
    else:
        raw_search = find_query_despite_whitespace(document, target)
        if raw_search is not None:
            return raw_search

    # Split the text into sentences
    sentences = re.split(r'[.!?]\s*|\n', document)

    # Find the sentence that matches the query best
    best_match = find_best_match(target, sentences, threshold=98)

    if best_match is None:
        return None

    reference = best_match[0]
    start_index = document.find(reference)
    end_index = start_index + len(reference)
    return reference, start_index, end_index