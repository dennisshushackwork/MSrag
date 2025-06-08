"""
This file helps find exact or very similar text snippets within a larger document, even when there are minor inconsistencies.
ts primary purpose is to reliably locate specific "target" strings (like a reference answer) within a "document"
(like a source text), accounting for common real-world text variations.

This file is meant to accurately map the extracted text "chunks" or "references"
 back to their precise locations within the original, full document.
"""

import re
import difflib
from enum import Enum
from typing import Optional, Tuple


def normalize_text(text: str) -> str:
    """Normalize text for better matching by handling common tokenization artifacts."""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()

    # Handle common tokenization artifacts
    text = text.replace('<unk>', '')  # Remove unknown tokens
    text = text.replace('@-@', '-')  # Convert back to hyphens
    text = text.replace(' @,@ ', ', ')
    text = text.replace(' @.@ ', '. ')

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    return text


def find_with_sliding_window(document: str, target: str, min_similarity: float = 0.85) -> Optional[
    Tuple[str, int, int]]:
    """
    Use a sliding window approach to find the best match in the document.
    This handles cases where the chunk might not align perfectly with sentence boundaries.
    """
    target_len = len(target)
    best_match = None
    best_ratio = 0
    best_start = -1

    # Try different window sizes around the target length
    for window_size in [target_len, int(target_len * 0.9), int(target_len * 1.1)]:
        for start in range(0, len(document) - window_size + 1, max(1, window_size // 10)):
            window = document[start:start + window_size]

            # Calculate similarity
            ratio = difflib.SequenceMatcher(None, target.lower(), window.lower()).ratio()

            if ratio > best_ratio and ratio >= min_similarity:
                best_ratio = ratio
                best_match = window
                best_start = start

    if best_match:
        return best_match, best_start, best_start + len(best_match)
    return None


def find_query_despite_whitespace(document: str, query: str) -> Optional[Tuple[str, int, int]]:
    """Enhanced whitespace-tolerant search."""
    # Normalize the query
    normalized_query = re.sub(r'\s+', ' ', query).strip()

    # Create a more flexible pattern that allows for various whitespace and special chars
    words = normalized_query.split()
    if not words:
        return None

    # Escape special regex characters but allow flexible spacing
    escaped_words = [re.escape(word) for word in words]
    pattern = r'\s*'.join(escaped_words)

    # Make it even more flexible by allowing optional special characters between words
    pattern = pattern.replace(r'\s\*', r'[\s<>@,.-]*')

    try:
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        match = regex.search(document)

        if match:
            return document[match.start():match.end()], match.start(), match.end()
    except re.error:
        # If regex fails, fall back to simpler approach
        pass

    return None


def rigorous_document_search(document: str, target: str) -> Optional[Tuple[str, int, int]]:
    """
    Optimized document search with multiple fallback strategies."""
    if not target or not document:
        return None

    original_target = target
    if target.endswith('.'):
        target = target[:-1]

    # Strategy 1: Direct exact match
    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index

    # Strategy 2: Normalize both texts and try exact match
    normalized_doc = normalize_text(document)
    normalized_target = normalize_text(target)

    if normalized_target in normalized_doc:
        # Find the original position in the unnormalized document
        norm_start = normalized_doc.find(normalized_target)
        # This is approximate - we'd need more complex mapping for exact positions
        # For now, use sliding window to find the best match around this area
        search_start = max(0, norm_start - 50)
        search_end = min(len(document), norm_start + len(normalized_target) + 50)
        search_area = document[search_start:search_end]

        result = find_with_sliding_window(search_area, target, min_similarity=0.8)
        if result:
            match_text, rel_start, rel_end = result
            return match_text, search_start + rel_start, search_start + rel_end

    # Strategy 3: Whitespace-tolerant regex search
    regex_result = find_query_despite_whitespace(document, target)
    if regex_result:
        return regex_result

    # Strategy 4: Sliding window with similarity matching
    window_result = find_with_sliding_window(document, target, min_similarity=0.80)
    if window_result:
        return window_result

    # Strategy 5: Try with first and last parts of the chunk
    if len(target) > 100:
        # Try matching just the beginning
        start_part = target[:50]
        start_result = find_query_despite_whitespace(document, start_part)
        if start_result:
            # Found the start, now try to extend to find the full chunk
            found_start = start_result[1]
            # Look for a reasonable endpoint
            for end_offset in [len(target), int(len(target) * 1.1), int(len(target) * 0.9)]:
                if found_start + end_offset <= len(document):
                    candidate = document[found_start:found_start + end_offset]
                    ratio = difflib.SequenceMatcher(None, target.lower(), candidate.lower()).ratio()
                    if ratio >= 0.8:
                        return candidate, found_start, found_start + end_offset

    # Strategy 6: Sentence-based fuzzy matching (original approach)
    sentences = re.split(r'[.!?]\s*|\n', document)
    sentences = [s.strip() for s in sentences if s.strip()]

    best_match = None
    best_score = 0

    for sentence in sentences:
        score = difflib.SequenceMatcher(None, target.lower(), sentence.lower()).ratio() * 100
        if score > best_score and score >= 85:  # Lowered threshold
            best_score = score
            best_match = sentence

    if best_match:
        start_index = document.find(best_match)
        if start_index != -1:
            end_index = start_index + len(best_match)
            return best_match, start_index, end_index

    # Strategy 7: Try partial matches for very long chunks
    if len(target) > 200:
        # Split target into smaller parts and try to find consecutive matches
        chunk_size = 100
        for i in range(0, len(target) - chunk_size, chunk_size // 2):
            part = target[i:i + chunk_size]
            result = find_query_despite_whitespace(document, part)
            if result:
                # Found a part, try to expand around it
                found_start = result[1]
                # Try to find the best boundaries
                for start_offset in range(-50, 51, 10):
                    for length in [len(target), int(len(target) * 1.1), int(len(target) * 0.9)]:
                        search_start = max(0, found_start + start_offset)
                        search_end = min(len(document), search_start + length)
                        candidate = document[search_start:search_end]

                        ratio = difflib.SequenceMatcher(None, target.lower(), candidate.lower()).ratio()
                        if ratio >= 0.75:
                            return candidate, search_start, search_end

    return None


class Language(str, Enum):
    """Enum of the programming languages."""
    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"