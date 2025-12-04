import re

def extract_product_block(full_text: str):
    """
    Attempts to extract a block starting from 'MTV...' up to the 'Model X' phrase.
    Returns matched substring or None.
    """
    if not full_text:
        return None
    # Normalize whitespace
    s = re.sub(r"\s+", " ", full_text).strip()
    # Pattern: start with MTV (MTV03PA/A or similar), lazy match up to 'Model' and the following identifier
    m = re.search(r"(M\S*.*?Model\s*[A0-9A-Za-z\-]+)", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # If not found, try looser pattern: find 'MTV' and return that line + next words
    m2 = re.search(r"(MTV\S*(?:[^\n]{0,120}))", s, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
