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
    m = re.search(r"(PA/A\S*.*?Model\s*[A0-9A-Za-z\-]+)", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # If not found, try looser pattern: find 'MTV' and return that line + next words
    m2 = re.search(r"(MTV\S*(?:[^\n]{0,120}))", s, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
def extract_info(ocr_results):
    """
    ocr_results: list of dicts containing:
    ['crop', 'poly', 'score', 'save_path', 'rec_text', 'rec_score']
    """

    imei = None
    imei2 = None
    product_type = None

    # Pattern for product type: ends with number + "GB"
    product_regex = re.compile(r".*\b\d{2,4}GB\b", re.IGNORECASE)

    for r in ocr_results:
        raw = r.get("rec_text", "") or ""
        text = raw.strip()

        if not text:
            continue

        # ---------- PRODUCT TYPE ----------
        if product_type is None and product_regex.match(text):
            product_type = text

        # Normalize upper for label checks
        upper = text.upper()

        if imei is None and text.upper().startswith(("IMEI:", "IMEI/MEID", "IMEI ")):
            # Extract digits
            digits = re.findall(r"\d{14,18}", text)
            if digits:
                imei = digits[0]
        # ---------- IMEI2 ----------
        if imei2 is None and "IMEI2" in upper:
            # remove any leading 'IMEI2' (with or without punctuation/spaces)
            cleaned = re.sub(r'(?i)^.*?IMEI2[:\s-]*', '', text, count=1)
            # also try removing a stuck 'IMEI2' without separator (e.g. "IMEI2353...")
            cleaned = re.sub(r'(?i)^imei2', '', cleaned, count=1)

            digits = re.findall(r"\d+", cleaned)
            if digits:
                digits = "".join(digits)
                # if OCR left a stray leading '2' (label glued), drop it when appropriate:
                # typical IMEI length is 15. If we have 16 and it starts with '2', drop first char.
                if len(digits) == 16 and digits.startswith("2"):
                    digits = digits[1:]
                imei2 = digits

    return {
        "product_type": product_type,
        "imei": imei,
        "imei2": imei2,
    }