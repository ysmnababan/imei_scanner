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

def parse_amount(text):
    """
    Extract currency, numeric value, and raw text.
    Returns dict: { raw_text, currency, value }
    """

    raw_text = text.strip()

    # -------------------------------
    # 1. Extract currency prefix
    # -------------------------------
    # Matches currency-like patterns:
    # Rp, RM, USD, SGD, IDR, $, €, ¥, ₩, £, etc.
    currency_pattern = r"^(Rp|RM|IDR|USD|SGD|EUR|GBP|AUD|CAD|\$|€|¥|₩|£)"
    match = re.match(currency_pattern, raw_text, flags=re.IGNORECASE)

    currency = match.group(0) if match else None

    # -------------------------------
    # 2. Extract numeric portion
    # -------------------------------
    # Extract all digits, commas, periods:
    num = re.findall(r"[\d.,]+", raw_text)

    if not num:
        return {
            "raw_text": raw_text,
            "currency": currency,
            "value": None,
        }

    num = num[0]

    # Determine comma/dot interpretation:
    # - If both . and , appear → often European format (e.g. 1.234,56)
    # - If only . → decimal or thousands
    # - If only , → decimal or thousands
    if "." in num and "," in num:
        # Assume European-style → last separator is decimal
        if num.rfind(",") > num.rfind("."):
            numeric = num.replace(".", "").replace(",", ".")
        else:
            numeric = num.replace(",", "").replace(".", ".")
    else:
        # Simplify:
        numeric = num.replace(",", "").replace(".", "")

    # convert to integer (invoice totals are integers)
    try:
        value = int(float(numeric))
    except ValueError:
        value = None

    return {
        "raw_text": raw_text,
        "currency": currency,
        "value": value,
    }
    
def extract_total_amount(ocr_results, keyword="TOTAL TAGIHAN", 
                         vertical_tol=20, horizontal_tol=20):
    """
    ocr_results: list of dicts (each with keys including 'rec_text' and 'poly')
    keyword: keyword to locate (e.g., 'TOTAL TAGIHAN')
    vertical_tol: pixel tolerance for vertical alignment
    horizontal_tol: pixel tolerance for left/right spacing
    
    Returns: detected total amount text (string) or None
    """

    keyword = keyword.upper()

    # Normalize OCR result
    items = []
    for r in ocr_results:
        text = (r.get("rec_text") or "").strip()
        if not text:
            continue

        poly = r["poly"]
        # print(poly)
        if poly is None:
            continue
        
        # Convert poly: list of [x, y]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]

        items.append({
            "text": text,
            "upper": text.upper(),
            "poly": poly,
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "center_y": (min(ys) + max(ys)) / 2,
            "center_x": (min(xs) + max(xs)) / 2,
        })

    # -------------------------------------------------------------------------
    # STEP 1: Find the keyword position
    # -------------------------------------------------------------------------
    keyword_items = [i for i in items if keyword in i["upper"]]
    if not keyword_items:
        return None
    
    # Pick the top-most or left-most keyword occurrence
    keyword_item = sorted(keyword_items, key=lambda i: (i["min_y"], i["min_x"]))[0]

    key_y = keyword_item["center_y"]
    key_right_x = keyword_item["max_x"]

    # -------------------------------------------------------------------------
    # STEP 2: Find candidate texts on the SAME horizontal line (with tolerance)
    # -------------------------------------------------------------------------
    candidates = []
    for i in items:
        if i is keyword_item:
            continue

        # Vertical alignment check
        if abs(i["center_y"] - key_y) <= vertical_tol:
            # Horizontal position check: must be to the RIGHT of keyword
            if i["min_x"] >= key_right_x - horizontal_tol:
                candidates.append(i)

    if not candidates:
        return None

    # -------------------------------------------------------------------------
    # STEP 3: Choose the right-most / nearest text
    # -------------------------------------------------------------------------
    # Prefer the text closest to the keyword horizontally
    candidates = sorted(candidates, key=lambda c: c["min_x"])
    selected = candidates[0]  # nearest to keyword
    
    # -------------------------------------------------------------------------
    # STEP 4: Cleanup → extract only the number (optional)
    # -------------------------------------------------------------------------
    text = selected["text"]
    return parse_amount(text)