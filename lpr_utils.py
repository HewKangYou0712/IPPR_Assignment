# lpr_utils.py
import cv2
import pytesseract
import numpy as np
import easyocr
import re
from collections import Counter
import difflib

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False) 

# Set Tesseract command path (Windows specific)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(img):
    """Applies basic preprocessing for edge detection (grayscale, blur, Canny)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)
    return edged

def rotate_image(image, angle):
    """Rotates an image by a given angle around its center."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the affine transformation (rotation)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def pick_best_plate(candidates):
    # Step 1: Prioritize valid formats
    valid = [c for c in candidates if c['valid']]
    if valid:
        # Choose the most frequent valid cleaned text
        texts = [c['clean'] for c in valid]
        common = Counter(texts).most_common(1)[0][0]
        best = next(c for c in valid if c['clean'] == common)
        print(f"\nFinal Pick (Valid Match): {best['clean']}")
        return best['image'], best['bbox']

    # Step 2: Fuzzy match top similar partials
    all_cleaned = [c['clean'] for c in candidates if len(c['clean']) >= 5]
    best_guess = ""
    score = 0

    for base in all_cleaned:
        matches = [difflib.SequenceMatcher(None, base, other).ratio() for other in all_cleaned]
        avg_score = sum(matches) / len(matches)
        if avg_score > score:
            score = avg_score
            best_guess = base

    if best_guess:
        best = next(c for c in candidates if c['clean'] == best_guess)
        print(f"\nFinal Pick (Fuzzy Match): {best_guess} (score: {score:.2f})")
        return best['image'], best['bbox']

    print("\nNo reliable plate found.")
    return None, None

def detect_plate(image):
    """
    Detects a license plate in the given image by trying various rotations.
    Uses edge detection, contour finding, aspect ratio, and a format validator.
    """
    # Define a range of angles to try for robustness against tilted plates
    angles = [0, -15, 15, -30, 30, -45, 45, -60, 60, -75, 75, -90, 90, -135, 135, 180] # Expanded angles for better detection
    candidates = []

    for angle in angles:
        rotated = rotate_image(image, angle)
        gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blur, 30, 200)

        # Find contours in the edged image
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort contours by area and keep top 30 (potential plate candidates)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        for cnt in contours:
            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            
            if len(approx) == 4: # Look for 4-sided shapes (rectangles)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h) # Calculate aspect ratio HERE

                # Check if the aspect ratio falls within typical license plate ranges
                if 0.3 <= aspect_ratio <= 5.0: # Broadened aspect ratio range
                    candidate = gray[y:y + h, x:x + w] # Crop the potential plate region
                    
                    # Attempt to recognize text on the candidate plate
                    print(f"\nTrying angle {angle} with aspect ratio {aspect_ratio:.2f}")
                    text = recognize_text(candidate)
                    cleaned = text.strip().upper().replace("\n", " ")
                    is_valid = is_valid_plate_format(cleaned)

                    candidates.append({
                        'angle': angle,
                        'text': text,
                        'clean': cleaned,
                        'valid': is_valid,
                        'image': candidate,
                        'bbox': (x, y, w, h)
                    })
    if candidates:
        return pick_best_plate(candidates)

    return None, None # No plate found after trying all angles and contours

def is_valid_plate_format(plate_text):
    """
    Validates if the extracted text matches common Malaysian license plate formats
    using regular expressions.
    """
    plate_text = plate_text.strip().upper()

    # Common Malaysian license plate formats (can be expanded for more specificity)
    patterns = [
        r"^[A-Z]{1,3}\d{1,4}[A-Z]{0,2}$",      # e.g. W1234A, JKL1234, ABC123 (single line)
        r"^[A-Z]{1,3}\d{1,4}$",                # e.g. B1234, JKL123 (single line)
        r"^[A-Z]{1,2}\d{1,4}[A-Z]{1,2}$",      # e.g. WA1234B (single line)
        r"^[A-Z]{1,2}\d{1,4}$",                # e.g. WP1234 (single line)
        r"^\d{1,4}[A-Z]{1,3}$",                # e.g. 1234JK (less common, single line)
        r"^[A-Z]{1,3}\s*\d{1,4}$",             # e.g. ABC 1234 (with space, common for two-row concatenation)
        r"^[A-Z]{1,2}\s*\d{1,4}[A-Z]{0,2}$",   # e.g. WA 1234B (with space)
        r"^[A-Z]{1,3}\s*\d{1,4}[A-Z]{0,2}$",
        r"^[A-Z]{1,3}\n\s*\d{1,4}[A-Z]{0,2}$",
        r"^[A-Z]{1,3}\n\s*\d{1,4}$",
        r"^[A-Z]{1,2}\n\s*\d{1,4}[A-Z]{1,2}$",
        r"^[A-Z]{1,2}\n\s*\d{1,4}$"
    ]

    for pattern in patterns:
        if re.fullmatch(pattern, plate_text):
            return True
    return False

def recognize_text(plate_img):
    """
    Recognizes text from a license plate image using both EasyOCR and Tesseract.
    Chooses the one that best matches known plate formats.
    """
    if plate_img is None:
        return ""

    # -------- EasyOCR --------
    easyocr_results = reader.readtext(plate_img, detail=1)
    easyocr_text = " ".join([text for (_, text, conf) in easyocr_results if conf > 0.4])
    easyocr_text_clean = re.sub(r'[^A-Za-z0-9\n ]', '', easyocr_text.strip().upper())
    easyocr_text_clean = easyocr_text_clean.replace("\n", " ")

    # -------- Tesseract --------
    tesseract_text_raw = pytesseract.image_to_string(plate_img, config='--psm 7')
    tesseract_text_clean = re.sub(r'[^A-Za-z0-9\n ]', '', tesseract_text_raw.strip().upper())
    tesseract_text_clean = tesseract_text_clean.replace("\n", " ")

    # -------- Format Check --------
    easyocr_valid = is_valid_plate_format(easyocr_text_clean)
    tesseract_valid = is_valid_plate_format(tesseract_text_clean)

    print(f"EasyOCR: {easyocr_text_clean} \nTesseract: {tesseract_text_clean}")

    # -------- Decision Logic --------
    if easyocr_valid and not tesseract_valid:
        return easyocr_text_clean
    elif tesseract_valid and not easyocr_valid:
        return tesseract_text_clean
    elif easyocr_valid and tesseract_valid:
        # Prefer EasyOCR if both are valid (optional)
        return easyocr_text_clean
    else:
        # Fall back to longer string
        return easyocr_text_clean if len(easyocr_text_clean) > len(tesseract_text_clean) else tesseract_text_clean

def identify_state(text):
    """
    Identifies the registered state based on the initial characters of the license plate text.
    This mapping is specific to Malaysian license plate prefixes.
    """
    state_prefixes = {
        "W": "Kuala Lumpur", "J": "Johor", "P": "Penang", "B": "Selangor", "A": "Perak",
        "K": "Kedah", "T": "Terengganu", "C": "Pahang", "D": "Kelantan", "M": "Melaka",
        "N": "Negeri Sembilan", "S": "Sabah", "Q": "Sarawak", "R": "Perlis",
        "V": "Kuala Lumpur", "KV": "Langkawi", "L": "Labuan", "F": "Putrajaya", "Z": "Military"
        # For diplomatic: 'CD' prefix, or specific ambassadorial codes
        # For special series (e.g., 'MALAYSIA', 'PATRIOT', 'G1M'): these need specific handling
    }
    if not text:
        return "Not Detected"
    
    # Prioritize checking for longer prefixes (e.g., "KV", "CD") before shorter ones ("K", "C")
    # This is crucial for correct identification.
    if len(text) >= 2 and text[:2].upper() in state_prefixes:
        return state_prefixes[text[:2].upper()]
    elif text[0].upper() in state_prefixes:
        return state_prefixes[text[0].upper()]
    
    return "Unknown or Special Series" # Default for unmatched prefixes
