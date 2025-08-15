import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ===== Global State =====
uploaded_image_path = None  # Stores uploaded image path

# ===== Colors & Fonts =====
BG_COLOR = "#1e1e2f"
SIDEBAR_COLOR = "#252538"
ACCENT_COLOR = "#0078D7"
TEXT_COLOR = "#ffffff"
CONTENT_BG = "#f4f4f4"

# ===== Main Window =====
root = tk.Tk()
root.title("LPR & SIS - License Plate Recognition & State Identification")
root.geometry("1100x700")
root.configure(bg=BG_COLOR)

# ===== Sidebar =====
sidebar = tk.Frame(root, bg=SIDEBAR_COLOR, width=180)
sidebar.pack(side="left", fill="y")

tk.Label(
    sidebar,
    text="📌 Menu",
    font=("Segoe UI", 14, "bold"),
    bg=SIDEBAR_COLOR,
    fg=TEXT_COLOR,
    pady=15
).pack()

# ===== Main Content Area =====
content_frame = tk.Frame(root, bg=CONTENT_BG)
content_frame.pack(side="right", expand=True, fill="both")

# ===== Single Image Display Frame =====
single_frame = tk.Frame(root, bg=CONTENT_BG)
single_img_label = tk.Label(single_frame, bg="white", relief="ridge")
single_img_label.pack(padx=20, pady=20)

def show_single_image(image_pil):
    """Show one large image in single_frame"""
    # Hide grid frame
    canvas.itemconfig(canvas_window, window=single_frame)
    img_tk = ImageTk.PhotoImage(image_pil)
    single_img_label.configure(image=img_tk)
    single_img_label.image = img_tk

def show_grid():
    """Switch to the 2x2 grid view"""
    canvas.itemconfig(canvas_window, window=grid_frame)

header = tk.Label(
    content_frame,
    text="📷 Stage 1: Upload Image",
    font=("Segoe UI", 18, "bold"),
    bg=ACCENT_COLOR,
    fg="white",
    pady=10
)
header.pack(fill="x")

# ===== Scrollable Image Display Area =====
canvas = tk.Canvas(content_frame, bg=CONTENT_BG, highlightthickness=0)
scrollbar = tk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")
canvas.configure(yscrollcommand=scrollbar.set)

# Frame inside the canvas
grid_frame = tk.Frame(canvas, bg=CONTENT_BG)
canvas_window = canvas.create_window(canvas.winfo_width() / 2, 0, window=single_frame, anchor="n")

def update_scrollregion(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

def center_grid(event):
    canvas.coords(canvas_window, event.width / 2, 0)

grid_frame.bind("<Configure>", update_scrollregion)
canvas.bind("<Configure>", center_grid)

# ===== Stage Image Labels =====
stage_labels = {}
caption_labels = {}
stages = ["Original", "Grayscale", "Blurred", "Edges"]

# Create a 2×2 grid of images
for idx, stage in enumerate(stages):
    row = idx // 2
    col = idx % 2

    img_label = tk.Label(grid_frame, bg="white", relief="ridge")
    img_label.grid(row=row * 2, column=col, padx=20, pady=10, ipadx=10, ipady=10)
    stage_labels[stage] = img_label

    caption = tk.Label(
        grid_frame,
        text=stage,
        font=("Segoe UI", 12, "bold"),
        bg=CONTENT_BG
    )
    caption.grid(row=row * 2 + 1, column=col, pady=(0, 20))
    caption_labels[stage] = caption

# Center columns equally
for col in range(2):
    grid_frame.grid_columnconfigure(col, weight=1)

# ===== Single Image Display Frame =====
single_frame = tk.Frame(root, bg=CONTENT_BG)

single_img_label = tk.Label(single_frame, bg="white", relief="ridge")
single_img_label.pack(padx=20, pady=(20, 5))

# ===== Status Label =====
status_label = tk.Label(
    single_frame,
    font=("Segoe UI", 10),
    fg="gray",
    bg=CONTENT_BG
)
status_label.pack(pady=(0, 20))

# ===== Upload Image =====
def upload_image():
    global uploaded_image_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not file_path:
        return

    uploaded_image_path = file_path
    img = Image.open(file_path)
    img.thumbnail((500, 350))
    show_single_image(img)
    img_tk = ImageTk.PhotoImage(img)
    stage_labels["Original"].configure(image=img_tk)
    stage_labels["Original"].image = img_tk

    # Clear other stages
    for stage in ["Grayscale", "Blurred", "Edges"]:
        stage_labels[stage].config(image='', text='')

    filename = file_path.split("/")[-1]
    status_label.config(text=f"Uploaded: {filename}", fg="green")
    status_label.pack(pady=(0, 20))
    header.config(text="📷 Stage 1: Upload Image")

# ===== Preprocess Image =====
def preprocess_image():
    global uploaded_image_path
    if not uploaded_image_path:
        messagebox.showerror("Error", "No image uploaded! Please upload an image first.")
        return

    header.config(text="⚙️ Stage 2: Preprocessing Image...")
    show_grid()

    # Load & preprocess
    img_cv = cv2.imread(uploaded_image_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    processed_images = {
        "Original": img_cv,
        "Grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        "Blurred": cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB),
        "Edges": cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    }

    for stage, img_array in processed_images.items():
        img_pil = Image.fromarray(img_array)
        img_pil.thumbnail((400, 300))
        img_tk = ImageTk.PhotoImage(img_pil)
        stage_labels[stage].configure(image=img_tk)
        stage_labels[stage].image = img_tk

# ===== Globals =====
handles = []
rect_id = None
selected_rect = None
resize_dir = None
dragging = False
auto_detected_coords = None
original_img = None  # full RGB image

def detect_plate():
    global uploaded_image_path, detected_plate, auto_detected_coords, original_img

    if not uploaded_image_path:
        messagebox.showerror("Error", "No image uploaded! Please upload an image first.")
        return

    header.config(text="🔍 Stage 3: Detecting License Plate...")

    # Load and store original image
    img_rgb = cv2.imread(uploaded_image_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    original_img = img_rgb.copy()

    # Contrast enhancement
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # Noise reduction
    blur = cv2.bilateralFilter(gray_enhanced, 11, 17, 17)

    # Morphology + edges
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel_rect)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_rect)
    edges = cv2.Canny(morph, 30, 150)

    # Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    debug_img = img_rgb.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h != 0 else 0
        area = w * h
        if 1.0 < aspect_ratio < 7.0 and area > 100:
            candidates.append((x, y, w, h))

    if not candidates:
        messagebox.showwarning("Warning", "No license plate detected!")

    # Pick largest candidate
    x, y, w, h = max(candidates, key=lambda c: c[2] * c[3])
    auto_detected_coords = (x, y, w, h)
    detected_plate = img_rgb[y:y+h, x:x+w].copy()

    # Draw green box on detected area
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # Display in 1×2 grid
    detect_frame = tk.Frame(canvas, bg=CONTENT_BG)
    canvas.itemconfig(canvas_window, window=detect_frame)

    # Left: original with detection
    left_label = tk.Label(detect_frame, bg="white", relief="ridge")
    left_label.grid(row=0, column=0, padx=20, pady=20)
    img_pil = Image.fromarray(debug_img)
    img_pil.thumbnail((400, 300))
    img_tk = ImageTk.PhotoImage(img_pil)
    left_label.configure(image=img_tk)
    left_label.image = img_tk

    # Right: cropped plate
    global right_label
    right_label = tk.Label(detect_frame, bg="white", relief="ridge")
    right_label.grid(row=0, column=1, padx=20, pady=20)
    plate_pil = Image.fromarray(detected_plate)
    plate_pil.thumbnail((400, 300))
    plate_tk = ImageTk.PhotoImage(plate_pil)
    right_label.configure(image=plate_tk)
    right_label.image = plate_tk

    # Captions
    tk.Label(detect_frame, text="Detected in Image", bg=CONTENT_BG, font=("Segoe UI", 12, "bold")).grid(row=1, column=0)
    tk.Label(detect_frame, text="Cropped Plate", bg=CONTENT_BG, font=("Segoe UI", 12, "bold")).grid(row=1, column=1)

    # Manual select + confirm buttons
    manual_btn = tk.Button(detect_frame, text="Manual Select", command=enable_manual_selection)
    manual_btn.grid(row=2, column=0, pady=10)

    ocr_btn = tk.Button(detect_frame, text="Run OCR", command=run_ocr_on_plate, bg="#2196F3", fg="white", font=("Segoe UI", 12, "bold"))
    ocr_btn.grid(row=3, column=0, columnspan=2, pady=20)

    global confirm_btn
    confirm_btn = tk.Button(detect_frame, text="Confirm Selection", command=confirm_manual_crop)
    confirm_btn.grid(row=2, column=1, pady=10)
    confirm_btn.grid_remove()  # Hide until manual mode is enabled

# ===== Manual selection helpers =====
handle_size = 6

def enable_manual_selection():
    if original_img is None:
        messagebox.showerror("Error", "No image loaded!")
        return

    popup = tk.Toplevel()
    popup.title("Manual License Plate Selection")
    popup.configure(bg="white")

    # Resize image to fit popup
    img_pil = Image.fromarray(original_img)
    display_w, display_h = 800, 600
    img_pil.thumbnail((display_w, display_h))
    img_tk = ImageTk.PhotoImage(img_pil)

    canvas_popup = tk.Canvas(popup, width=img_tk.width(), height=img_tk.height(), bg="white")
    canvas_popup.pack(padx=10, pady=10)
    canvas_popup.create_image(0, 0, anchor="nw", image=img_tk)
    canvas_popup.image = img_tk

    # Variables for rectangle selection
    selection = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "rect": None}

    def on_mouse_down(event):
        selection["x1"], selection["y1"] = event.x, event.y
        if selection["rect"]:
            canvas_popup.delete(selection["rect"])
        selection["rect"] = canvas_popup.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

    def on_mouse_drag(event):
        selection["x2"], selection["y2"] = event.x, event.y
        canvas_popup.coords(selection["rect"], selection["x1"], selection["y1"], event.x, event.y)

    def on_confirm():
        global detected_plate
        if not selection["rect"]:
            messagebox.showerror("Error", "No area selected!")
            return

        # Map coordinates to original image size
        scale_x = original_img.shape[1] / img_tk.width()
        scale_y = original_img.shape[0] / img_tk.height()
        x1 = int(selection["x1"] * scale_x)
        y1 = int(selection["y1"] * scale_y)
        x2 = int(selection["x2"] * scale_x)
        y2 = int(selection["y2"] * scale_y)

        detected_plate = original_img[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)].copy()

        # Update the Cropped Plate preview in main detect window
        plate_pil = Image.fromarray(detected_plate)
        plate_pil.thumbnail((400, 300))
        plate_tk = ImageTk.PhotoImage(plate_pil)
        right_label.configure(image=plate_tk)
        right_label.image = plate_tk

        messagebox.showinfo("Success", "Manual selection saved for OCR!")
        popup.destroy()

    def on_cancel():
        popup.destroy()

    # Bind events
    canvas_popup.bind("<Button-1>", on_mouse_down)
    canvas_popup.bind("<B1-Motion>", on_mouse_drag)

    # Buttons
    btn_frame = tk.Frame(popup, bg="white")
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Confirm", command=on_confirm, bg="#4CAF50", fg="white").pack(side="left", padx=10)
    tk.Button(btn_frame, text="Cancel", command=on_cancel, bg="#f44336", fg="white").pack(side="left", padx=10)

    popup.transient()
    popup.grab_set()

def confirm_manual_crop():
    global detected_plate
    x1, y1, x2, y2 = selected_rect
    detected_plate = original_img[int(y1):int(y2), int(x1):int(x2)].copy()
    messagebox.showinfo("Success", "Manual selection saved for OCR!")

# helper: postprocess common confusions given an expected letter/digit split
def _postprocess_plate_text(raw):
    s = re.sub(r'[^A-Z0-9]', '', raw.upper())
    if not s:
        return ''

    # heuristic: split into letters then digits by finding first digit
    m = re.search(r'\d', s)
    if m:
        split = m.start()
        letters = list(s[:split])
        digits = list(s[split:])
        # maps to convert obvious confusions
        to_letter = {'0':'O','1':'I','5':'S','2':'Z','8':'B','6':'G','4':'A','7':'T'}
        to_digit  = {'O':'0','D':'0','Q':'0','I':'1','L':'1','Z':'2','S':'5','B':'8','G':'6','T':'7'}
        letters = [to_letter.get(ch,ch) if ch.isdigit() else ch for ch in letters]
        digits  = [to_digit.get(ch,ch) if ch.isalpha() else ch for ch in digits]
        return ''.join(letters) + ' ' + ''.join(digits)
    else:
        # no digits found — just return cleaned string
        return s
        
def run_ocr_on_plate():
    global detected_plate
    if detected_plate is None:
        messagebox.showerror("Error", "No plate image available for OCR!")
        return ''

    # ---- 1) Ensure correct color -> grayscale ----
    gray = cv2.cvtColor(detected_plate, cv2.COLOR_RGB2GRAY)

    # ---- 2) Upscale to improve Tesseract (keep aspect ratio) ----
    h, w = gray.shape
    target_h = 120  # tune: make characters ~30-50 px high if possible
    if h < target_h:
        scale = target_h / float(h)
        gray = cv2.resize(gray, (int(w*scale), target_h), interpolation=cv2.INTER_CUBIC)

    # ---- 3) Local contrast & denoise ----
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # ---- 4) Slight sharpening (unsharp mask) ----
    gauss = cv2.GaussianBlur(gray, (0,0), 3)
    gray = cv2.addWeighted(gray, 1.4, gauss, -0.4, 0)

    # ---- 5) Threshold (Otsu) and morphological close ----
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure dark text on light background for Tesseract
    if np.mean(th) < 127:  # image mostly dark — invert
        th = cv2.bitwise_not(th)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ---- 6) OCR with multi-line mode ----
    pil_img = Image.fromarray(th)
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    raw_text = pytesseract.image_to_string(pil_img, config=custom_config)

    # ---- 7) Clean multi-line result ----
    lines = [re.sub(r'[^A-Z0-9]', '', l.upper()) for l in raw_text.splitlines() if l.strip()]
    cleaned = ' '.join(lines)  # join rows with a space
    cleaned = _postprocess_plate_text(cleaned)
    state = identify_state(cleaned)

    if cleaned:
        messagebox.showinfo("OCR Result", f"Detected Plate: {cleaned}\nRegistered State: {state}")
    else:
        messagebox.showwarning("OCR Result", "No valid plate characters detected.")

    return cleaned

def identify_state(text):
    # State mapping for Malaysian license plates
    state_prefixes = {
        "W": "Kuala Lumpur", "J": "Johor", "P": "Penang", "B": "Selangor", "A": "Perak",
        "K": "Kedah", "T": "Terengganu", "C": "Pahang", "D": "Kelantan", "M": "Melaka",
        "N": "Negeri Sembilan", "S": "Sabah", "Q": "Sarawak", "R": "Perlis",
        "V": "Kuala Lumpur", "KV": "Langkawi", "L": "Labuan", "F": "Putrajaya", "Z": "Military"
    }

    if not text:
        return "Not Detected"

    # Remove spaces & make uppercase for consistent matching
    cleaned = text.replace(" ", "").upper()

    # Correct common OCR misreads in the prefix only
    prefix_corrections = {"8": "B", "0": "O", "1": "I", "5": "S", "2": "Z"}
    cleaned = ''.join(prefix_corrections.get(ch, ch) for ch in cleaned)

    # Try first two characters, then first one
    if len(cleaned) >= 2 and cleaned[:2] in state_prefixes:
        return state_prefixes[cleaned[:2]]
    elif cleaned[0] in state_prefixes:
        return state_prefixes[cleaned[0]]

    return "Unknown or Special Series"

# ===== Sidebar Buttons =====
def add_sidebar_button(text, command=None):
    tk.Button(
        sidebar,
        text=text,
        font=("Segoe UI", 12),
        bg=ACCENT_COLOR,
        fg="white",
        relief="flat",
        activebackground="#005A9E",
        activeforeground="white",
        command=command
    ).pack(fill="x", padx=15, pady=5, ipady=5)

add_sidebar_button("📂 Upload Image", upload_image)
add_sidebar_button("⚙️ Preprocess", preprocess_image)
add_sidebar_button("🔍 Detect Plate", detect_plate)
add_sidebar_button("❌ Exit", root.destroy)

root.mainloop()
