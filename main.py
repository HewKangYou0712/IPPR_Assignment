# main.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

# Import functions from your lpr_utils.py
from lpr_utils import detect_plate, recognize_text, identify_state, preprocess_image # Import preprocess_image

class LPR_GUI:
    def __init__(self, root):
        self.auto_detect = False # This variable's usage seems incomplete, consider its purpose.
        self.root = root
        self.root.title("License Plate Recognition System")
        self.root.geometry("800x800")
        self.panel = None # Panel to display the main image (original or processed)

        # Label to display results (detected text and state)
        self.label_result = tk.Label(root, text="", font=("Arial", 14))
        self.label_result.pack(pady=10)

        # Button to load an image from file
        self.button_load = tk.Button(root, text="Load Image", command=self.load_image)
        self.button_load.pack()

        # Button to process the loaded image (detect plate), initially disabled
        self.button_process = tk.Button(root, text="Detect Plate", command=self.process_image, state=tk.DISABLED)
        self.button_process.pack(pady=5)

        # Button to preview edges and original image in a new window
        self.button_preview = tk.Button(root, text="Preview Preprocessing", command=self.preview_preprocessing)
        self.button_preview.pack(pady=5)

    def load_image(self):
        """Opens a file dialog to load an image and displays it in the main panel."""
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if path:
            self.img_path = path
            self.image = cv2.imread(path) # Load the image using OpenCV
            if self.image is None: # Handle cases where image loading fails
                messagebox.showerror("Error", "Failed to load image. Please select a valid image file.")
                return
            self.display_image(self.image) # Display the original image
            self.label_result.config(text="") # Clear any previous results
            self.button_process.config(state=tk.NORMAL) # Enable the process button

    def display_image(self, img, target_panel=None):
        """
        Converts an OpenCV image (BGR) to a Tkinter PhotoImage and displays it.
        Can display to a specified panel or the main panel.
        """
        # Ensure image is in BGR format before converting to RGB for PIL
        if len(img.shape) == 2: # If grayscale, convert to BGR
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else: # Otherwise, assume BGR and convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img_pil = Image.fromarray(img_rgb)
        # Resize for display, maintaining aspect ratio if needed (here, fixed size)
        img_pil = img_pil.resize((600, 600), Image.LANCZOS) # Use LANCZOS for better quality resizing
        img_tk = ImageTk.PhotoImage(img_pil)

        panel_to_use = target_panel if target_panel else self.panel

        if panel_to_use:
            panel_to_use.config(image=img_tk)
            panel_to_use.image = img_tk # Keep a reference to prevent garbage collection
        else:
            # If no target_panel specified and self.panel is None, create the main panel
            self.panel = tk.Label(image=img_tk)
            self.panel.image = img_tk
            self.panel.pack(pady=10)

    def process_image(self):
        """Initiates license plate detection, text recognition, and state identification."""
        if not hasattr(self, 'image') or self.image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        # Create a copy of the original image to draw on for display
        display_img_copy = self.image.copy() 

        # Call the plate detection function from lpr_utils
        plate, bbox = detect_plate(display_img_copy) 

        if plate is not None:
            x, y, w, h = bbox
            # Draw a green rectangle around the detected plate on the copy
            cv2.rectangle(display_img_copy, (x, y), (x+w, y+h), (0,255,0), 2)
            self.display_image(display_img_copy) # Display the image with the bounding box

            # Recognize text from the cropped plate image
            text = recognize_text(plate)
            # Identify the state based on the recognized text
            state = identify_state(text)

            self.label_result.config(text=f"Detected Text: {text}\nRegistered State: {state}")
            self.auto_detect = False # Reset auto_detect (its full purpose needs clarification)
        else:
            self.label_result.config(text="No License Plate Detected. Please try another image or adjust conditions.")
            # The auto_detect retry logic seems incomplete or intended for a different use case (e.g., video stream)
            # if self.auto_detect:
            #     self.root.after(500, self.process_image)

    def preview_preprocessing(self):
        """
        Displays the original image and its pre-processed (edged) version
        in a new Toplevel window.
        """
        if not hasattr(self, 'image') or self.image is None:
            messagebox.showwarning("No Image", "Please load an image first to preview preprocessing.")
            return

        # Get the pre-processed (edged) image from lpr_utils
        edged_image = preprocess_image(self.image)
        
        # Convert the grayscale edged image to BGR so it can be displayed by display_image
        edged_image_bgr = cv2.cvtColor(edged_image, cv2.COLOR_GRAY2BGR)

        # Show both images in a new window
        self.show_two_images(self.image, "Original Image", edged_image_bgr, "Pre-processed (Edged) Image")

    def show_two_images(self, img1, title1, img2, title2):
        """
        Creates a new Toplevel window and displays two images side-by-side.
        """
        top = tk.Toplevel(self.root)
        top.title("Preprocessing Preview")
        top.geometry("1250x500") # Adjust size to accommodate two images

        # Frame for the first image
        frame1 = tk.Frame(top)
        frame1.pack(side=tk.LEFT, padx=10, pady=10)
        label1_title = tk.Label(frame1, text=title1, font=("Arial", 12, "bold"))
        label1_title.pack()
        panel1 = tk.Label(frame1)
        panel1.pack()
        self.display_image(img1, target_panel=panel1) # Display img1 in panel1

        # Frame for the second image
        frame2 = tk.Frame(top)
        frame2.pack(side=tk.RIGHT, padx=10, pady=10)
        label2_title = tk.Label(frame2, text=title2, font=("Arial", 12, "bold"))
        label2_title.pack()
        panel2 = tk.Label(frame2)
        panel2.pack()
        self.display_image(img2, target_panel=panel2) # Display img2 in panel2

# Main execution block
if __name__ == "__main__":
    root = tk.Tk()
    app = LPR_GUI(root)
    root.mainloop()

#py -3.13 main.py