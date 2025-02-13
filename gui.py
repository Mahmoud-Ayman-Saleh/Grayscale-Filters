# gui.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2 as ocv
from image_filters import ImageFilters
import numpy as np

class ImageProcessingGUI:
    def __init__(self):
        self.img = None
        self.modified_image = None  # <-- New attribute to store the modified image
        self.tk_img_original = None
        self.tk_img_modified = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Image Filters GUI")
        
        # Create canvases
        self.canvas_original = tk.Canvas(self.root, width=300, height=300, bg="lightgray", bd=5, relief="raised")
        self.canvas_original.grid(row=0, column=0, padx=10, pady=10)
        
        self.canvas_modified = tk.Canvas(self.root, width=300, height=300, bg="lightgray", bd=5, relief="raised")
        self.canvas_modified.grid(row=0, column=1, padx=10, pady=10)
        
        # Button configuration
        self.button_config = {
            "width": 18,
            "height": 1,
            "bg": "#81C784",
            "fg": "black",
            "font": ("Helvetica", 10, "bold"),
            "relief": "raised",
            "bd": 4
        }
        
        # Create buttons
        self.create_buttons()
    
    def create_buttons(self):
        buttons = [
            ("Load Image", self.load_image),
            ("Original", self.show_original),
            ("Decrease Brightness", self.decrease_brightness),
            ("Increase Brightness", self.increase_brightness),
            ("Negative", self.negative_filter),
            ("Power Law", self.power_law_filter),
            ("Log Filter", self.log_filter),
            ("Inverse Log", self.inverse_log_filter),
            ("Sobel Filter", self.sobel_filter),
            ("Normalize Image", self.normalize_image),
            ("Gaussian Blur", self.gaussian_filter),
            ("Histogram", self.histogram_filter),
            ("Histogram Equalization", self.histogram_equalization),
            ("Match Histogram", self.match_histogram),
            ("Prewitt Filter", self.prewitt_filter),
            ("Average Filter", self.average_filter),
            ("Max Filter", self.max_filter),
            ("Min Filter", self.min_filter),
            ("Median Filter", self.median_filter)
        ]
        
        for i, (text, command) in enumerate(buttons):
            row = (i // 2) + 1
            col = i % 2
            tk.Button(self.root, text=text, command=command, **self.button_config).grid(row=row, column=col, pady=5)
        
        # Add Export Image button below the other buttons
        export_row = (len(buttons) // 2) + 1
        tk.Button(self.root, text="Export Image", command=self.export_image,
                  width=38, height=1, bg="#81C784", fg="black",
                  font=("Helvetica", 10, "bold"), relief="raised", bd=4)\
                  .grid(row=export_row, column=0, columnspan=2, pady=5)
    
    def check_image(self):
        if self.img is None:
            messagebox.showerror("Error", "Please load an image first!")
            return False
        return True
    
    def update_image(self, new_img):
        self.modified_image = new_img  # <-- Store the modified image for export
        img_pil_original = Image.fromarray(self.img)
        img_pil_modified = Image.fromarray(new_img)
        self.tk_img_original = ImageTk.PhotoImage(img_pil_original)
        self.tk_img_modified = ImageTk.PhotoImage(img_pil_modified)
        self.canvas_original.create_image(0, 0, anchor="nw", image=self.tk_img_original)
        self.canvas_modified.create_image(0, 0, anchor="nw", image=self.tk_img_modified)
    
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img_loaded = ocv.imread(file_path, 0)
            if img_loaded is not None:
                self.img = ocv.resize(img_loaded, (300, 300))
                self.update_image(self.img)
            else:
                messagebox.showerror("Error", "Failed to load the image!")
    
    def show_original(self):
        if not self.check_image(): return
        self.update_image(self.img)
    
    def decrease_brightness(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.decrease_brightness(self.img))
    
    def increase_brightness(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.increase_brightness(self.img))
    
    def negative_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.negative_filter(self.img))
    
    def power_law_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.power_law_filter(self.img))
    
    def log_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.log_filter(self.img))
    
    def inverse_log_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.inverse_log_filter(self.img))
    
    def sobel_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.sobel_filter(self.img))
    
    def normalize_image(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.normalize_image(self.img))
    
    def gaussian_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.gaussian_filter(self.img))
    
    def histogram_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.histogram_filter(self.img))
    
    def histogram_equalization(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.histogram_equalization(self.img))
    
    def match_histogram(self):
        if not self.check_image(): return
        file_path = filedialog.askopenfilename()
        if file_path:
            new_image = ocv.imread(file_path, 0)
            if new_image is not None:
                new_image_resized = ocv.resize(new_image, (300, 300))
                matched_image = ImageFilters.histogram_matching(self.img, new_image_resized)
                self.update_image(matched_image)
            else:
                messagebox.showerror("Error", "Failed to load the target image!")
    
    def prewitt_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.prewitt_filter(self.img))
    
    def average_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.average_filter(self.img))
    
    def max_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.max_filter(self.img))
    
    def min_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.min_filter(self.img))
    
    def median_filter(self):
        if not self.check_image(): return
        self.update_image(ImageFilters.median_filter(self.img))
    
    def export_image(self):
        """Exports the currently displayed (modified) image."""
        if self.modified_image is None:
            messagebox.showerror("Error", "No modified image to export!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")]
        )
        if file_path:
            # Ensure the image is in uint8 format before saving.
            img_to_save = self.modified_image
            if img_to_save.dtype != np.uint8:
                # If the image values are normalized between 0 and 1, convert them.
                if img_to_save.max() <= 1.0:
                    img_to_save = (img_to_save * 255).astype(np.uint8)
                else:
                    img_to_save = img_to_save.astype(np.uint8)
            
            success = ocv.imwrite(file_path, img_to_save)
            if success:
                messagebox.showinfo("Image Saved", f"Image saved successfully to:\n{file_path}")
            else:
                messagebox.showerror("Error", "Failed to save the image!")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ImageProcessingGUI()
    app.run()
