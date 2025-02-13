
# Image Filters GUI Project

This project is a Python-based application that provides a graphical user interface (GUI) for applying various filters to grayscale images. Users can easily load an image, apply a variety of filters, and export the modified image.

## Features

- **Load and Display Images:**  
  Load grayscale images and view the original alongside the modified version.

- **Apply Multiple Filters:**  
  Use a variety of filters including:
  - Brightness adjustment (increase/decrease)
  - Negative filter
  - Power law (gamma) filter
  - Logarithmic and inverse logarithmic filters
  - Edge detection filters (Sobel, Prewitt)
  - Smoothing filters (Gaussian blur, Average, Median, Max, Min)
  - Histogram operations (display, equalization, matching)

- **Export Functionality:**  
  Save the modified image to your file system.

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python 3.x**  
- **pip** (Python package installer)

The required Python packages are:

- `numpy`
- `opencv-python`
- `Pillow`

You can install these packages using pip:

```bash
pip install numpy opencv-python pillow
```

> **Note:**  
> Tkinter is used for the GUI and is usually bundled with Python. If it is not installed, refer to your operating system's documentation for installing Tkinter.

## Project Structure

- **`image_filters.py`**  
  Contains the `ImageFilters` class with static methods implementing various image filters.

- **`gui.py`**  
  Implements the Tkinter-based GUI, providing buttons to load images, apply filters, and export the filtered image.

- **`README.md`**  
  Provides an overview of the project, instructions, and prerequisites.

## How to Run

1. **Clone or Download the Repository:**  
   Ensure all project files (`gui.py`, `image_filters.py`, `README.md`) are in the same directory.

2. **Install Dependencies:**  
   Open a terminal in the project directory and run:
   ```bash
   pip install numpy opencv-python pillow
   ```

3. **Run the Application:**  
   Execute the following command in your terminal:
   ```bash
   python gui.py
   ```

4. **Using the GUI:**  
   - Click **Load Image** to select a grayscale image.
   - Use the various filter buttons to apply your desired effect.
   - Click **Export Image** to save the modified image to your desired location.

## Build Instructions

This project is written in Python and does not require a complex build process. Simply install the necessary dependencies and run `gui.py` as shown above.
