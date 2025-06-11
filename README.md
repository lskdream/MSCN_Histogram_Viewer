# MSCN Histogram Viewer

This tool computes and visualizes the **MSCN (Mean Subtracted Contrast Normalized)** coefficients from an input grayscale image. MSCN is commonly used in no-reference image quality assessment models like BRISQUE.

---

## 🔍 What is MSCN?

MSCN coefficients are calculated as:
MSCN(x, y) = (I(x, y) - μ(x, y)) / (σ(x, y) + ε)


Where:
- `μ(x, y)` is the local mean (using a Gaussian kernel)
- `σ(x, y)` is the local standard deviation
- `ε` is a small constant for numerical stability

They normalize an image’s local structure and are often modeled as Gaussian-like distributions.

---

## 🚀 How to Use

### 📦 Installation

```bash
pip install numpy scipy opencv-python matplotlib

▶️ Run from CLI
python calc_mscn.py --img path/to/image.jpg
This will:
Read the image
Compute MSCN coefficients
Plot and display the histogram using matplotlib

📊 Output
The histogram shows the distribution of MSCN values, typically centered around zero and symmetric for natural images.

🧪 Notes
Input image should be grayscale; if not, it's converted automatically.
You can modify the bins count in the plot_histogram() function to change histogram resolution.

✍️ Author
Tool refactored by Yu-Chih Chen, based on BRISQUE MSCN logic.
