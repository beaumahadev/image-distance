# Image Distance Comparison Tool

This tool compares two images using three different metrics: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and MSE (Mean Squared Error).

## Setup

1. Activate the virtual environment:
```bash
source venv/bin/activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the comparison tool with two image paths:

```bash
python image_metrics.py input_image.jpg output_image.jpg
```

### Example:
```bash
python image_metrics.py original.png compressed.jpg
```

## Metrics Explained

- **MSE (Mean Squared Error)**: Measures the average squared difference between pixel values. Lower values indicate better similarity (0 = identical).

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the ratio between the maximum possible power of a signal and the power of corrupting noise. Higher values indicate better quality (>30 dB is generally considered good).

- **SSIM (Structural Similarity Index)**: Measures the structural similarity between images, considering luminance, contrast, and structure. Values range from -1 to 1, where 1 = identical.

## Features

- Automatic image resizing if dimensions don't match
- Support for common image formats (JPEG, PNG, etc.)
- Detailed interpretation of results
- Error handling for invalid files or formats
