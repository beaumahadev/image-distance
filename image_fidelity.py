import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse
import os
import torch
import lpips
import csv
from datetime import datetime
from PIL import Image


def load_image(image_path):
    """Load an image from file path."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB for consistency
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def calculate_mse(image1, image2):
    """Calculate Mean Squared Error between two images."""
    # Ensure images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Convert to float to avoid overflow
    img1 = image1.astype(np.float64)
    img2 = image2.astype(np.float64)
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    return mse


def calculate_psnr(image1, image2):
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    # Ensure images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Calculate PSNR using scikit-image
    psnr_value = psnr(image1, image2, data_range=255)
    return psnr_value


def calculate_ssim(image1, image2):
    """Calculate Structural Similarity Index between two images."""
    # Ensure images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Convert to grayscale for SSIM calculation
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    else:
        gray1 = image1
        gray2 = image2
    
    # Calculate SSIM
    ssim_value = ssim(gray1, gray2, data_range=255)
    return ssim_value


def calculate_lpips(image1, image2):
    """Calculate Learned Perceptual Image Patch Similarity between two images."""
    # Ensure images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Initialize LPIPS model (using AlexNet backbone)
    lpips_model = lpips.LPIPS(net='alex')
    
    # Convert images to torch tensors
    # Images should be in range [0, 1] and shape [B, C, H, W]
    img1_tensor = torch.from_numpy(image1).float() / 255.0
    img2_tensor = torch.from_numpy(image2).float() / 255.0
    
    # Transpose from HWC to CHW format
    img1_tensor = img1_tensor.permute(2, 0, 1).unsqueeze(0)
    img2_tensor = img2_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(img1_tensor, img2_tensor)
    
    return lpips_value.item()


def save_results_to_csv(results_list, filename=None):
    """Save comparison results to a CSV file inside the 'metrics_output' folder."""
    import os

    output_dir = "metrics_output"
    os.makedirs(output_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_comparison_results.csv"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['input_image', 'output_image', 'mse', 'psnr', 'ssim', 'lpips', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results_list:
            writer.writerow(result)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def resize_output(input_img, output_img):
    """Resize the output image so its smaller dimension matches the input image's larger dimension"""
    # Scale the output image so its smaller dimension matches the input image's larger dimension
    in_h, in_w = input_img.shape[:2]
    out_h, out_w = output_img.shape[:2]
    input_larger_dim = max(in_h, in_w)
    output_smaller_dim = min(out_h, out_w)
    scale = input_larger_dim / output_smaller_dim
    new_out_w = int(round(out_w * scale))
    new_out_h = int(round(out_h * scale))
    output_img = cv2.resize(output_img, (new_out_w, new_out_h), interpolation=cv2.INTER_AREA)
    # Crop to match input image size
    out_h, out_w = output_img.shape[:2]
    crop_top = max((out_h - in_h) // 2, 0)
    crop_left = max((out_w - in_w) // 2, 0)
    output_img = output_img[crop_top:crop_top + in_h, crop_left:crop_left + in_w]     
    return output_img

def resize_output_zero_pad(input_img, output_img):
    """
    Use padding to make the images the same size.
    """
    # Get image shapes
    in_h, in_w = input_img.shape[:2]
    out_h, out_w = output_img.shape[:2]
    diff_h = in_h - out_h
    diff_w = in_w - out_w

    # Decide which dimension to match (the one with the smallest difference)
    if abs(diff_h) <= abs(diff_w):
        # Match heights
        if diff_h > 0:
            # Scale input image height down to match output image height
            scale = out_h / in_h
            new_in_w = int(round(in_w * scale))
            input_img = cv2.resize(input_img, (new_in_w, out_h), interpolation=cv2.INTER_AREA)
        if diff_h < 0: 
            # Scale output image height down to match input image height
            scale = in_h / out_h
            new_out_w = int(round(out_w * scale))
            output_img = cv2.resize(output_img, (new_out_w, in_h), interpolation=cv2.INTER_AREA)
    else:
        # Match widths
        if diff_w > 0:
            # Scale input image width down to match output image width
            scale = out_w / in_w
            new_in_h = int(round(in_h * scale))
            input_img = cv2.resize(input_img, (out_w, new_in_h), interpolation=cv2.INTER_AREA)
        if diff_w < 0: 
            # Scale output image width down to match input image width
            scale = in_w / out_w
            new_out_h = int(round(out_h * scale))
            output_img = cv2.resize(output_img, (in_w, new_out_h), interpolation=cv2.INTER_AREA)

    # Check that at least one image dimension matches after resizing
    in_h, in_w = input_img.shape[:2]
    out_h, out_w = output_img.shape[:2]
    if in_h != out_h and in_w != out_w:
        print(f"Warning: After resizing, neither image dimension matches! input_img: ({in_h}, {in_w}), output_img: ({out_h}, {out_w})")
    # Determine which dimension still doesn't match


    if in_w != out_w:
        # Pad width
        if in_w > out_w:
            # Pad output_img width
            diff = in_w - out_w
            pad_left = diff // 2
            pad_right = diff - pad_left
            output_img = cv2.copyMakeBorder(output_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        if in_w < out_w:
            # Pad input_img width
            diff = out_w - in_w
            pad_left = diff // 2
            pad_right = diff - pad_left
            input_img = cv2.copyMakeBorder(input_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    if in_h != out_h:
        # Pad height
        if in_h > out_h:
            # Pad output_img height
            diff = in_h - out_h
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            output_img = cv2.copyMakeBorder(output_img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        if in_h < out_h:
            # Pad input_img height
            diff = out_h - in_h
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            input_img = cv2.copyMakeBorder(input_img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

    return input_img, output_img

def compare_images(input_image_path, output_image_path, method = "pad"):
    """Compare two images using PSNR, SSIM, and MSE metrics."""
    print("-" * 50)
    print(f"Comparing images:")
    print(f"Input image: {input_image_path}")
    print(f"Output image: {output_image_path}")
    
    # Load images
    try:
        input_img = load_image(input_image_path)
        output_img = load_image(output_image_path)
    except Exception as e:
        print(f"Error loading images: {e}")
        return None
    
    # Handle image dimensions mismatch
    if input_img.shape != output_img.shape:
        if method == "crop":
            output_img = resize_output(input_img, output_img)
        if method == "pad":
            input_img,output_img = resize_output_zero_pad(input_img, output_img)

        # Save the processed images for later visualization in 'crops' folder
        for img, path in zip([input_img, output_img], [input_image_path, output_image_path]):
            crops_dir = os.path.join(os.path.dirname(path), "crops")
            os.makedirs(crops_dir, exist_ok=True)
            ext = os.path.splitext(path)[1] =
            scaled_path = os.path.join(crops_dir, f"{os.path.splitext(os.path.basename(path))[0]}-{method}{ext}")
            cv2.imwrite(scaled_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Saved scaled and cropped image to: {scaled_path}")

    
    # Calculate metrics
    try:
        mse = calculate_mse(input_img, output_img)
        psnr_value = calculate_psnr(input_img, output_img)
        ssim_value = calculate_ssim(input_img, output_img)
        lpips_value = calculate_lpips(input_img, output_img)
        
        # Display results
        print(f"MSE (Mean Squared Error): {mse:.4f}")
        print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr_value:.4f} dB")
        print(f"SSIM (Structural Similarity Index): {ssim_value:.4f}")
        print(f"LPIPS (Learned Perceptual Image Patch Similarity): {lpips_value:.4f}")
        
        return {
            'input_image': input_image_path,
            'output_image': output_image_path,
            'mse': mse,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'lpips': lpips_value,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None


def main():
    """Main function to handle command line arguments and run comparison."""
    import os

    import argparse

    parser = argparse.ArgumentParser(description="Compare two images using PSNR, SSIM, MSE, and LPIPS.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--examples", action="store_true", help="Compare all files in each folder under 'examples'")
    group.add_argument("--images", nargs=2, metavar=('input_image', 'output_image'), help="Paths to the input and output images")
    parser.add_argument("--csv", help="Save results to CSV file (optional filename)")
    args = parser.parse_args()

    if args.examples:
        examples_dir = "examples"
        if not os.path.isdir(examples_dir):
            print(f"Directory '{examples_dir}' not found.")
            exit(1)
        
        all_results = []
        for folder in sorted(os.listdir(examples_dir)):
            folder_path = os.path.join(examples_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            files = [f for f in sorted(os.listdir(folder_path)) if os.path.isfile(os.path.join(folder_path, f))]
            if not files:
                continue
            input_files = [f for f in files if 'input' in f.lower()]
            if not input_files:
                print(f"Input file in '{folder_path}' must contain 'input' in file name.")
                continue
            input_file = input_files[0]
            print(f"\n=== Comparing '{input_file}' to other files in '{folder}' ===")
            print(f"{'Compared File':>20}{'MSE':>15}{'PSNR':>10}{'SSIM':>10}{'LPIPS':>10}")
            for f in files:
                if f == input_file:
                    continue
                img1 = os.path.join(folder_path, input_file)
                img2 = os.path.join(folder_path, f)
                metrics = compare_images(img1, img2)
                if metrics:
                    print(f"{f:>20}{metrics['mse']:>15.2f}{metrics['psnr']:>10.2f}{metrics['ssim']:>10.3f}{metrics['lpips']:>10.3f}")
                    all_results.append(metrics)
                else:
                    print(f"{f:>20}{'ERROR':>15}{'ERROR':>10}{'ERROR':>10}{'ERROR':>10}")
        
        # Save to CSV if requested
        if args.csv or len(all_results) > 0:
            csv_filename = args.csv if args.csv else None
            save_results_to_csv(all_results, csv_filename)
        
        exit(0)


    results = compare_images(args.images[0], args.images[1])
    
    if results:
        print(f"\nComparison completed successfully!")
        # Save to CSV if requested
        if args.csv:
            save_results_to_csv([results], args.csv)
        
    else:
        print(f"\nComparison failed!")
        exit(1)


if __name__ == "__main__":
    main()
