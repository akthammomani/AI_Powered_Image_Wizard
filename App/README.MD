# AI-Powered Image Wizard

## Overview
The **AI-Powered Image Wizard** is an interactive web application built using Streamlit that allows users to enhance, transform, and process images with state-of-the-art AI models. It integrates cutting-edge techniques for denoising, super-resolution, and other image adjustments, providing users with an intuitive and powerful tool for image enhancement.

---

## Features
### 1. **Image Denoising**
- **Model:** Denoising Diffusion Probabilistic Model (DDPM)
- **Details:** 
  - Utilizes a fine-tuned UNet model (`UNet2DModel`) and a scheduler (`DDIMScheduler`) to reduce noise in images.
  - Restores image quality by reconstructing noise-free details.
- **Pipeline:** `DDPMPipeline`

### 2. **Image Super-Resolution**
- **Model:** Enhanced Deep Residual Network for Super-Resolution (EDSR)
- **Details:**
  - Fine-tuned EDSR model with 16 residual blocks and a scaling factor of 4.
  - Enhances image resolution while preserving fine details.
- **Implementation:** EDSR loaded via custom configuration.

### 3. **Image Adjustments**
- **Adjustable Parameters:**
  - **Brightness**
  - **Sharpness**
  - **Contrast**
  - **Saturation**
  - **Color Intensity**
  - **Grayscale**
  - **Rotation**
- **Details:**
  - Uses Python’s PIL library (`ImageEnhance`, `ImageOps`) for flexible, real-time adjustments to image properties.

---

## Methods and Models Used

### **A. Denoising Diffusion Probabilistic Model (DDPM)**
- **How It Works:**
  - Predicts and removes noise from images using a reverse diffusion process.
  - Incorporates random noise generation and its iterative removal based on learned priors.
- **Libraries Used:**
  - `torch`
  - `diffusers` (`UNet2DModel`, `DDIMScheduler`, `DDPMPipeline`)

### **B. Super-Resolution with EDSR**
- **How It Works:**
  - Leverages a deep residual network (ResNet) architecture to upscale images by learning high-frequency details.
  - Implements sub-pixel convolution for efficient image reconstruction.
- **Libraries Used:**
  - Custom EDSR implementation loaded via `EDSR-PyTorch/src`.

### **C. Image Processing with PIL**
- **How It Works:**
  - Enhances specific properties of images (brightness, sharpness, etc.) using operations available in PIL.
  - Implements rotation and grayscale conversion.

---

## User Interface

### **Input:**
- **File Upload:** Users can upload images in `.jpg`, `.png`, or `.jpeg` formats.
- **Crop Tool:** Users can crop the image for precise editing using `streamlit_cropper`.

### **Adjustments:**
- **Sliders for Image Properties:** 
  - Easily modify properties like brightness, contrast, saturation, etc.
- **Feature Toggles:**
  - Enable or disable denoising and super-resolution features.

### **Output:**
- Displays both the **Original Image** and the **Processed Image** side by side for comparison.
- Allows users to **download the enhanced image** in `.png` format.

---

## Technologies and Libraries
### **Frontend**
- **Streamlit:** Interactive UI with real-time updates and sliders.
- **Streamlit-Cropper:** Easy cropping of uploaded images.

### **Backend**
- **PyTorch:** Deep learning framework for all models.
- **Diffusers:** For Stable Diffusion pipelines and other image processing tasks.
- **Pillow (PIL):** For basic image transformations.

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Libraries:
  - `streamlit`
  - `torch`
  - `diffusers`
  - `Pillow`
  - `streamlit_cropper`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-image-wizard.git
   cd ai-image-wizard

