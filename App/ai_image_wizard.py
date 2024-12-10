import io
import sys
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont
import numpy as np
from streamlit_cropper import st_cropper
import streamlit as st
import torch
from diffusers import DDPMPipeline, UNet2DModel, DDIMScheduler
from torchvision.transforms import Resize, ToTensor, Normalize
import torch.nn.functional as F
from argparse import Namespace

# Add the EDSR module path
sys.path.append("./EDSR-PyTorch/src")  # Replace with the actual path to "src"
from model.edsr import EDSR

icon = Image.open("cv_lgo.jpg")
st.set_page_config(layout='wide', page_title='AI-Powered Image Wizard', page_icon=icon)

# Function to apply adjustments
# Function to apply adjustments
def adjust_image(image, contrast, saturation, rotation, grayscale, brightness, sharpness, color_intensity, apply_denoising=False, apply_super_resolution=False):
    # Apply Denoising
    if apply_denoising:
        image = denoise_image(models["denoising"], image)

    # Apply Super-Resolution
    if apply_super_resolution:
        image = super_resolve_image(models["super_resolution"], image)

    # Apply brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    # Apply sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)

    # Apply contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    # Apply saturation
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation)

    # Apply color intensity (stronger or lighter colors)
    if color_intensity != 1.0:
        hsv_image = np.array(image.convert("HSV"))
        hsv_image[..., 1] = np.clip(hsv_image[..., 1] * color_intensity, 0, 255)  # Adjust the S channel (saturation)
        image = Image.fromarray(hsv_image, "HSV").convert("RGB")

    # Apply rotation
    if rotation != 0:
        image = image.rotate(rotation, expand=True)

    # Convert to grayscale if selected
    if grayscale:
        image = ImageOps.grayscale(image)

    return image

# Function to resize images to a consistent width
def resize_image(image, target_width=400):
    aspect_ratio = image.height / image.width
    target_height = int(target_width * aspect_ratio)
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

# Define the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned model once globally
@st.cache_resource
def load_fine_tuned_model(unet_path, scheduler_path, device=device):
    # Load the UNet
    unet = UNet2DModel.from_pretrained(unet_path)

    # Load the scheduler
    scheduler = DDIMScheduler.from_pretrained(scheduler_path)

    # Create the pipeline
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler)
    pipeline.to(device)
    return pipeline

# Load Super-Resolution Model
@st.cache_resource
def load_super_resolution_model():
    args = Namespace(
        scale=[4],
        n_resblocks=16,
        n_feats=64,
        res_scale=1.0,
        rgb_range=255,
        n_colors=3,
        precision='single'
    )
    model = EDSR(args).to('cpu')
    model.load_state_dict(torch.load('edsr_finetuned_x4.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

# Paths to the saved model and scheduler
unet_path = "./fine_tuned_ddpm_unet_v3"  
scheduler_path = "./fine_tuned_ddpm_scheduler_v3"

# Load the model
#model = load_fine_tuned_model(unet_path, scheduler_path, device)
#print("Fine-tuned DDPM model loaded successfully!")


# Load both models
@st.cache_resource
def load_models():
    return {
        "denoising": load_fine_tuned_model(unet_path, scheduler_path),
        "super_resolution": load_super_resolution_model()
    }

models = load_models()

# Function to apply denoising
def denoise_image(model, image, target_size=(256, 256)):
    """
    Applies the denoising model to the input image.

    Args:
        model: Fine-tuned denoising model.
        image: Input PIL image.
        target_size: Target size for resizing the image.

    Returns:
        Denoised PIL image.
    """
    # Resize and normalize the image
    transform = Resize(target_size, antialias=True)
    image_resized = transform(image)
    
    image_tensor = ToTensor()(image_resized).unsqueeze(0).to(device)  # Add batch dimension
    image_tensor = image_tensor * 2 - 1  # Normalize to [-1, 1]

    # Ensure even dimensions for compatibility
    h, w = image_tensor.shape[2], image_tensor.shape[3]
    if h % 2 != 0 or w % 2 != 0:
        image_tensor = F.pad(image_tensor, (0, w % 2, 0, h % 2))

    # Generate random timesteps
    timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (1,), device=device).long()

    # Predict noise and reconstruct the denoised image
    model.unet.eval()
    with torch.no_grad():
        pred_noise = model.unet(image_tensor, timesteps).sample
        denoised_tensor = image_tensor - pred_noise

    # Post-process the denoised image
    denoised_tensor = torch.clamp((denoised_tensor + 1) / 2, 0, 1)  # Scale back to [0, 1]
    denoised_image = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    denoised_image = denoised_image.astype("uint8")
    
    return Image.fromarray(denoised_image)

# Function to apply super-resolution
def super_resolve_image(model, image):
    image_tensor = ToTensor()(image).unsqueeze(0).to('cpu')
    with torch.no_grad():
        output_tensor = model(image_tensor).squeeze(0).clamp(0, 1)
    output_image = output_tensor.permute(1, 2, 0).mul(255).byte().cpu().numpy()
    return Image.fromarray(output_image)

# Default slider values
default_values = {
    "brightness": 1.0,
    "sharpness": 1.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "color_intensity": 1.0,
    #"hue": 0,
    "rotation": 0,
    "grayscale": False,
}

# Landing Page
st.title("AI-Powered Image Wizard")
st.write("Enhance and transform your images with AI-powered adjustments.")

col1_a, colc, cold = st.columns((4, 4, 4))
# File Upload
with col1_a:
    st.write("") 
    st.write("") 
    st.write("""
                ##### Magic Starts Here!
                """)
uploaded_file = col1_a.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

with colc:
    st.write("") 
    st.write("") 
    st.write("""
                ##### AI-Powered Adjustments!
                """)
    st.write("") 
    st.write("") 
denoising = colc.checkbox(
    "Image Denoising",
    value=False,
    key="denoising" )  
super_resolution = colc.checkbox(
    "Image Super-Resolution",
    value=False,
    key="super_resolution" ) 
with cold:
    st.write("") 
    st.write("") 
    st.write("""
                ##### 
                """)  
    st.write("") 
    st.write("")   
colorization = cold.checkbox(
    "Image Colorization",
    value=False,
    key="colorization" )  
inpainting = cold.checkbox(
    "Image Inpainting",
    value=False,
    key="inpainting" )     

if uploaded_file:
    # Load the uploaded image
    original_image = Image.open(uploaded_file)

    # Resize the original image for consistency
    resized_original = resize_image(original_image)

    #col1_a, col1_b, colc, cold = st.columns((2, 2, 4, 4))

    col1_0, col1_1, col1, col2 = st.columns((2, 2, 4, 4))

    # Sidebar options for cropping and adjustments
    #reset_clicked = col1_0.button("Reset Fine Tune-Ups")
    with col1_0:
        st.write("""
                    #
                    ##### Fine Tune-Ups
                    """)
    with col1_1:
        st.write("""
                    #
                    ##### 
                    """)    
    
    #col1_0.title("Fine Tune-Ups")
    #enable_crop = st.sidebar.checkbox("Enable Cropping", value=False)

    #st.sidebar.title("Adjustments")

    # Sliders for Fine Tune-Ups in the left column
    brightness = col1_0.slider(
        "Brightness (Default: 1.0)",
        0.5, 3.0,
        default_values["brightness"],
        key="brightness"
    )
    sharpness = col1_0.slider(
        "Sharpness (Default: 1.0)",
        0.5, 3.0,
        default_values["sharpness"],
        key="sharpness"
    )
    contrast = col1_0.slider(
        "Contrast (Default: 1.0)",
        0.5, 3.0,
        default_values["contrast"],
        key="contrast"
    )
    enable_crop = col1_0.checkbox(
        "Enable Cropping",
        value=False,
        key="enable_crop"
    )

    # Sliders for Fine Tune-Ups in the right column
    saturation = col1_1.slider(
        "Saturation (Default: 1.0)",
        0.5, 3.0,
        default_values["saturation"],
        key="saturation"
    )
    color_intensity = col1_1.slider(
        "Color Intensity (Default: 1.0)",
        0.5, 3.0,
        default_values["color_intensity"],
        key="color_intensity"
    )
    rotation = col1_1.slider(
        "Rotation (degrees - Default: 0)",
        -180, 180,
        default_values["rotation"],
        key="rotation"
    )
    grayscale = col1_1.checkbox(
        "Convert to Grayscale",
        value=default_values["grayscale"],
        key="grayscale"
    )

    # Side-by-side display
    #col1, col2 = st.columns(2)
    #col1_0, col1_1, col1, col2 = st.columns((1, 1, 4, 4))

    # Cropping and original image display
    with col1:
        st.subheader("Original Image")
        if enable_crop:
            # Use cropping tool directly on the resized original image
            cropped_image = st_cropper(resized_original, box_color="blue", aspect_ratio=None)
            image_to_process = cropped_image  # Pass cropped image for further processing
        else:
            st.image(resized_original, caption="Uploaded Image", use_container_width=True)
            image_to_process = resized_original  # Pass the resized original image if cropping is disabled

    # Processed image display
    with col2:
        st.subheader("Processed Image")
        # Apply adjustments to the image, including denoising
        processed_image = adjust_image(
            image=image_to_process,
            contrast=contrast,
            saturation=saturation,
            rotation=rotation,
            grayscale=grayscale,
            brightness=brightness,
            sharpness=sharpness,
            color_intensity=color_intensity,
            apply_denoising=denoising,
            apply_super_resolution=super_resolution
        )
        st.image(processed_image, caption="Transformed Image", use_container_width=True)

    # Save processed image to a buffer
    buffer = io.BytesIO()
    processed_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Download Processed Image
    col2.download_button(
        "Download Processed Image",
        data=buffer,
        file_name="processed_image.png",
        mime="image/png",
    )
