# AI-Powered Image Wizard

## Denoising: 

The Denoising Stage in our AI-Powered Image Wizard is designed to enhance the quality of noisy images by leveraging state-of-the-art deep learning techniques. This stage is particularly focused on reducing unwanted noise in images, thereby restoring clarity and detail. By employing advanced denoising methods, users can transform grainy, unclear images into sharper and more visually appealing outputs.

### **Dataset**

To fine-tune our model and ensure exceptional performance, we have utilized the **Berkeley Segmentation Dataset 500 (BSDS500)**. This dataset is widely known for its high-quality images and diversity, making it an excellent choice for training and validating denoising models. The dataset provides a robust benchmark for image restoration tasks, ensuring the model's performance is consistent across different noise patterns and image types.

### **Adding Gaussian Noise for Data Augmentation in Image Denoising**

To simulate real-world scenarios and train a robust image denoising model, it is crucial to expose the model to noisy data during training. In this step, we introduce Gaussian Noise to the clean images from the training and validation datasets. Gaussian noise is a common type of noise characterized by a bell-curve distribution, often encountered in real-world scenarios like low-light photography or transmission errors.

The function add_gaussian_noise generates noisy images by adding randomly sampled noise with a specified mean and standard deviation to the original image pixels. This step ensures that the model learns to effectively identify and remove noise patterns while preserving image details.

![denoising_sample](https://github.com/user-attachments/assets/d4fb0aa3-6c3e-4b6f-9cec-13dde9c2adca)


### **Leveraging Pre-Trained Models for Enhanced Image Denoising**

In this stage, we fine-tune a pre-trained Denoising Diffusion Probabilistic Model (DDPM) to enhance its performance on the task of image denoising. We utilize the "google/ddpm-celeba-hq" pre-trained model, a state-of-the-art diffusion-based architecture known for its high fidelity and robustness in image generation and reconstruction tasks. To further optimize the model, we integrate a VGG19-based perceptual loss to focus on preserving fine image details while reducing noise.

**Why Choose "google/ddpm-celeba-hq"?**

The "google/ddpm-celeba-hq" model is specifically designed for high-resolution image synthesis and reconstruction tasks, making it an excellent choice for denoising. Its pre-training on the CelebA-HQ dataset provides the following benefits:

* High-Resolution Training: The CelebA-HQ dataset consists of high-quality facial images, enabling the model to learn fine-grained details and intricate noise patterns.
* Diffusion-Based Generative Power: DDPMs are inherently powerful in modeling complex distributions, making them adept at reconstructing clean images from noisy ones.
* Transfer Learning Capability: Using a pre-trained model significantly reduces the computational cost and training time while leveraging the learned features for image denoising.

**Model Architecture:**

* DDPM Backbone:

  * The UNet architecture in DDPM acts as the backbone, processing noisy images and predicting noise components for denoising.
  * The diffusion process involves learning to reverse the gradual addition of Gaussian noise to generate clean images.
  * The scheduler used in this implementation is the DDIMScheduler, which accelerates the sampling process without compromising output quality.

* Perceptual Loss with VGG19:

  * We enhance the model’s capability by introducing a perceptual loss using the pre-trained VGG19 network.
  * VGG19 extracts deep features from both the denoised and clean images, and the mean squared error (MSE) between these features guides the model to retain structural and perceptual details.

* Loss Functions:

  * MSE Loss: Directly minimizes the pixel-wise differences between the denoised and clean images.
  * Weighted Perceptual Loss: Focuses on high-level feature similarity to ensure the visual realism of the denoised images.

* Optimizer: The AdamW optimizer is employed for stable convergence during training, with a small learning rate to fine-tune the model precisely.

* Training Loop:

  * Each training step involves dynamically sampling timesteps, predicting noise, reconstructing the denoised image, and calculating the combined loss.
  * The model is trained for 20 epochs, and the dynamic timestep generation ensures robustness across varying noise levels.

By combining the strengths of DDPMs with perceptual loss from VGG19, this architecture achieves superior results in preserving image details while effectively removing noise, setting a strong foundation for advanced denoising tasks in the AI-powered Image Wizard pipeline.

### Fine-Tuning: 

* Dataset and Model:

  * The training involved fine-tuning a "google/ddpm-celeba-hq" pre-trained diffusion model.
  * The dataset was processed with Gaussian noise and split into training and validation sets.

* Training Configuration:

  * Number of Epochs: 20
  * Optimizer: AdamW with a learning rate of 1e-5
  * Loss Function: Combination of MSE Loss and Perceptual Loss (weighted at 0.1)
  * Batch Size: 50 batches per epoch

* Performance Highlights:

  * Initial Loss (Epoch 1): 0.2402
  * Final Loss (Epoch 20): 0.0697
  * The loss consistently decreased over the epochs, demonstrating steady model improvement.
  * Significant drops in loss were observed in the early epochs,  stabilizing in later epochs (e.g., around Epoch 10).

* Training Time: Each epoch took approximately 8 minutes, maintaining a processing rate of ~5.66 iterations/second.

* Observations:

  * The model successfully minimized the loss, indicating it effectively learned to reconstruct clean images from noisy inputs.
  * The relatively low final loss suggests a strong convergence to optimal weights for the denoising task.


### **Denoising Evaluation**

In the context of image restoration tasks, such as denoising, evaluating the performance of a model is a crucial step to ensure its effectiveness and generalization. Denoising evaluation typically involves two complementary approaches: visual evaluation and metric-based evaluation.

Combining both visual and metric-based evaluations ensures a holistic assessment of the denoising model:

* Visual evaluation guarantees that results meet perceptual standards for real-world deployment.
* Metric-based evaluation offers statistical evidence of performance, enabling benchmarking and comparisons with other models.

This dual evaluation strategy ensures that the model is both qualitatively pleasing and quantitatively robust.


#### **Visual Evaluation**

* Visual inspection helps to assess the perceptual quality of denoised images, focusing on human perception.
* The goal is to verify that the model reconstructs clean images without losing critical details or introducing artifacts.
* Side-by-side comparisons of clean, noisy, and denoised images allow for qualitative evaluation, ensuring that the results align with human expectations.

![noisy_1](https://github.com/user-attachments/assets/d489fc88-20f9-4f0b-92d1-83c9c106815e)


**Summary Highlights**

* Clean vs. Noisy vs. Denoised:

  * Clean images: Serve as the reference for quality comparison.
  * Noisy images: Generated using Gaussian noise, mimicking real-world distortions.
  * Denoised images: Output of the model, demonstrating noise removal while retaining image structure and key visual features.

* Strengths:

  * Effective noise reduction: The model removes significant noise while maintaining key structural elements.
  * Preserved edges and textures: Features such as object contours and sharp transitions remain intact.
  * Adaptability: Handles diverse scenes, including urban, natural, and human subjects.

* Limitations:

  * Loss of fine details: Subtle textures like fur or grass show mild smoothing.
  * Minor color inconsistencies: In some complex or shaded regions, slight deviations in color tones are observed post-denoising.

* Conclusion: The model demonstrates strong capabilities in balancing noise removal and detail preservation, making it effective for a variety of scenarios. However, further refinements may enhance fine-detail recovery and color consistency.

#### **Metric-based Evaluation**

Quantitative metrics provide objective measurements of model performance by comparing denoised images with their corresponding clean references.
* **Peak Signal-to-Noise Ratio (PSNR):**
  * Measures the ratio between the maximum possible signal power and the noise power in the image.
  * Higher PSNR values indicate better image quality and closer resemblance to the ground truth.
* **Structural Similarity Index (SSIM):**
  * Captures perceptual differences, focusing on luminance, contrast, and structural similarities.
  * SSIM values range between 0 and 1, with values closer to 1 indicating higher similarity to the original image.

These metrics provide quantitative insights into the fidelity and perceptual accuracy of denoised images.

![denoising_metric](https://github.com/user-attachments/assets/0200d5ab-e8f7-457d-994e-2674ed611885)










