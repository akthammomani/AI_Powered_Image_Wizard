
# **AI-Powered Image Wizard: Colorization**

## **Introduction**

The colorization stage of the AI-Powered Image Wizard is designed to transform grayscale (black-and-white) images into vivid, realistic color versions. Leveraging advanced machine learning techniques, this stage enables the AI to understand image context, structure, and textures to recreate accurate and visually appealing color representations. Whether it's restoring old photographs or adding life to artistic sketches, the colorization stage uses deep neural networks to breathe color into every pixel.

## **Dataset**

In this pipeline, we utilize the COCO Dataset (Common Objects in Context) — a large-scale, richly annotated dataset. It includes a diverse range of images with objects and environments commonly encountered in real life. This diversity allows our AI to learn colorization patterns for a wide variety of objects, from faces to natural landscapes, making the model highly versatile and robust for real-world applications.

By combining the COCO dataset's visual richness with state-of-the-art deep learning techniques, the colorization stage ensures highly detailed and contextually appropriate color restoration for any grayscale image.

## **Creating Grayscale Images from Colored Images Using COCO Dataset**

The process of creating grayscale images from colored images is a crucial preprocessing step in many computer vision tasks, such as colorization and image inpainting. By separating the original colored images and their grayscale counterparts, we can effectively build datasets that allow AI models to learn how to reconstruct or enhance images.

This task uses a directory of COCO dataset images to generate two sets of images:

* Grayscale Images: These are created by converting the colored images into single-channel intensity values, removing the color information while retaining structural details.
* Colored Images: The original images are stored as-is to serve as the ground truth for supervised training tasks.

By organizing these images into separate directories, we enable efficient visualization and comparison of grayscale-to-color transformations. This step not only prepares the dataset but also provides a foundation for exploring advanced tasks like AI-driven colorization and restoration.


## **Downloading and Using DeOldify: A State-of-the-Art Model for Image Colorization**

**DeOldify** is a deep learning-based project that leverages modern neural networks to colorize black-and-white images and videos with stunning results. Developed using **PyTorch** and **FastAI**, **DeOldify** is widely recognized for its artistic and photo-realistic image colorization capabilities.

**Why Choose DeOldify for Colorization?**

* High-Quality Results: DeOldify produces visually appealing and vibrant colorized images. It balances realism and creativity, making it suitable for artistic and professional use.

* Pretrained State-of-the-Art Models:

  * Artistic Model: Focuses on adding vibrant, high-saturation colors for dramatic and visually striking results.
  * Stable Model: Designed for realism, making it suitable for archival and historical image restoration.

* Flexibility: DeOldify allows customization through parameters like the render factor, enabling users to control the level of detail and color intensity.

* Ease of Use: The model comes with pretrained weights (ColorizeArtistic_gen.pth and others) and tools for loading, processing, and colorizing images or videos without requiring extensive knowledge of machine learning.

* Versatility: DeOldify supports:

  * Grayscale photo colorization
  * Old video restoration
  * Enhancing faded or partially colored media

* Community and Open-Source Support: As an open-source project, DeOldify has an active community and comprehensive documentation, making it accessible for developers and enthusiasts.


## **Colorization using DeOldify**

Colorization of grayscale images is a computer vision task that aims to add realistic and aesthetically pleasing colors to black-and-white images. This process transforms an image that lacks color information into a colorful representation that is visually engaging and informative. In this implementation, we utilize deep learning techniques, specifically the DeOldify model, to achieve high-quality image colorization with minimal manual intervention.

**Key Highlights of the Colorization Process:**

* Render Factor Tuning: The render factor is an adjustable parameter in DeOldify that allows fine-tuning of the colorization process. By varying this factor, you can control the saturation, vibrancy, and color intensity of the output images.

* Fully Automated Workflow: Once the directories are set up and the model is initialized, the entire process runs automatically, processing all images in the specified directory.

* Scalability: This method supports batch processing of multiple images, making it suitable for large datasets.

* Integration with Deep Learning Models: DeOldify leverages advanced deep learning techniques like GANs to create realistic and artistic colorizations by learning from massive datasets of color images.


## **Colorization Evaluation**

Evaluating the performance of an image colorization model like DeOldify involves assessing how effectively the model can transform grayscale images into realistic and visually appealing colorized versions. The evaluation process can be broadly categorized into two complementary approaches:

### **Visual Evaluation**

* This involves visually comparing the output colorized images with either their original color counterparts or human expectations for the scene. It helps to assess the aesthetic quality, consistency, and realism of the colorization.

* Since colorization is inherently a creative task, subjective visual assessment allows you to evaluate the model's ability to produce outputs that are pleasing to the human eye.

![colorized_visual](https://github.com/user-attachments/assets/6c50e3b8-a1b3-48eb-a4d8-29f9c2e3625a)

**Summary Highlights**

* Color Accuracy:

  * Colorized images closely resemble the originals, with major objects like trams, cars, and wooden cabinets showing realistic colors.
  * Some areas, such as skies and grass, lack vibrancy compared to the original.

* Semantic Understanding:

  * The model effectively understands objects and context, applying appropriate colors (e.g., cat fur and vehicles).
  * Background details and intricate textures are less consistent
  in complex scenes.

* Texture and Scene Coherence:

  * Outputs retain texture and overall scene coherence, with minimal color bleeding.
  * Shadows and brightness transitions are not always accurate in low-contrast areas.

* Strengths and Limitations:

  * Strengths: Handles distinct objects and simple scenes well.
  * Limitations: Struggles with complex lighting, fine details, and subtle gradients.

### **Metric-based Evaluation**

Objective evaluation using numerical metrics to quantify the quality of colorized images compared to their original counterparts.
Common metrics:
* Peak Signal-to-Noise Ratio (PSNR): Measures the fidelity of the colorized image by comparing its pixel-by-pixel difference with the original image. Higher PSNR indicates better quality.
* Structural Similarity Index (SSIM): Evaluates the perceptual similarity between the colorized and original images by considering luminance, contrast, and structural patterns. Higher SSIM reflects greater similarity.

Numerical metrics provide an unbiased and reproducible way to measure the accuracy of the model’s outputs, especially for large datasets.

![colorization_metric](https://github.com/user-attachments/assets/3dc4f012-dac1-41ce-931d-bbd62c25c386)

**Summary Highlights:**

**PSNR Insights:**

* A good balance between noise reduction and detail preservation is evident in most images. However, a few images with lower PSNR suggest issues like color bleeding or loss of finer details.
* The high percentile values indicate the majority of the images are well-restored, with minimal perceptual differences from the originals.

**SSIM Insights:**

* Strong structural similarity across the dataset highlights the model's ability to maintain spatial and textural fidelity.
* Lower SSIM in some cases may point to challenges in capturing complex patterns or fine-grained features in certain regions.
*High upper-percentile SSIM indicates consistent performance for most images, showcasing robustness in preserving the integrity of the original structures.





