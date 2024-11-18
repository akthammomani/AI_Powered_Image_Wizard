# AI Powered Image Wizard

This project is a part of the **Applied Computer Vision for AI** (AAI-521) course in [the Applied Artificial Intelligence Master Program](https://onlinedegrees.sandiego.edu/masters-applied-artificial-intelligence/) at [the University of San Diego (USD)](https://www.sandiego.edu/). 

-- **Project Status: Ongoing**

## Introduction

The AI Powered Image Wizard is an AI system designed to restore and enhance images using cutting-edge generative AI models. This application leverages pretrained models from Hugging Face to perform advanced image enhancement tasks, including denoising, super-resolution, colorization, and inpainting. The goal is to provide users with an intuitive tool for reviving old, damaged, or low-quality images effortlessly.


## Objective

The objective of this project is to develop an AI-based web application that:

* Restores old or damaged images by filling in missing parts and removing imperfections.
* Enhances low-resolution images by increasing clarity and sharpness.
* Adds vibrant colors to grayscale or black-and-white photos.
* Provides seamless functionality for users to process and improve their images in one platform.


## Dataset

The datasets used in this project are carefully selected to cover diverse image enhancement tasks:

* **Denoising:** [BSD500](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500) and [DIV2K](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images) datasets, with synthetic noise added to train the model.
* **Super-Resolution:** [DIV2K](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images) and [Flickr2K](https://www.kaggle.com/datasets/yeueee/flickr2k) datasets, providing high-quality image pairs for upscaling.
* **Colorization:** [COCO](https://cocodataset.org/#home) and [ImageNet](https://www.kaggle.com/c/imagenet-object-localization-challenge) datasets, with grayscale versions created for input.
* **Inpainting:** [Paris Street View](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset) and [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) datasets, where masks simulate missing or damaged areas.

## Methods Used

The project incorporates the following methods:

* **Image Denoising:** Removal of noise while preserving critical details using pretrained models like Denoising Diffusion Probabilistic Models (DDPM).
* **Image Super-Resolution:** Upscaling low-resolution images with SwinIR and ESRGAN models.
* **Image Colorization:** Automatically adding colors to grayscale images using colorization transformers and DeOldify.
* **Image Inpainting:** Filling in missing or damaged parts with LaMa and Stable Diffusion Inpainting models.

## Technologies

The application is built with the following technologies:

* **Frontend and Deployment:** Streamlit for the user interface, deployed on Streamlit Cloud.
* **Backend:** Python for model integration and data processing.
* **AI Frameworks:** TensorFlow, PyTorch, and Hugging Face for pretrained model fine-tuning.
* **Visualization:** Matplotlib and Seaborn for visualizing results.
 

## Future Improvements


## Contributing
Contributions are welcome for future improvements after the initial development phase.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Acknowledgements
A special thanks to **Professor Roozbeh Sadeghian, Ph.D.**, for his invaluable guidance and support throughout this class/project.


