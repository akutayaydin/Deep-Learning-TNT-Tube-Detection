# TNT Tube Detection

This repository contains a Python script for detecting TNT tubes in images using deep learning techniques. The script utilizes a pre-trained VGG16 model for feature extraction and binary classification.

## Usage
1. Clone this repository to your local machine.
2. Open the provided Jupyter Notebook `TNT_Project_v10.ipynb` using Google Colab.
3. Ensure you have the required dependencies installed (`opencv-python`, `numpy`, `pandas`, `scikit-image`, `scipy`, `keras`, `tensorflow`, `google-colab`, `matplotlib`).
4. Run the notebook cells to perform image preprocessing, model training, and prediction.

## Functionality
- **Image Preprocessing:** The notebook performs various image preprocessing steps, including background and shading correction using the rolling ball algorithm or Gaussian filter.
- **Data Augmentation:** Data augmentation techniques such as rotation, shearing, zooming, and flipping are applied to increase the diversity of the training dataset.
- **Model Building:** A sequential model is created by removing the prediction layer from the pre-trained VGG16 model and adding a new output layer for binary classification.
- **Model Training:** The model is trained using the training dataset, and early stopping and learning rate reduction callbacks are implemented to prevent overfitting.
- **Model Evaluation:** The model is evaluated using the validation dataset, and the accuracy and loss are visualized.
- **Prediction:** The trained model is used to predict the presence of TNT tubes in unlabeled images, and the results are compared with the ground truth labels.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Pandas
- scikit-image
- SciPy
- Keras
- TensorFlow
- Google Colab
- Matplotlib

## Disclaimer
This script is provided for educational and informational purposes only. Ensure compliance with applicable regulations and ethical guidelines when using this script for real-world applications.
