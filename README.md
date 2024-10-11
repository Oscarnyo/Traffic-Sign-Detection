# Traffic Sign Detection Using YOLO

This project is part of a machine learning assignment aimed at detecting traffic signs using the YOLO (You Only Look Once) object detection framework. The notebook contains code to train and evaluate a traffic sign detection model using a custom dataset of traffic signs.

## Project Overview

The goal of this project is to use a deep learning approach for object detection to accurately identify various traffic signs in images. We leverage the YOLO model, which is known for its speed and efficiency in detecting objects in real time.

## Features

- **YOLOv8 model** for real-time traffic sign detection.
- Custom dataset exploration and preparation.
- Data visualization using `matplotlib` and `seaborn`.
- Image preprocessing using OpenCV (`cv2`).
- Evaluation of model performance on test images.

## Dataset

The dataset used consists of labeled images of traffic signs. It includes both training and validation sets.

### Data Preprocessing

- Images are preprocessed using OpenCV (`cv2`).
- Labels are in YOLO format, with corresponding bounding boxes for each traffic sign.

## Libraries Used

- `ultralytics`: For the YOLOv8 model.
- `OpenCV` (`cv2`): For image processing and manipulation.
- `PIL`: For handling image files.
- `matplotlib`, `seaborn`: For visualizing the data and the model's performance.
- `torch` and `torchvision`: For managing the deep learning model and dataset handling.
- `tqdm`: For progress bars during training.
  
## Steps

1. **Environment Setup**: Install necessary libraries (e.g., `ultralytics`).
2. **Data Exploration**: Visualize and analyze the dataset, check the structure of the training images, and their corresponding labels.
3. **Model Training**: Fine-tune the YOLOv8 model on the traffic sign dataset.
4. **Evaluation**: Test the model on validation data and visualize the results using `matplotlib`.

## How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/traffic-sign-detection.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter notebook `ml-a2.ipynb` to train the model and visualize the results.

## Results

After training, the model is able to detect various traffic signs with reasonable accuracy. The performance can be visualized in terms of bounding boxes on test images.

