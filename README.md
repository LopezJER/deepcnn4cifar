# **Very Deep Convolutional Networks for Large-Scale Image Recognition**

Cite- https://doi.org/10.48550/arXiv.1409.1556 , Simonyan and Zisserman (2015)

🌟 deepcnn4cifar 🌟
This repository implementing deep convolutional neural networks for the CIFAR-10 dataset. This project demonstrates training, evaluation, and visualization of deep learning models for image classification. Dive in to explore the world of computer vision with powerful tools and stunning visualizations.

_VGG16 Architecture:_
![alt text](image-2.png)

**Project Structure:**

```
deepcnn4cifar/
├── core/
│   ├── config.py             # Configuration settings
│   ├── model.py              # Model definitions
├── scripts/
│   ├── train.py              # Training scripts
│   ├── evaluate.py           # Evaluation scripts
│   ├── visualize.py          # Visualization scripts
├── utils/
│   ├── load_data.py          # Data loading utilities
│   ├── load_model.py         # Pre-trained model utilities
├── data_preprocessing.ipynb  # Data exploration and preprocessing
├── README.md                 # This file
```

🚀 **Features**

🎨 Visualizations
Our visualizations provide detailed insights into training and evaluation. Below is an example of a training accuracy vs. epochs chart:

![alt text](image.png)

We also include confusion matrices for evaluating model performance:

![alt text](image-1.png)

You can easily adjust the depth and parameters of the vggs in core/model.py.
Utilize scripts/visualize.py to generate interactive charts and matrices.
Track and compare model accuracy and loss metrics with detailed graphs.

**How To Use:**

1. Clone the Repository
   ````git clone https://github.com/yourusername/deepcnn4cifar.git
   cd deepcnn4cifar```
   ````
2. Install Dependencies
   `pip install -r requirements.txt`

3. Train a model on CIFAR-10 using:
   `python scripts/train.py`

4. Evaluate the Model
   `python scripts/evaluate.py`

5. Visualize Results
   `python scripts/visualize.py`

References:
CIFAR-10 Dataset: A dataset of 60,000 32x32 color images in 10 classes.
Deep Learning Framework: Built using PyTorch/TensorFlow.
Paper : "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, 2015).
