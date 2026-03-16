# Gender Identification using Deep Learning 

## Project Overview
This project focuses on automatic gender identification from facial images using deep learning. The model is built using ResNet50 (a pretrained convolutional neural network) with transfer learning to classify images into Male or Female categories.

The system takes a facial image as input and predicts the gender of the person. This project demonstrates the use of computer vision, deep learning, and transfer learning techniques to build an efficient classification model.

This project can be applied in various domains such as:

- Smart surveillance systems
- Human-computer interaction
- Demographic analysis
- Personalized recommendation systems

---

## Objectives

The main objectives of this project are:

- To build a deep learning model for gender classification
- To apply transfer learning using ResNet50
- To preprocess and augment facial image datasets
- To train a CNN model capable of identifying gender accurately
- To evaluate the performance using standard classification metrics

---

## Technologies Used

| Technology | Purpose |
|------------|--------|
| Python | Programming language |
| TensorFlow / Keras | Deep learning framework |
| ResNet50 | Pretrained CNN model |
| OpenCV | Image processing |
| NumPy | Numerical computations |
| Matplotlib | Visualization |
| Scikit-learn | Model evaluation |

---

## Project Architecture

The workflow of the system follows these steps:

1. Image Dataset Collection  
2. Image Preprocessing  
3. Data Augmentation  
4. Feature Extraction using ResNet50  
5. Model Training  
6. Model Evaluation  
7. Gender Prediction  

```
Input Image
     ↓
Image Preprocessing
     ↓
ResNet50 Feature Extraction
     ↓
Fully Connected Layers
     ↓
Softmax Classification
     ↓
Male / Female Prediction
```

---

## Dataset

The dataset consists of facial images categorized into two classes:

- Male
- Female

### Dataset Structure

```
dataset/
    train/
        male/
        female/

    test/
        male/
        female/
```

Each image is resized to **224 × 224 pixels**, which is the required input size for ResNet50.

---

## Data Preprocessing

The following preprocessing steps are applied:

- Image resizing
- Normalization
- Data augmentation

Data augmentation techniques used:

- Rotation
- Horizontal flipping
- Zooming
- Width and height shifting

Example preprocessing:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)
```

---

## Model Architecture

The model uses **ResNet50 pretrained on ImageNet** as the base model.

Steps followed:

1. Load pretrained ResNet50
2. Freeze convolutional layers
3. Add custom classification layers
4. Train the model on gender dataset

Example model implementation:

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

base_model = ResNet50(weights='imagenet', include_top=False)

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
```

---

## Training the Model

The model is trained using:

- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Metrics: Accuracy  

Example training:

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_generator,
          validation_data=validation_generator,
          epochs=10)
```

---

## Model Evaluation

The trained model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Example evaluation:

```python
loss, accuracy = model.evaluate(test_generator)
print("Accuracy:", accuracy)
```

---

## Prediction

After training, the model can predict gender from a new image.

Example:

```python
import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img('test.jpg', target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)
```

Output:

```
Male
or
Female
```

---

## Results

The model achieves high classification accuracy on the test dataset.

Example results:

| Metric | Value |
|------|------|
| Training Accuracy | 94% |
| Validation Accuracy | 91% |
| Test Accuracy | 90% |

The confusion matrix shows strong classification performance for both categories.

---

## Applications

This project can be used in:

- Smart retail systems
- Security and surveillance
- Demographic analytics
- Social media applications
- Human-computer interaction

---

## Future Improvements

The project can be improved by:

- Increasing dataset size
- Using more advanced architectures (EfficientNet, Vision Transformers)
- Implementing real-time gender detection
- Deploying the model using Flask or FastAPI
- Integrating the model with a web or mobile application

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/gender-identification-resnet50.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python train.py
```

---

## Requirements

```
Python 3.x
TensorFlow
Keras
NumPy
Matplotlib
OpenCV
Scikit-learn
```

---

## Project Structure

```
Gender-Identification/
│
├── dataset/
│
├── models/
│   └── resnet_gender_model.h5
│
├── notebooks/
│   └── training.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── preprocessing.py
│
├── requirements.txt
│
└── README.md
```

---

## Author

Jaisudharshan  
Final Year Student  
Machine Learning & AI Enthusiast
