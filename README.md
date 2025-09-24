

---

# Pneumonia and COVID-19 Detection using Deep Learning

This project uses a deep learning-based Convolutional Neural Network (CNN) to classify chest X-ray images into three categories:  
1. **Normal**  
2. **Pneumonia**  
3. **COVID-19**

## Overview  
Early detection of respiratory conditions like pneumonia and COVID-19 can significantly improve patient outcomes. This project leverages a dataset of chest X-rays to train a model capable of detecting these conditions with high accuracy.

---

## Features
- **Multi-Class Classification**: Distinguishes between Normal, Pneumonia, and COVID-19 cases.  
- **Deep Learning Architecture**: Uses CNNs for feature extraction and classification.  
- **Web-Based Deployment**: Model is deployed using **Flask**, providing a simple web interface for testing.  

---

## Dataset  
The dataset used is from **Kaggle**:  
- **Name**: Chest X-ray Dataset (COVID-19 & Pneumonia)  
- **Size**: 2 GB  
- **Classes**:  
  - Normal  
  - Pneumonia  
  - COVID-19  
- **Preprocessing**: Images were resized to 224x224 pixels and normalized for faster training.  

---

## Tools and Libraries  
- **Python**  
- **TensorFlow/Keras** for building the CNN model  
- **OpenCV** and **Pillow** for image preprocessing  
- **Flask** for deploying the model  
- **NumPy** and **Pandas** for data handling  
- **Matplotlib** and **Seaborn** for visualizations  

---

## Installation

### 1. Clone the Repository  
```bash
git clone https://github.com/Gowdakiran-ui/pneumonia-covid19-detection.git
cd pneumonia-covid19-detection
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Download Dataset  
Download the dataset from [Kaggle](https://www.kaggle.com) and place it in the `data/` directory.

---

## How to Run the Project  

### Training the Model  
```bash
python train_model.py
```
This will preprocess the data, train the CNN, and save the trained model to the `models/` directory.

### Testing the Model  
```bash
python test_model.py
```
Evaluate the performance of the model on unseen data.

### Deploy the Model using Flask  
```bash
python app.py
```
Access the web app at `http://127.0.0.1:5000/` to upload X-ray images for prediction.

---

## Results  
- **Training Accuracy**: X%  
- **Validation Accuracy**: Y%  
- **Model Performance**:  
  - Precision:  
  - Recall:  
  - F1-Score:  

Visualization of sample predictions:  

| Image         | Predicted Class | Actual Class |  
|---------------|-----------------|--------------|  
| ![Sample1](images/sample1.png) | Normal        | Normal       |  
| ![Sample2](images/sample2.png) | COVID-19      | COVID-19     |  

---

## Future Work  
- Incorporate transfer learning for faster convergence.  
- Extend the model for additional respiratory conditions.  
- Optimize deployment for cloud-based platforms.  

---

## Author  
[Kiran Gowda](https://github.com/Gowdakiran-ui)  

Feel free to reach out for any questions or collaborations!  

--- 

