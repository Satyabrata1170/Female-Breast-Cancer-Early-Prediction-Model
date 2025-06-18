# Female-Breast-Cancer-Early-Prediction-Model


This project is an AI-powered web application for detecting breast cancer using image classification. It allows users to either **upload an image** or **use a live camera feed** to get real-time predictions based on a trained deep learning model.

---

## Project Structure

```
BREAST_CANCER_PREDICTION/
│
├── static/uploads/                # Stores uploaded images
├── templates/index.html           # Frontend page
│
├── app.py                         # Flask backend application
│
├── Breast_Cancer_Classification.ipynb  # Jupyter notebook for model development
├── model.weights.best.keras       # Trained Keras model weights
├── README.md                      # Project documentation
```

---

##  Features

- Two prediction modes:
  - 📤 **Upload Mode**: Upload an image from your computer
  - 📷 **Camera Mode**: Use your device's camera to capture a live image
- Predicts if the image shows:
  - **Cancer**
  - **Non-Cancer**
- Displays the uploaded or captured image with the prediction result
- Responsive and intuitive web design with hover and animation effects

---

## 📚 Dataset Description

**Mammogram Mastery: A Robust Dataset for Breast Cancer Detection and Medical Education**

This dataset presents a comprehensive data comprising breast cancer images collected from patients, encompassing two distinct sets: one from individuals diagnosed with breast cancer and another from those without the condition. The dataset is meticulously curated, vetted, and classified by specialist clinicians, ensuring its reliability and accuracy for research and educational purposes.

The dataset offers a unique perspective on breast cancer prevalence and characteristics in the region. With   9,685  images, this dataset provides a rich resource for training and evaluating deep learning algorithms aimed at breast cancer detection. The dataset's inclusion of  X-rays offers enhanced versatility for algorithm development and educational initiatives.

This dataset holds immense potential for advancing medical research, aiding in the development of innovative diagnostic tools, and fostering educational opportunities for medical students interested in breast cancer detection and diagnosis.

---

## Model Details

- Framework: TensorFlow / Keras
- Model Type: Convolutional Neural Network (CNN)
- Accuracy: Achieved **98% classification accuracy**
- Classes: Cancer, Non-Cancer
- Trained on: 9,685 images


## Quotes Added

> _"Early detection is the best protection."_  
> _"Hope is stronger than fear."_  

---

##  Acknowledgements

- Dataset: Mammogram Mastery Dataset (Sulaymaniyah, Iraq)
- TensorFlow/Keras
- Flask Web Framework

---

##  Contact

For queries or suggestions: **satyabratabrahmachary1610@gmail.com**

---
