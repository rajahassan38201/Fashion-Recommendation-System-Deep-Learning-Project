https://github.com/user-attachments/assets/1792b319-fe37-4107-9525-1b8096b9860e

# Fashion Recommendation System

A Deep Learning–based Fashion Recommendation System that uses image embeddings to find and recommend visually similar fashion items. This system leverages **TensorFlow**, **ResNet50**, and **K-Nearest Neighbors (KNN)** to deliver image-based recommendations. The web interface is developed using **Streamlit**, making it easy for users to upload their own fashion item images and receive personalized suggestions.

---

## 📌 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Project Architecture](#project-architecture)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Dataset](#dataset)
8. [Project Structure](#project-structure)
9. [Future Enhancements](#future-enhancements)
10. [Author](#author)

---

## 📖 Overview

This project is a **Fashion Recommendation System** built with Deep Learning and traditional machine learning techniques. The system uses a **pre-trained ResNet50 model** to generate **image embeddings** from a dataset of fashion images. These embeddings are then used with a **K-Nearest Neighbors** algorithm to find the closest visual matches for any uploaded image. The frontend is built with **Streamlit**, allowing users to interact with the model through a simple web interface.

---

## ✅ Features

- Upload a fashion item image (e.g., shirt, dress, shoes).
- Automatically generate an image vector using ResNet50.
- Retrieve and display the top 6 visually similar items using KNN.
- Interactive user interface using Streamlit.
- Pre-processed dataset of 45,000+ images.

---

## 🛠️ Technologies Used

- **Programming Language:** Python 3.12
- **Deep Learning:** TensorFlow, ResNet50
- **Machine Learning:** Scikit-learn (K-Nearest Neighbors)
- **Web Framework:** Streamlit
- **Image Processing:** Pillow (PIL)
- **Data Handling:** NumPy, Pandas
- **Serialization:** Pickle

---

## 🧱 Project Architecture

```plaintext
User Image
    │
    ▼
ResNet50 (Pre-trained CNN)
    │
    ▼
Image Embedding Vector
    │
    ▼
K-Nearest Neighbors (KNN)
    │
    ▼
Top 6 Recommended Fashion Items
    │
    ▼
Display on Streamlit Web Interface
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/rajahassan38201/Fashion-Recommendation-System-Deep-Learning-Project.git
cd fashion-recommendation-system
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scriptsctivate     # For Windows
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file, here are the main packages:

```bash
pip install tensorflow scikit-learn streamlit numpy pandas Pillow
```

---

## 🚀 Usage

### 1. Preprocess the Dataset

- Load your 45k fashion image dataset.
- Use **ResNet50** to extract embeddings and save them using Pickle:

```python
# Pseudocode Example
model = ResNet50(...)
embeddings = model.predict(images)
pickle.dump(embeddings, open('embeddings.pkl', 'wb'))
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

### 3. Using the Web Interface

- Upload an image.
- Wait while the system computes the embedding.
- View the top 6 recommended fashion items.

---

## 🗂️ Dataset

- Source: [Kaggle Fashion Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- Size: 45,000+ labeled fashion images.
- Preprocessing: Resized and normalized for ResNet50 compatibility.

---

## 🧾 Project Structure

```plaintext
fashion-recommendation-system/
│
├── app.py                    # Main Streamlit app
├── model/
│   ├── resnet_model.py       # ResNet50 loading and embedding functions
│   ├── knn_model.pkl         # Pickled KNN model
│   └── embeddings.pkl        # Pickled image embeddings
│
├── data/
│   └── images/               # Folder of 45k fashion images
│
├── utils/
│   └── helper_functions.py   # Utility functions for preprocessing, loading, etc.
│
├── requirements.txt
└── README.md
```

---

## 🚧 Future Enhancements

- Add user preference filtering (e.g., color, style).
- Include text-based descriptions for multimodal recommendations.
- Deploy using Docker or cloud platforms (AWS/GCP).
- Use FAISS or Annoy for faster nearest neighbor search.

---

## 👨‍💻 Author

**Hassan**  
GitHub: [github.com/rajahassan38201](https://github.com/rajahassan38201)  
Email: rajahassan38201@gmail.com

---
