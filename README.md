# 🏏 AI Cricket Talent Identification System
## 🚀 Final Year Project | AI + ML + Deep Learning

This project is an end-to-end intelligent system designed to identify and rank cricket talent using Machine Learning and Deep Learning.

---

### 🌐 Live Demo
You can access the live dashboard here:
👉 [Click here to view the App](https://cricket-talent-identification-system-pqcvahy8h56ujuncwp4m89.streamlit.app/)

---

### 📋 Table of Contents
* [Features](#-features)
* [Tech Stack](#-tech-stack)
* [System Architecture](#-system-architecture)
* [Installation](#-installation)

---

### ✨ Features
* **Statistical Talent Scoring**: Uses a **Random Forest Regressor** to analyze player performance metrics (Runs, Strike Rate, Wickets, Economy).
* **Action Recognition**: A **Convolutional Neural Network (CNN)** built with TensorFlow to analyze player images.
* **Visual Analytics**: Interactive graphs and side-by-side player comparisons.

---

### 🛠 Tech Stack
* **Frontend**: Streamlit
* **Machine Learning**: Scikit-learn
* **Deep Learning**: TensorFlow & Keras
* **Data Processing**: Pandas & NumPy
* **Visualization**: Matplotlib

---

### 🏗 System Architecture
1. **Data Input**: CSV upload for stats or Image upload for CNN analysis.
2. **Preprocessing**: Data normalization and image resizing ($64 \times 64$).
3. **Modeling**: 
   - **ML**: Random Forest Pipeline.
   - **DL**: Conv2D -> Max Pooling -> Dense Layers.
4. **Output**: Live Leaderboards and AI Talent Ratings.

---

### 🚀 Installation & Local Setup

```bash
# Clone the repository
git clone [https://github.com/singhmahip688-hue/CRICKET-TALENT-IDENTIFICATION-SYSTEM.git](https://github.com/singhmahip688-hue/CRICKET-TALENT-IDENTIFICATION-SYSTEM.git)

# Enter the directory
cd CRICKET-TALENT-IDENTIFICATION-SYSTEM

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py


