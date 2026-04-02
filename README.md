🏏 AI Cricket Talent Identification System
🚀 Final Year Project | AI + ML + Deep Learning
This project is an end-to-end intelligent system designed to identify and rank cricket talent using Machine Learning (Random Forest) for statistical data and Deep Learning (CNN) for image-based action analysis.
🌐 Live DemoYou can access the live dashboard here:
https://cricket-talent-identification-system-pqcvahy8h56ujuncwp4m89.streamlit.app/
📖 Table of Contents
Features
Tech Stack
System Architecture
Installation
Screenshots
✨ Features
Statistical Talent Scoring: Uses a Random Forest Regressor to analyze player performance (Runs, Strike Rate, Wickets, Economy) and provide a "Talent Score."
Action Recognition: A Convolutional Neural Network (CNN) built with TensorFlow/Keras to analyze player images and action shots.
Player Comparison: Side-by-side visual analytics to compare two players' strengths and weaknesses.
Interactive Dashboard: Built with Streamlit for a clean, user-friendly professional interface.
🛠 Tech Stack
Frontend: Streamlit (Python-based Web Framework)
Machine Learning: Scikit-learn (Random Forest)
Deep Learning: TensorFlow & Keras (CNN)
Data Handling: Pandas & NumPy
Visualization: Matplotlib
🏗 System Architecture
Data Input: CSV upload for statistics or Image upload for action analysis.
Preprocessing: Normalization of cricket stats and image resizing ($64 \times 64$) for the CNN.
Processing:
        ML Pipeline: Feature extraction $\rightarrow$ Random Forest Prediction.
        DL Pipeline: Conv2D Layers - Max Pooling - Dense Layer Output.
Output: Live Leaderboards, Comparison Graphs, and AI Talent Ratings.
🚀 Installation & Local Setup
Clone the repository
Bash
        
cd your-repo-name
Install dependencies
Bash
        pip install -r requirements.txt
Run the App
Bash
        streamlit run app.py
👥 Author
MAHIP SINGH - Final Year Information Science Engineering Student At Sir. M Visvesvaraya Institute Of Technology
