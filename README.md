🧠 Brain Tumor Detection using Deep Learning

This project is a deep learning–based web application that detects the presence of a brain tumor from MRI images.
A Convolutional Neural Network (CNN) model is trained on MRI data and deployed using a Flask web application.

🚀 Features

Upload MRI image through a web interface

Predicts Tumor / No Tumor

Displays prediction confidence

Responsive UI using Bootstrap

Model managed using Git LFS

Deployable on cloud platforms like Render

🧰 Tech Stack

Python

TensorFlow / Keras

Flask

Bootstrap

Git & Git LFS

Google Colab (Training)

Kaggle Dataset

📊 Model Accuracy

The CNN model was trained and evaluated on the Brain Tumor MRI Dataset.
It achieved an overall test accuracy of approximately 86% on unseen MRI images.



🖼️ Accuracy Graph / Evaluation Screenshot:
<img width="693" height="539" alt="image" src="https://github.com/user-attachments/assets/766e5eeb-4289-4b72-8fc7-5744b682d953" />


📂 Project Structure
├── app.py
├── brain_tumor_model.h5
├── requirements.txt
├── templates/
│   └── index.html
└── README.md

🧠 Dataset

Brain Tumor MRI Dataset from Kaggle

Two classes: yes (tumor), no (no tumor)

▶️ How to Run Locally

Clone the repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Install dependencies

pip install -r requirements.txt


Run the Flask app

python app.py


Open in browser

http://127.0.0.1:5000

☁️ Deployment

The application can be deployed on Render using:

gunicorn app:app


The trained model is stored using Git Large File Storage (LFS).

⚠️ Disclaimer

This project is for educational purposes only and should not be used for medical diagnosis.
