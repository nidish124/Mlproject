## Student Performance Prediction App

# END to END Machine learning project
This repository contains a **full-stack machine learning project** built with Python.  
It demonstrates how to design a **modular ML pipeline** covering data ingestion, preprocessing, model training, evaluation, and deployment through a REST API. 

### 1. Overview

This is a full-stack machine learning project in Python. It includes data ingestion, model training, inference through application.py, and reproducible packaging using setup.py

### 2. Features

- Modular Architecture: Organized into src/, data/, and artifacts/ for clean separation of concerns

- REST Interface: application.py serves model predictions programmatically

- Reproducible Environments: Uses requirements.txt and setup.py to manage dependencies and packaging.

### 3. Getting started

git clone https://github.com/nidish124/Mlproject.git

cd Mlproject

conda create --name mlproj python=3.10 -y

conda activate mlproj

pip install -r requirements.txt

streamlit run application.py

### 4. Modules Breakdown

- src/: Core ML code — data processing, model training, evaluation.

- data/: Input datasets.

- artifacts/: Stored models, outputs, or other artifacts.

## 5. Next Steps / Future Work

- Integrate Docker deployment workflows.

- Add MLflow tracking for experiment versioning.

- Include an automated CI/CD pipeline using GitHub Actions.