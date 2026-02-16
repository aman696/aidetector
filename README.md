# AI Image Detector
 
 > **A robust, rule-based system for detecting AI-generated images using advanced forensic analysis.**
 
 ---
 
 ## 📖 Table of Contents
 - [Features](#-features)
 - [Tech Stack](#-tech-stack)
 - [Getting Started](#-getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
 - [Usage](#-usage)
 - [Testing Strategy](#-testing-strategy)
 - [Project Structure](#-project-structure)
 - [Research & References](#-research--references)
 - [License](#-license)
 
 ---
 
 ## ✨ Features
 
 - **Multi-Layered Analysis**: Combines multiple forensic techniques to identify artifacts commonly left by generative models.
   - **FFT Analysis**: Detects frequency domain anomalies.
   - **ELA (Error Level Analysis)**: Identifies compression inconsistencies.
   - **Noise Residuals**: Analyzes local noise patterns.
   - **Metadata Inspection**: Checks for missing or suspicious EXIF data.
 - **Web Interface**: User-friendly drag-and-drop interface for easy testing.
 - **Visual Reports**: Provides detailed score breakdowns and visual heatmaps for each analysis method.
 
 ## 🛠 Tech Stack
 
 - **Language**: Python 3.8+
 - **Web Framework**: FastAPI, HTML5, CSS3, JavaScript
 - **Computer Vision**: OpenCV, Pillow, NumPy, SciPy
 - **Machine Learning**: Scikit-learn (SVM Classifier)
 - **Testing**: Pytest, Bandit (Security), Locust (Load Testing)
 
 ## 🚀 Getting Started
 
 ### Prerequisites
 - **Python 3.8** or higher
 - **pip** (Python Package Installer)
 
 ### Installation
 
 1.  **Clone the Repository**
     ```bash
     git clone <repository-url>
     cd aidetector
     ```
 
 2.  **Set Up Virtual Environment**
     It is recommended to use a virtual environment to manage dependencies.
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
     ```
 
 3.  **Install Dependencies**
     ```bash
     pip install -r requirements.txt
     ```
 
 ## 💻 Usage
 
 ### Starting the Web Server
 To launch the application locally:
 
 ```bash
 python app.py
 ```
 
 Once the server is running, open your browser and navigate to:
 **[http://localhost:8000](http://localhost:8000)**
 

 ## 📂 Project Structure
 
 ```text
 aidetector/
 ├── src/                  # Core analysis logic and classifiers
 │   ├── classifier.py     # Main ensemble classifier
 │   ├── ela_analyzer.py   # Error Level Analysis module
 │   └── ...
 ├── web/                  # Frontend assets for the web interface
 ├── data/                 # Dataset storage (Real vs AI images)
 ├── models/               # Trained ML models (e.g., svm_classifier.pkl)
 ├── notebooks/            # Jupyter notebooks for research & experiments
 ├── reports/              # Generated test reports (Security, Load Tests)
 ├── logs/                 # Application server logs
 ├── tests/                # Test suite
 │   └── load/             # Load testing configuration
 ├── app.py                # FastAPI application entry point
 └── requirements.txt      # Project dependencies
 ```
 
 ## 📚 Research & References
 
 This project implements techniques discussed in the following research:
 
 - **Durall et al. (2020)**: *"Unmasking DeepFakes with simple Features"*
 - **Corvi et al. (2023)**: *"Intriguing Properties of Synthetic Images"*

