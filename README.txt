# Credit Card Fraud Detection with Flask and Machine Learning

Welcome to the **Credit Card Fraud Detection** project, a comprehensive solution that integrates web development, machine learning, and database management to detect fraudulent credit card transactions.

## Project Overview

This project is designed to identify fraudulent credit card transactions using advanced machine learning algorithms. It features a web application built with Flask, allowing users to upload transaction data for fraud analysis.

## Technology Stack

- **Backend:** Flask
- **Frontend:** HTML, CSS (Bootstrap and Tailwind CSS), JavaScript
- **Database:** MongoDB
- **Machine Learning:** Scikit-learn models (Isolation Forest, SVM, Logistic Regression)

## Key Features

- **CSV File Upload:** Users can upload transaction data in CSV format.
- **Fraud Detection:** Utilizes machine learning models to analyze and detect fraudulent transactions.
- **Responsive Design:** Built with Bootstrap and Tailwind CSS for a seamless user experience.
- **Smart FAQ:** An interactive FAQ section to answer questions about credit card fraud detection and the application.

## Installation

### Prerequisites

- Python 3.x
- MongoDB (local or remote)
- Git

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/valakalpesh/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. **Set up Python Environment:**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset:**

   - The dataset is not included in the repository due to size constraints.
   - Download it from [Google Drive](https://drive.google.com/file/d/1tUFNkZL9KaHETca5-e_Q6UaCvgHuXcMt/view?usp=drive_link) and place it in a `data/` directory at the root of your project.

5. **Set up MongoDB:**

   - Install MongoDB locally or use a cloud-based service.
   - Configure the MongoDB connection URI in `app.py` or a separate configuration file.

6. **Run the Application:**

   ```bash
   python app.py
   ```

   Access the application at [http://localhost:5000](http://localhost:5000) in your web browser.

## Project Structure

- **app.py:** Flask application setup, routes, and machine learning models.
- **templates/:** HTML templates for rendering the frontend.
- **static/:** CSS stylesheets and other static files.
- **data/:** Directory to store the dataset (not included in the repository).

## Machine Learning Models

This project employs the following models from scikit-learn for fraud detection:

- **Isolation Forest**
- **Support Vector Classifier (SVC)**
- **Logistic Regression**

## Libraries Used

- **pandas**
- **Flask**
- **pymongo**
- **bcrypt**
- **werkzeug**

## Project Team

Developed by Group 67:

- Anikalp Jaiswal
- Ayushi Singh
- Yashaswi Shahi
