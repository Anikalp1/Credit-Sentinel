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
- Git

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Anikalp1/Credit-Sentinel.git
   cd Credit_Sentinel
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
   - Download it from [Google Drive](https://drive.google.com/drive/folders/1jczSrcBCYfgg7khNGWl07jVcxpg1u8y8?usp=sharing) and place this folder in the directory at the root of your project.

5. **Run the Application:**

   ```bash
   python app.py
   ```

   Access the application at [http://localhost:5000](http://localhost:5000) in your web browser.

## Configuration

### Setting Up OpenAI API Key

To use the OpenAI API for this project, you need to set up your API key. Follow these steps:

1. **Create a `.env` file** in the root directory of your project if it doesn't already exist.

2. **Add your OpenAI API key** to the `.env` file:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key.

3. **Ensure the `python-dotenv` package is installed** to load environment variables from the `.env` file. This package is already listed in your `requirements.txt`:

   ```plaintext
   python-dotenv
   ```

4. **The application will automatically load the API key** from the `.env` file when it starts, as shown in the `app.py` file:

   ```python:app.py
   startLine: 29
   endLine: 33
   ```

By following these steps, you will successfully configure your application to use the OpenAI API key. Make sure to keep your API key secure and do not share it publicly.


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
