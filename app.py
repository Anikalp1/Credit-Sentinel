import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import bcrypt
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from werkzeug.utils import secure_filename
import openai
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import seaborn as sns
from dotenv import load_dotenv

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Define the directory where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI API client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Ensure the static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Add these column names at the top of your file with other global variables
FEATURE_COLUMNS = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                  'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                  'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Class']

def generate_model_metrics_graphs(y_true, y_pred, model_name):
    # Set the backend explicitly for this function
    plt.switch_backend('Agg')
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    # Create a bar graph
    metrics = ['Accuracy', 'Precision', 'Recall']
    values = [accuracy, precision, recall]

    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'red'])
    plt.ylim(0, 1)
    plt.title(f'{model_name} Metrics')
    plt.ylabel('Score')
    plt.xlabel('Metrics')

    # Save the plot as an image file
    plt.savefig(f'static/{model_name}_metrics.png')
    plt.close()

def prepare_model():
    # Load your dataset
    data = pd.read_csv('data/creditcard.csv')  # Make sure this file exists
    
    # Split features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'static/fraud_detection_model.joblib')
    
    return X_train, y_train

def generate_additional_graphs(data):
    # Set the backend explicitly for this function
    plt.switch_backend('Agg')

    # Histogram for the 'Amount' column
    plt.figure(figsize=(8, 6))
    plt.hist(data['Amount'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Amount')
    plt.ylabel('Frequency')
    plt.savefig('static/amount_distribution.png')
    plt.close()

    # Scatter plot for 'Time' vs 'Amount'
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Time'], data['Amount'], alpha=0.5, c=data['Class'], cmap='coolwarm')
    plt.title('Time vs Amount')
    plt.xlabel('Time')
    plt.ylabel('Amount')
    plt.savefig('static/time_vs_amount.png')
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('static/correlation_heatmap.png')
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")
    
@app.route('/admin')
def admin_page():
    return render_template("admin.html",
                           statistical_analysis="No data available",
                           fraudulent_count=0, non_fraudulent_count=0,
                           iso_forest_accuracy=0, iso_forest_error=0,
                           iso_forest_classification_report="No report available",
                           svm_accuracy=0, svm_error=0,
                           svm_classification_report="No report available",
                           logistic_accuracy=0, logistic_error=0,
                           logistic_classification_report="No report available")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Assuming the file is a CSV file, you can read it into a DataFrame
    data = pd.read_csv(file)

    # Generate additional graphs
    generate_additional_graphs(data)

    # Statistical analysis
    statistical_analysis = data.describe()

    # Number of fraudulent and non-fraudulent data points
    fraudulent_count = (data['Class'] == 1).sum()
    non_fraudulent_count = (data['Class'] == 0).sum()

    # Split the dataset into features and target variable
    X = data.drop(columns=["Class"])
    y = data["Class"]

    # Train Isolation Forest Model
    iso_forest = IsolationForest()
    iso_forest.fit(X)
    iso_forest_predictions = iso_forest.predict(X)
    iso_forest_accuracy = accuracy_score(y, [-1 if pred == -1 else 0 for pred in iso_forest_predictions])
    iso_forest_error = 1 - iso_forest_accuracy
    iso_forest_classification_report = classification_report(y, [-1 if pred == -1 else 0 for pred in iso_forest_predictions])

    # Generate graph for Isolation Forest
    generate_model_metrics_graphs(y, [-1 if pred == -1 else 0 for pred in iso_forest_predictions], 'Isolation_Forest')

    # Train SVM Model
    svm_model = SVC()
    svm_model.fit(X, y)
    svm_predictions = svm_model.predict(X)
    svm_accuracy = accuracy_score(y, svm_predictions)
    svm_error = 1 - svm_accuracy
    svm_classification_report = classification_report(y, svm_predictions)

    # Generate graph for SVM
    generate_model_metrics_graphs(y, svm_predictions, 'SVM')

    # Train Logistic Regression Model
    logistic_model = LogisticRegression()
    logistic_model.fit(X, y)
    logistic_predictions = logistic_model.predict(X)
    logistic_accuracy = accuracy_score(y, logistic_predictions)
    logistic_error = 1 - logistic_accuracy
    logistic_classification_report = classification_report(y, logistic_predictions)

    # Generate graph for Logistic Regression
    generate_model_metrics_graphs(y, logistic_predictions, 'Logistic_Regression')

    return render_template('admin.html', statistical_analysis=statistical_analysis,
                           fraudulent_count=fraudulent_count, non_fraudulent_count=non_fraudulent_count,
                           iso_forest_accuracy=iso_forest_accuracy, iso_forest_error=iso_forest_error,
                           iso_forest_classification_report=iso_forest_classification_report,
                           svm_accuracy=svm_accuracy, svm_error=svm_error,
                           svm_classification_report=svm_classification_report,
                           logistic_accuracy=logistic_accuracy, logistic_error=logistic_error,
                           logistic_classification_report=logistic_classification_report,
                           additional_graphs=True)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_question = request.form['question']
    
    # Use the new OpenAI API method with a prompt for concise answers
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Provide short and precise answers to the user's questions. If the user asks about the project, tell them that it is a credit card fraud detection project. If the user asks about the dataset, tell them that it is a credit card dataset from kaggle. If the user asks about the models, tell them that there are three models: Isolation Forest, SVM, and Logistic Regression. Tell them that Isolation Forest is used for outlier detection, SVM is used for classification, and Logistic Regression is used for logistic regression."},
            {"role": "user", "content": user_question}
        ]
    )
    ai_response = response['choices'][0]['message']['content'].strip()
    
    return {'response': ai_response}

@app.route('/faq')
def faq_page():
    return render_template('faq.html')

@app.route('/predict_form')
def predict_form():
    return render_template('predict_single.html')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        # Get the input string from the request
        input_data = request.json.get('data', '')
        
        # Split the string into values and convert to float
        values = [float(x) for x in input_data.split(',')]
        
        # Remove the class label (last value) and create a DataFrame
        features = values[:-1]  # Remove the last value (class label)
        df = pd.DataFrame([features], columns=FEATURE_COLUMNS[:-1])  # Exclude 'Class' from columns
        
        # Load the trained model
        try:
            model = joblib.load('static/fraud_detection_model.joblib')
        except:
            # If model doesn't exist, train it
            X_train, y_train = prepare_model()
            model = joblib.load('static/fraud_detection_model.joblib')
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'result': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'probability': {
                'legitimate': float(probability[0]),
                'fraudulent': float(probability[1])
            }
        })
        
    except ValueError as e:
        return jsonify({'error': 'Invalid input format. Please provide comma-separated numerical values'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
