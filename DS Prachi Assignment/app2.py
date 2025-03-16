import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
import warnings
warnings.filterwarnings("ignore")
import pickle
import os
from flask import Flask, request, jsonify, render_template
#Load the dataset and display the first few rows.
df = pd.read_csv('C:\\Users\\vaish\\OneDrive\\Desktop\\DS Prachi Assignment\\diabetes.csv')
df.head()
#Exploratory Data Analysis (EDA)
print(df.describe())
df.info()
#Handle missing values appropriately.
df.fillna(df.mean(), inplace=True)
print("Missing values per column:\n", df.isnull().sum())
#Get summary statistics
# Summary statistics for specific columns
df[['Pregnancies', 'Glucose', 'Outcome']].describe()
# Histograms for all columns
df.hist(figsize=(15, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle('Distribution of All Features')
plt.show()

# Box plots for all columns
plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(y=df[column])
    plt.title(f'Box Plot - {column}')
plt.tight_layout()
plt.show()
# Outlier detection using box plots
sns.boxplot(data=df, orient='h')
plt.show()
# Removing outliers using Z-score
from scipy.stats import zscore

z_scores = zscore(df)
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]
#Assuming df is your DataFrame and "Outcome" is your target colummn
correlation_matrix = df.corr()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
# Feature Selection
selected_features = ["Age", "Glucose", "BMI", "BloodPressure"]
# Ensure these columns exist in the dataset
for feature in selected_features:
    if feature not in df.columns:
        raise KeyError(f"Error: '{feature}' column not found in dataset!")

X = df[selected_features]  # Features
y = df["Outcome"]  # Target Variable
print(f"Selected Features: {selected_features}")
print(f"Outcome Variable: {y.name}")
# Part 3: Model Development & Training
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Save models
pickle.dump(logreg_model, open("logistic_regression.pkl", "wb"))
pickle.dump(dt_model, open("decision_tree.pkl", "wb"))
# Model Training
# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Save the scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))
# Model Evaluation
lr_preds = np.round(lr_model.predict(X_test))  # Rounding for classification
dt_preds = dt_model.predict(X_test)

print("Logistic Regression Model")
print(classification_report(y_test, lr_preds))
print("Logistic Regression Accuracy",accuracy_score(y_test, lr_preds))
print("Confusion matrix: ")
print(confusion_matrix(y_test, lr_preds))
print("\n")
print("Decision Tree Model")
print(classification_report(y_test, dt_preds))
print("Decision Tree Model Accuracy",accuracy_score(y_test, dt_preds))
print("Confusion matrix: ")
print(confusion_matrix(y_test, dt_preds))
accuracy_dt = accuracy_score(y_test, dt_preds)
accuracy_lr = accuracy_score(y_test, lr_preds)


print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")
print(f"Logistic Regression Accuracy: {accuracy_lr * 100:.2f}%")
# Confusion Matrix Logistic Regression
plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, cmap='Blues', fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.show()
# Confusion Matrix Decision Tree Model
plt.figure(figsize=(5, 5))
sns.heatmap(confusion_matrix(y_test, dt_preds), annot=True, cmap='Blues', fmt='d')
plt.title("Decision Tree Confusion Matrix")
plt.show()
# Save models
pickle.dump(lr_model, open("Logistic_regression.pkl", "wb"))
pickle.dump(dt_model, open("decision_tree.pkl", "wb"))
print("Models saved successfully!")
# Flask App
app = Flask(__name__)

# Check if model files exist
if os.path.exists("scaler.pkl") and os.path.exists("decision_tree.pkl"):
    scaler = pickle.load(open("scaler.pkl", "rb"))
    dt_model = pickle.load(open("decision_tree.pkl", "rb"))
else:
    raise FileNotFoundError("Error: Required model files not found! Train and save models first.")

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect only 4 required features
        features = [float(request.form.get(key)) for key in ["Age", "Glucose", "BMI", "BloodPressure"]]

        # Convert to NumPy array and reshape
        input_data = np.array(features).reshape(1, -1)

        # Apply Standard Scaling
        input_data = scaler.transform(input_data)

        # Make Prediction
        prediction = dt_model.predict(input_data)

        return render_template('index2.html', prediction_text=f'Predicted Disease Status: {prediction[0]}')

    except Exception as e:
        return render_template('index2.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)