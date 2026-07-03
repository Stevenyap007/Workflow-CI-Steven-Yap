import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Adjust the path to point to your preprocessing output
data_path = 'mall_customers_preprocessing/cleaned_data.csv'
df = pd.read_csv(data_path)

# Define Features (X) and Target (y)
X = df.drop('Age_Group', axis=1)
y = df['Age_Group']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Configure MLflow Tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Mall_Customers_Classification")

mlflow.sklearn.autolog()

# 3. Train the Model inside an MLflow run
with mlflow.start_run():
    print("Training Random Forest Model...")
    
    # Initialize a simple Scikit-Learn model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Test the model
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f" Training complete! Accuracy: {acc:.2f}")
    print("Check your local MLflow UI to see the logged artifacts.")

    import shutil
    if os.path.exists("saved_model"):
        shutil.rmtree("saved_model")
    mlflow.sklearn.save_model(model, "saved_model")
    print("Model saved locally for Docker build!")