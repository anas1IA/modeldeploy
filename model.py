# Import necessary libraries
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier

def load_and_preprocess_data(file_path):
    # Load your dataset from the specified file path
    data = pd.read_csv(file_path)
    
    # Separate the features (X) and target variable (y)
    X = data.drop('bankruptcy status', axis=1)
    y = data['bankruptcy status']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_random_forest_model(X_train, y_train):
    # Initialize and train your RandomForestClassifier model
    rf_classifier = RandomForestClassifier()  # Use RandomForestClassifier here
    rf_classifier.fit(X_train, y_train)
    
    return rf_classifier

if __name__ == "__main__":
    # Define the path to your dataset
    dataset_path = 'data_perentille.csv'  # Replace with the actual path to your dataset
    
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
    
    # Train the Random Forest model
    rf_model = train_random_forest_model(X_train, y_train)
    
    # Save the model using the pickle module
    with open("model.pkl", 'wb') as file:
        pickle.dump(rf_model, file)
    
    predictions = rf_model.predict(X_test)
    accuracy = rf_model.score(X_test, y_test)
