# Import necessary libraries
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_and_preprocess_data(file_path):
    # Load your dataset from the specified file path
    data = pd.read_csv(file_path)
    
    # Separate the features (X) and target variable (y)
    X = data.drop('bankruptcy status', axis=1)
    y = data['bankruptcy status']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train, y_train):
    # Initialize and train your XGBoostClassifier model
    xgb_classifier = RandomForestClassifier()
    xgb_classifier.fit(X_train, y_train)
    
    return xgb_classifier

if __name__ == "__main__":
    # Define the path to your dataset
    dataset_path = 'data_perentille.csv'  # Replace with the actual path to your dataset
    
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
    
    # Train the XGBoost model
    xgb_model = train_xgboost_model(X_train, y_train)
    
    # Optionally, you can save the trained model to a file
    # xgb_model.save_model('xgb_model.model')
    predictions = xgb_model.predict(X_test)
    xgb_model.score(X_test,y_test)
    #save the model
    file = open("model.pkl", 'wb')
    pickle.dump(xgb_model, file)
