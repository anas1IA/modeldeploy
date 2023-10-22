import joblib
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define phrases for predictions
bankruptcy_phrase = "The company is at risk of bankruptcy."
non_bankruptcy_phrase = "The company is not at risk of bankruptcy."

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Define your list of input feature names
    input_names = ['retained earnings / total assets', 'sales / total assets',
                   '(total liabilities * 365) / (gross profit + depreciation)',
                   'sales (n) / sales (n-1)',
                   'profit on operating activities / total assets',
                   'profit on operating activities / financial expenses',
                   'total sales / total assets',
                   '(current assets - inventories) / long-term liabilities',
                   'profit on operating activities / sales',
                   'rotation receivables + inventory turnover in days',
                   '(inventory * 365) / cost of products sold', 'working capital',
                   'total costs / total sales', 'long-term liabilities / equity',
                   'sales / inventory', 'sales / fixed assets']

    if request.method == 'POST':
        inputs = np.array([])
        for name in input_names:
            input_value = request.form.get(name)
            if input_value is not None:
                input_data = float(input_value)
                inputs = np.append(inputs, input_data)
        data = {name: [value] for name, value in zip(input_names, inputs)}
        df = pd.DataFrame(data)
        df.index = ['Row1']

        if len(inputs) == len(input_names):
            # Make the prediction if all input values were successfully collected
            prediction = model.predict(df)
            # Use the phrases based on the prediction result
            if prediction[0] == 1:
                return bankruptcy_phrase
            else:
                return non_bankruptcy_phrase
        else:
            print("len inputs: ")
            print(len(inputs))
            print("\n")
            print("len input_names: ")
            print(len(input_names))
            return "Prediction: none"

if __name__ == "__main__":
    app.run(debug=True)
