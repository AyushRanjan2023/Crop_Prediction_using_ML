import numpy as np
from flask import Flask, request, render_template
import pickle

# Create Flask app
flask_app = Flask(__name__)
# Load the pre-trained model (ensure model.pkl is in the correct directory)
try:
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    print("Error: 'model.pkl' not found. Please ensure the model file is in the correct directory.")
    exit()

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve input data from the form
        float_features = [float(x) for x in request.form.values()]

        # Ensure correct number of features (7 inputs: Nitrogen, Phosphorous, Potassium, Temperature, Humidity, pH, Rainfall)
        if len(float_features) != 7:
            return render_template("index.html",
                                   prediction_text="Error: Expected 7 inputs, got {}.".format(len(float_features)))
        # Convert features into a format suitable for prediction
        features = [np.array(float_features)]
        # Get the prediction from the model
        prediction = model.predict(features)
        # Render the prediction result on the page
        return render_template("index.html", prediction_text="The Predicted Crop is: {}".format(prediction[0]))

    except Exception as e:
        # Handle errors gracefully
        return render_template("index.html", prediction_text="Error: Could not make prediction. {}".format(str(e)))

if __name__ == "__main__":
    flask_app.run(debug=True)
