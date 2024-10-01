from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from sklearn.exceptions import NotFittedError

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Collect form data
            data = CustomData(
                age=int(request.form.get('age')),
                sex=request.form.get('sex'),
                bmi=float(request.form.get('bmi')),
                children=int(request.form.get('children')),
                smoker=request.form.get('smoker'),
                region=request.form.get('region')
            )
            
            # Convert the data into a dataframe
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            # Prediction Pipeline
            predict_pipeline = PredictionPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print("After Prediction")
            
            # Render the result
            return render_template('home.html', results=results[0])

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return render_template('home.html', results="Invalid input data.")
        except NotFittedError:
            print("Model or Preprocessor not fitted properly.")
            return render_template('home.html', results="Model or Preprocessor not fitted.")
        except Exception as e:
            print(f"Exception: {e}")
            return render_template('home.html', results="An error occurred during prediction.")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
