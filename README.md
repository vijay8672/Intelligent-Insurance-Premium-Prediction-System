## Insurance Premium Prediction Using Supervised Learning - Regression


### Project Overview
This project focuses on predicting insurance premiums based on customer data using supervised machine learning regression techniques. The target variable, charges (insurance premium), is continuous, making regression the ideal approach. By leveraging data analysis and predictive modeling, this project aims to provide a reliable and efficient system to estimate insurance premiums.

### Dataset: 
We utilized the Medical Cost Personal Dataset, which contains features related to customer demographics, lifestyle, and health.
📂 Dataset Features:

Age: Age of the customer.

BMI: Body Mass Index, an indicator of body fat based on height and weight.

Children: Number of dependents covered under the insurance.

Smoker: Whether the customer is a smoker or not (categorical: yes/no).

Region: Geographical region of the customer (categorical: northeast, northwest, etc.).

Charges: Insurance premium paid (target variable).

You can access the dataset from Kaggle: https://www.kaggle.com/datasets/mirichoi0218/insurance

### Data Preprocessing
Actions Taken:
Handling Missing Values:
Verified dataset integrity and addressed missing values using mean imputation for numerical features and mode imputation for categorical features.

Encoding Categorical Data:
Converted non-numerical columns (smoker and region) into numerical representations using:
Label Encoding for binary variables.
OneHot Encoding for multi-category variables.

Feature Scaling:
Standardized numerical features to ensure consistent magnitude across the dataset.
Applied StandardScaler to variables such as age, bmi, and charges.

### Exploratory Data Analysis (EDA)
__Insights and Visualizations:__

Feature Relationships:

Explored correlations between features and the target variable using heatmaps.Observed significant impact of smoker, bmi, and age on charges.

__Visualizations:__
Created scatter plots to analyze bmi vs. charges and age vs. charges.
Used box plots to identify outliers in numerical features.
Visualized the distribution of premium amounts using histograms.


### Model Building

Selected Machine Learning Models:

Linear Regression: A simple baseline model for understanding linear relationships.

Decision Tree Regressor: Captures non-linear interactions between features.

Random Forest Regressor: An ensemble method for reducing variance and improving accuracy.

XGBoost Regressor: A high-performing, gradient-boosting model for complex datasets.


### Model Training and Evaluation
Process Followed:
1. Data Splitting:
Divided the dataset into 70% training data and 30% testing data to evaluate model generalization.


2. Model Training:
Trained each regression model using the training dataset.
Monitored training loss to avoid overfitting.

3. Evaluation Metrics:

Evaluated model performance using:

Mean Absolute Error (MAE): Measures average absolute error.

Mean Squared Error (MSE): Penalizes large errors more heavily.

R-Squared (R²): Explains the proportion of variance in the target variable accounted for by the features.


### Hyperparameter Tuning

Optimization Techniques:

• Applied Grid Search to fine-tune hyperparameters for models like Random Forest and XGBoost.

• Improved model accuracy by optimizing parameters such as:

• Number of estimators.

• Maximum depth.

• Learning rate.


### Model Deployment
Web Application:
1. Framework:
Developed a Flask-based web application to make the predictive model accessible to users.

2. Features:
• Interactive interface to input customer data such as age, BMI, number of children, smoking status, and region.
• Real-time prediction of insurance premiums based on user inputs.

### Repository Structure

![image](https://github.com/user-attachments/assets/18b803f3-a8a1-4600-b48a-007eaa609398)



### Key Takeaways

• Achieved an R² score of 90% using XGBoost, outperforming other models.
• Integrated the model into a Flask web application, making it user-friendly and ready for real-world use.
• Gained actionable insights into how customer features such as smoking habits and BMI influence premium costs.


### Technologies Used
• Languages: Python

• Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, Flask

• Tools: Jupyter Notebooks, Visual Studio Code

• Environment: Local virtual environment

• Deployment: Flask Server


### Future Enhancements
1. Cloud Deployment:
   Host the web application using AWS, Azure, or Heroku for global access.

2. Feature Engineering:
   Add additional features like exercise habits, diet, or health conditions for better predictions.

3. Integration:
   Enable the model to integrate with third-party insurance APIs for seamless deployment in business applications.

### Conclusion

This project demonstrates a robust machine-learning pipeline for predicting insurance premiums. By combining data preprocessing, exploratory analysis, model optimization, and deployment, we successfully developed a predictive tool that offers significant value for insurers and customers alike.
