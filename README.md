# USA-predicting-house-prices

USA House Price Prediction
This repository contains a machine learning project focused on predicting house prices in the USA. The analysis is performed using Python, specifically with the pandas and scikit-learn libraries.

Files in this Repository
USA_Housing.csv
This is the dataset used for the analysis. It contains various features related to housing, such as:

Avg. Area Income

Avg. Area House Age

Avg. Area Number of Rooms

Avg. Area Number of Bedrooms

Area Population

Price (The target variable to be predicted)

USA_Predicting_House_Prices.ipynb
This Jupyter Notebook contains the core of the data analysis and model training. The key steps performed in this notebook include:

Data Loading and Exploration: Loading the USA_Housing.csv file and performing an initial check on its structure and basic statistics.

Data Preprocessing: Handling any missing values and preparing the data for the model.

Model Training: Using a Linear Regression model from the scikit-learn library to train on the data.

Model Evaluation: Evaluating the model's performance using metrics like Root Mean Square Error (RMSE) to measure its accuracy.

Model Saving: The trained model is saved using the joblib library to a file named usa house price pred model 100k rmse 25-08-25.pkl for later use.

usa housing for practise pkl file.ipynb
This notebook serves as a practical example for using the saved model. It loads the pre-trained usa house price pred model 100k rmse 25-08-25.pkl file and allows a user to input new data (income, age, rooms, etc.) to get a house price prediction. It demonstrates how to deploy and use a machine learning model for inference without needing to retrain it.

How to Use
Clone this repository to your local machine.

Ensure you have Python and Jupyter installed.

Install the required libraries:

pip install pandas scikit-learn joblib

Open and run the USA_Predicting_House_Prices.ipynb notebook to see the full analysis workflow.

Open and run the usa housing for practise pkl file.ipynb notebook to interact with the trained model and get predictions.
