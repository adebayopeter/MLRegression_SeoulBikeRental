# Seasonal Bike Rental Analysis

## Project Objective:
To build a simple web application using **ML Regression Models** that will allow users to explore data through a user-friendly interface and answer business-related questions.

## Business Problem
This project aims to optimize bike rental operations in Seoul by leveraging data-driven insights. By analyzing various factors affecting bike rentals, we can improve resource allocation, enhance user experience, and potentially increase revenue for the bike rental service. 

## Dataset Overview
The original dataset has the following information:

Entries: 8760

Data Types: float64(6), int64(4), object(4)

Data columns (total 14 columns):

| #  | Column                        | Non-Null Count | Dtype   |
|----|-------------------------------|----------------|---------|
| 0  | Date                          | 8760 non-null  | object  |
| 1  | Rented Bike Count             | 8760 non-null  | int64   |
| 2  | Hour                          | 8760 non-null  | int64   |
| 3  | Temperature(°C)               | 8760 non-null  | float64 |
| 4  | Humidity(%)                   | 8760 non-null  | int64   |
| 5  | Wind speed (m/s)              | 8760 non-null  | float64 |
| 6  | Visibility (10m)              | 8760 non-null  | int64   |
| 7  | Dew point temperature(°C)     | 8760 non-null  | float64 |
| 8  | Solar Radiation (MJ/m2)       | 8760 non-null  | float64 |
| 9  | Rainfall(mm)                  | 8760 non-null  | float64 |
| 10 | Snowfall (cm)                 | 8760 non-null  | float64 |
| 11 | Seasons                       | 8760 non-null  | object  |
| 12 | Holiday                       | 8760 non-null  | object  |
| 13 | Functioning Day               | 8760 non-null  | object  |

## Project Objectives
+ Identify and quantify the top 3-5 factors influencing bike rental demand in Seoul, providing actionable insights for operational decision-making.
+ Develop a regression model to predict hourly bike rental demand with at least 85% accuracy.
+ Design and deploy an interactive web application using Streamlit, allowing users to test the regression model through an intuitive interface.

## Specific Questions/Insights Provided
+ What are the most significant factors influencing bike rental demand in Seoul?
+ How do seasonal changes and weather conditions impact bike rental patterns?
+ Are there any trends or patterns in bike rental usage during holidays or specific hours of the day?
+ How does air quality (visibility) affect bike rental behavior?

## Value Generation for the Organization
+ Efficient Bike Distribution: Enabling more efficient bike distribution across the city based on predicted demand.
+ Marketing Strategies: Informing marketing strategies to boost rentals during typically low-usage times.
+ Financial Planning: Assisting in budget allocation and financial planning based on expected rental patterns.
+ Customer Satisfaction: Improving customer satisfaction by ensuring bike availability during high-demand periods.

## Installation
### Prerequisites
Before you can run the project, make sure you have the following installed:

+ Python 3.8 or higher
+ pip (Python package installer)
+ Virtual environment tool (optional but recommended)

### Step-by-Step Installation Guide
1. Clone the repository:

   ```sh
    git clone https://github.com/yourusername/seasonal-bike-rental-analysis.git
    cd seasonal-bike-rental-analysis
   ```
   
2. Create a virtual environment and activate it:

   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required packages:
   
   ```sh 
   pip install -r requirements.txt 
   ```
   
4. Run the Preprocessing and EDA script

   ```sh
   python preprocessing_eda.py
   ```

## Project Scope
+ Data Formatting and Exploration: Initial formatting and exploratory data analysis.
+ ML Model Development: Building a machine learning model to predict bike rental demand (Regression model).
+ Deployment: Deploying the model using a cloud service (AWS/GCP).
+ Web Application: Developing a Streamlit web app for external users to interact with the model.

# Project Phases
### Phase One
+ Data Acquisition: Gathering relevant datasets for analysis.
+ Data Exploration: Analyzing data to understand its structure and key characteristics.
+ Data Cleaning & Preprocessing: Preparing the data for modeling by handling missing values, outliers, etc.

### Phase Two
+ Feature Engineering: Creating new features or modifying existing ones to improve model performance.
+ Model Building & Selection: Developing and selecting the best regression model for predicting bike rental demand.
+ Model Evaluation & Training: Evaluating model performance and fine-tuning as necessary.

### Phase Three
+ Streamlit App GUI: Designing the graphical user interface for the web application.
+ Deployment: Deploying the web application on a cloud platform (e.g., AWS/GCP) for public use.
