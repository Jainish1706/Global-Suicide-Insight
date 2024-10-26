# Suicide Risk Prediction Using Machine Learning
This project focuses on predicting suicide rates based on various demographic and socio-economic factors. By leveraging machine learning algorithms, the project aims to identify patterns and risk factors that contribute to suicide rates, ultimately providing insights that could aid in preventive strategies.

# Dataset
The dataset, potentially sourced from public health records or relevant social data, includes features such as:

-Age Group<br>
-Gender<br>
-Socioeconomic Indicators (GDP, income level)<br>
-Geographic Information<br>

The dataset has been preprocessed to handle missing values, encode categorical variables, and scale numerical features to improve model performance.

# Project Structure
-**Data Preprocessing:** Initial cleaning, feature scaling, and selection techniques to prepare data for analysis.<br>
-**Exploratory Data Analysis (EDA):** Visualized feature distributions and correlations to identify significant predictors of suicide rates.<br>
-**Model Training and Evaluation:** Implemented and tuned several regression models including:<br>
  1)Linear Regression<br>
  2)Decision Tree<br>
  3)Random Forest<br>
  4)Gradient Boosting<br>
  5)XGBoost<br>
  6)MLP Regressor<br>
  7)KNN Regressor<br>
-**Model Comparison:** Evaluated models based on R² score and Root Mean Squared Error (RMSE) to determine the best-performing model.<br>

# Key Results<br>
-The best model(Random Forest) achieved an R² score of **0.99** and an RMSE of **1.54**.<br>
-Ensemble methods like Random Forest and XGBoost outperformed basic regression models, indicating complex relationships within the dataset.

# Requirements
To run this project, you need the following libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
# Conclusion
The project successfully demonstrates how machine learning can be applied to predict suicide risks, highlighting significant predictors. Such insights may be valuable for public health planning and interventions.

# Future Work
Future steps could involve:<br>
-Testing on a broader dataset for generalization.<br>
-Integrating additional social and psychological indicators.<br>
-Enhancing the model with advanced neural networks or time-series analysis.<br>
