# water-quality-potability-

                                        Water Quality Machine Learning Model Evaluation Report

      1. Dataset Overview
The dataset used for this analysis is sourced from [Kaggle](https://www.kaggle.com/adityakadiwal/…) and contains information about water potability. The dataset includes several features that help determine if the water is potable or not.

       2. Data Preprocessing
1.  Data Cleaning:
    - Checked for missing values and handled them appropriately.
    - Scaled the features using MinMaxScaler to normalize the data.

2. Feature Selection:
    - Selected relevant features that contribute significantly to the target variable 'Potability'.
    - Here we applied techniques such as finding the correlation , to figure out the columns that       are highly correlated to each other so that they can be dropped.

3.  Model Training
Several machine learning models were trained and evaluated. Here are the details of each model along with their parameters:

 Logistic Regression
```python
model = LogisticRegression(solver='liblinear', multi_class='auto')
params = {'C': [1,5,10,15,20,25,30]}
```

       b)  Random Forest Classifier
```python
model = RandomForestClassifier()
params = {'n_estimators': [60,70,80 ,100,200,300,400,500,600,700]}
```

     c) Support Vector Machine (SVM)
```python
model = svm.SVC(gamma='auto')
params = {'C': [1, 10, 20,30,50], 'kernel': ['rbf', 'linear']}
```

      d) Gradient Boosting Classifier
```python
model = GradientBoostingClassifier()
params = {
    'n_estimators': [60,70,80, 100, 200,300,400,500,600,700],
    'learning_rate': [0.01, 0.1, 0.5]
}
```

      e)  Decision Tree Classifier
```python
model = DecisionTreeClassifier()
params = {'criterion': ['gini', 'entropy']}
```
      f) Naive Bayes (Multinomial and Gaussian)
```python
model = MultinomialNB()
params = {}

model = GaussianNB()
params = {}
```
     

     4. Model Evaluation
Each model was evaluated using GridSearchCV to ensure robust evaluation metrics. The models were compared based on their accuracy, precision, recall, F1 score, and ROC-AUC score.
Based on the evaluation metrics, the RandomForestClassifier performed the best, achieving the highest accuracy, F1 score, and ROC-AUC score.

Conclusion
The RandomForestClassifier was selected as the best model for predicting water potability. This model showed superior performance in balancing true positives and false positives, which is crucial for ensuring accurate predictions.

Next Steps
1. Model Tuning: Further tuning of the RandomForestClassifier parameters could potentially enhance performance.
2. Feature Engineering: Additional feature engineering might uncover new insights and further improve model accuracy.

