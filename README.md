# Hepatitis-Survival-ML-Analysis

This project leverages machine learning to predict hepatitis patient survival outcomes. Using Scikit-Learn, it analyzes classifiers such as LinearSVC, DecisionTree, RandomForest, and KNN, recording performance metrics like accuracy and F1-score. Hyperparameter tuning on Random Forest and stacking ensemble techniques are explored to boost performance. The analysis identifies critical features influencing survival, and stacking achieves the highest accuracy. Recommendations include enhancing preprocessing, increasing classifier diversity, and employing advanced ensembles like XGBoost for future improvements.

# Key Features
1.	Models Used:
•	LinearSVC: A linear classification model for binary tasks.
•	DecisionTreeClassifier: A tree-based model dividing data into significant feature-based splits.
•	RandomForestClassifier: An ensemble of decision trees enhancing stability and reducing overfitting.
•	K-Nearest Neighbors (KNN): A distance-based lazy learner.
2.	Hyperparameter Tuning:
•	Random Forest hyperparameters tuned using random search:
•	Number of estimators: 200
•	Maximum depth: 10
•	Minimum samples required for a split: 5
3.	Ensemble Learning:
•	Predictions combined using stacking with a Multilayer Perceptron (MLP) as the meta-classifier.
4.	Performance Metrics:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
5.	Feature Importance:
•	Random Forest’s feature_importances_ property used to identify the top 5 features affecting survival prediction.

# Technologies Used
•	Programming Language: Python
•	Libraries:
•	Scikit-Learn for machine learning algorithms
•	RandomizedSearchCV for hyperparameter tuning
•	Data preprocessing tools for handling numerical and categorical variables
•	Platform: Jupyter Notebook

# Key Insights
1.	Default Classifier Results:
•	DecisionTreeClassifier performed best among individual models (default mode) with an accuracy of 81.8%.
•	Random Forest performed worse after tuning, suggesting the default parameters were already optimal.
2.	Hyperparameter Tuning:
•	Tuning Random Forest did not improve performance due to dataset characteristics (e.g., limited variability or balance).
3.	Stacking Results:
•	Achieved the highest accuracy (90.9%) and outperformed individual models, showcasing the potential of ensemble techniques.
4.	Feature Importance:
•	Identified critical features aligning with clinical insights, enhancing the model’s interpretability.
5.	Challenges:
•	Limited diversity among classifiers in the ensemble reduced stacking effectiveness.
•	The meta-learner (MLP) might not have been the optimal choice for this dataset.

# Recommendations
•	Increase classifier diversity in stacking with models like Gradient Boosting or non-linear SVMs.
•	Improve preprocessing by addressing data imbalances, handling missing values, and applying scaling.
•	Explore advanced ensemble techniques such as AdaBoost and XGBoost for potential performance gains.
