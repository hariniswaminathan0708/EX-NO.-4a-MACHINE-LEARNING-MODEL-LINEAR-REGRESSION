# EX-NO.-4a-MACHINE-LEARNING-MODEL-LINEAR-REGRESSION
## AIM 
To predict house prices using regression models and compare the performance of different machine learning regression models based on RMSE, MAE, and R².
1.Machine Learning:Machine Learning is used to learn patterns from existing data and make predictions. 
•	Regression is a supervised learning technique used to predict continuous numerical values. 
•	In this experiment, regression models are used to predict the price of a house. 
•	The dataset contains house-related features such as: 
o	square_feet 
o	num_rooms 
o	age 
o	distance_to_city(km) 
•	The target variable is: 
o	price 
## DATASET DESCRIPTION
•	Dataset: House Price Dataset 
•	Problem: Predict house price. 
•	Features (X): 
o	square_feet – size of the house. 
o	num_rooms – number of rooms. 
o	age – age of the house in years. 
o	distance_to_city(km) – distance from the city centre. 
•	Target (y): 
o	price – continuous house price. 
## PROBLEM STATEMENT
•	Develop a machine learning model to predict house prices. 
•	Use house characteristics as input. 
•	Train different regression models. 
•	Compare their prediction performance. 
•	Select the better-performing model based on evaluation metrics. 
## REGRESSION MODELS USED
The uploaded notebook compares the following models:
1.	Linear Regression 
2.	Ridge Regression 
3.	Lasso Regression 
4.	ElasticNet Regression 
5.	Polynomial Regression 
6.	Decision Tree Regressor 
7.	Random Forest Regressor 
8.	Gradient Boosting Regressor 
9.	Support Vector Regressor (SVR) 
10.	K-Nearest Neighbors (KNN) Regressor 
## PROCEDURE
    1.Import the required Python libraries for data processing, visualization, machine learning models, and model evaluation.
    2.Load the house price dataset from the specified CSV file using Pandas.
    3.Display the first five records of the dataset.
    4.Display the dataset information, shape, summary statistics, and check for missing values.
    5.Perform Exploratory Data Analysis (EDA) by studying the distribution of numerical features using histograms.
    6.Perform correlation analysis using a correlation heatmap to understand the relationship between the features and house price.
    7.Use scatter plots to study the relationship between individual features and house price.
    8.Detect outliers using boxplots.
    9.Remove extremely low and extremely high house prices using the 1st and 99th percentiles.
    10.Define the independent variables as square_feet, num_rooms, age, and distance_to_city(km), and define price as the target variable.
    11.Split the dataset into training and testing sets using an 80:20 ratio.
    12.Apply StandardScaler to scale the training and testing features.
    13.Create a baseline model that predicts the mean house price.
    14.Train different regression models including Linear Regression, Ridge, Lasso, ElasticNet, Polynomial Regression, Decision Tree, Random Forest,Gradient Boosting, SVR, and KNN.
    15.Evaluate all models using RMSE, MAE, and R² metrics.
    16.Compare the performance of the regression models based on their evaluation metrics.
    17.Plot actual versus predicted house prices.
    18.Perform residual analysis to study prediction errors.
    19.Calculate Random Forest and Gradient Boosting feature importance.
    20.Plot the RMSE comparison graph for all regression models.
## PROGRAM
    

	from google.colab import drive
	
	drive.mount('/content/drive')
	
	import pandas as pd
	
	df = pd.read_csv('/content/drive/My Drive/house_datasets.csv')
	
	df.head()

	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns

	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler, PolynomialFeatures

	from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
	from sklearn.svm import SVR
	from sklearn.neighbors import KNeighborsRegressor

	from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

	print("Dataset Information:")
	df.info()
	
	print("\nDataset Shape:")
	print(df.shape)
	
	print("\nStatistical Summary:")
	print(df.describe())
	
	print("\nMissing Values:")
	print(df.isnull().sum())
	
	df.hist(figsize=(12, 8))
	plt.tight_layout()
	plt.show()
	
	plt.figure(figsize=(8, 6))
	
	sns.heatmap(
	df.corr(numeric_only=True),
	annot=True,
	cmap='coolwarm'
	)
	
	plt.title("Correlation Heatmap")
	plt.show()
	
	features = [
	'square_feet',
	'num_rooms',
	'age',
	'distance_to_city(km)'
	]
	
	for feature in features:

    plt.figure(figsize=(6, 4))

    plt.scatter(
        df[feature],
        df['price']
    )

    plt.xlabel(feature)
    plt.ylabel('Price')
    plt.title(feature + ' vs Price')

    plt.show()

	plt.figure(figsize=(6, 4))
	
	sns.boxplot(
	    y=df['price']
	)
	
	plt.title("Price Outliers")
	plt.show()
	
	
	
	lower_limit = df['price'].quantile(0.01)
	upper_limit = df['price'].quantile(0.99)
	
	df = df[
	    (df['price'] >= lower_limit) &
	    (df['price'] <= upper_limit)
	]
	
	print("Shape after removing outliers:")
	print(df.shape)
	
	
	
	X = df[
	    [
	        'square_feet',
	        'num_rooms',
	        'age',
	        'distance_to_city(km)'
	    ]
	]
	
	y = df['price']
	
	print("\nFeatures:")
	print(X.head())
	
	print("\nTarget:")
	print(y.head())
	
	
	
	X_train, X_test, y_train, y_test = train_test_split(
	    X,
	    y,
	    test_size=0.2,
	    random_state=42
	)
	
	print("\nTraining Data:")
	print(X_train.shape)
	
	print("\nTesting Data:")
	print(X_test.shape)
	
	
	
	scaler = StandardScaler()
	
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)
	
	print("\nFeature Scaling Completed")
	
	
	
	baseline_prediction = np.full(
	    len(y_test),
	    y_train.mean()
	)
	
	baseline_rmse = np.sqrt(
	    mean_squared_error(
	        y_test,
	        baseline_prediction
	    )
	)
	
	baseline_mae = mean_absolute_error(
	    y_test,
	    baseline_prediction
	)
	
	baseline_r2 = r2_score(
	    y_test,
	    baseline_prediction
	)
	
	print("\nBaseline Model")
	print("RMSE:", baseline_rmse)
	print("MAE:", baseline_mae)
	print("R2:", baseline_r2)
	
	
	
	linear_model = LinearRegression()
	
	linear_model.fit(
	    X_train_scaled,
	    y_train
	)
	
	linear_pred = linear_model.predict(
	    X_test_scaled
	)
	
	
	ridge_model = Ridge()
	
	ridge_model.fit(
	    X_train_scaled,
	    y_train
	)
	
	ridge_pred = ridge_model.predict(
	    X_test_scaled
	)
	
	
	
	lasso_model = Lasso()
	
	lasso_model.fit(
	    X_train_scaled,
	    y_train
	)
	
	lasso_pred = lasso_model.predict(
	    X_test_scaled
	)
	
	
	elastic_model = ElasticNet()
	
	elastic_model.fit(
	    X_train_scaled,
	    y_train
	)
	
	elastic_pred = elastic_model.predict(
	    X_test_scaled
	)
	
	
	
	poly = PolynomialFeatures(
	    degree=2
	)
	
	X_train_poly = poly.fit_transform(
	    X_train_scaled
	)
	
	X_test_poly = poly.transform(
	    X_test_scaled
	)
	
	poly_model = LinearRegression()
	
	poly_model.fit(
	    X_train_poly,
	    y_train
	)
	
	poly_pred = poly_model.predict(
	    X_test_poly
	)
	
	
	dt_model = DecisionTreeRegressor(
	    random_state=42
	)
	
	dt_model.fit(
	    X_train,
	    y_train
	)
	
	dt_pred = dt_model.predict(
	    X_test
	)
	
	
	
	rf_model = RandomForestRegressor(
	    n_estimators=100,
	    random_state=42
	)
	
	rf_model.fit(
	    X_train,
	    y_train
	)
	
	rf_pred = rf_model.predict(
	    X_test
	)
	
	
	
	gb_model = GradientBoostingRegressor(
	    n_estimators=100,
	    learning_rate=0.1,
	    random_state=42
	)
	
	gb_model.fit(
	    X_train,
	    y_train
	)
	
	gb_pred = gb_model.predict(
	    X_test
	)
	
	
	
	svr_model = SVR(
	    kernel='rbf',
	    C=100,
	    gamma=0.1,
	    epsilon=0.1
	)
	
	svr_model.fit(
	    X_train_scaled,
	    y_train
	)
	
	svr_pred = svr_model.predict(
	    X_test_scaled
	)
	
	
	
	knn_model = KNeighborsRegressor(
	    n_neighbors=5
	)
	
	knn_model.fit(
	    X_train_scaled,
	    y_train
	)
	
	knn_pred = knn_model.predict(
	    X_test_scaled
	)



	def evaluate_model(name, y_true, y_pred):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }


	results = []
	
	results.append(
	    evaluate_model(
	        'Linear Regression',
	        y_test,
	        linear_pred
	    )
	)
	
	results.append(
	    evaluate_model(
	        'Ridge Regression',
	        y_test,
	        ridge_pred
	    )
	)
	
	results.append(
	    evaluate_model(
	        'Lasso Regression',
	        y_test,
	        lasso_pred
	    )
	)
	
	results.append(
	    evaluate_model(
	        'ElasticNet',
	        y_test,
	        elastic_pred
	    )
	)
	
	results.append(
	    evaluate_model(
	        'Polynomial Regression',
	        y_test,
	        poly_pred
	    )
	)
	
	results.append(
	    evaluate_model(
	        'Decision Tree',
        y_test,
        dt_pred
    )
	)
	
	results.append(
	    evaluate_model(
	        'Random Forest',
	        y_test,
	        rf_pred
	    )
	)
	
	results.append(
	    evaluate_model(
	        'Gradient Boosting',
	        y_test,
	        gb_pred
	    )
	)
	
	results.append(
	    evaluate_model(
	        'SVR',
	        y_test,
	        svr_pred
	    )
	)
	
	results.append(
	    evaluate_model(
        'KNN',
        y_test,
        knn_pred
    )
	)
	
	
	results_df = pd.DataFrame(results)
	
	results_df = results_df.sort_values(
	    by='RMSE'
	)
	
	print("\nMODEL COMPARISON")
	print(results_df)
	
	
	
	predictions = {
	    'Linear Regression': linear_pred,
	    'Ridge': ridge_pred,
	    'Lasso': lasso_pred,
	    'ElasticNet': elastic_pred,
	    'Polynomial': poly_pred,
    'Decision Tree': dt_pred,
    'Random Forest': rf_pred,
    'Gradient Boosting': gb_pred,
    'SVR': svr_pred,
    'KNN': knn_pred
	}
	
	for name, prediction in predictions.items():

    plt.figure(figsize=(6, 4))

    plt.scatter(
        y_test,
        prediction
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

    plt.title(
        name + " - Actual vs Predicted"
    )

    plt.show()


	for name, prediction in predictions.items():

    residuals = y_test - prediction

    plt.figure(figsize=(6, 4))

    plt.scatter(
        prediction,
        residuals
    )

    plt.axhline(
        y=0,
        linestyle='--'
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")

    plt.title(
        name + " - Residual Plot"
    )

    plt.show()



	importance = rf_model.feature_importances_
	
	feature_importance = pd.DataFrame({
	    'Feature': X.columns,
	    'Importance': importance
	})
	
	feature_importance = feature_importance.sort_values(
	    by='Importance',
	    ascending=False
	)
	
	print("\nRandom Forest Feature Importance:")
	print(feature_importance)
	
	plt.figure(figsize=(8, 5))
	
	plt.bar(
	    feature_importance['Feature'],
	    feature_importance['Importance']
	)
	
	plt.xlabel("Features")
	plt.ylabel("Importance")
	plt.title("Random Forest Feature Importance")
	
	plt.xticks(rotation=45)
	
	plt.tight_layout()
	plt.show()
	
	
	gb_importance = gb_model.feature_importances_
	
	gb_feature_importance = pd.DataFrame({
	    'Feature': X.columns,
	    'Importance': gb_importance
	})
	
	gb_feature_importance = gb_feature_importance.sort_values(
	    by='Importance',
	    ascending=False
	)
	
	print("\nGradient Boosting Feature Importance:")
	print(gb_feature_importance)
	
	
	
	plt.figure(figsize=(10, 6))
	
	plt.bar(
	    results_df['Model'],
	    results_df['RMSE']
	)
	
	plt.xlabel("Models")
	plt.ylabel("RMSE")
	
	plt.title(
	    "RMSE Comparison of Regression Models"
	)
	
	plt.xticks(
	    rotation=45,
	    ha='right'
	)
	
	plt.tight_layout()
	plt.show()
		
## OUTPUT
<img width="705" height="615" alt="image" src="https://github.com/user-attachments/assets/2aeca6eb-18d0-47ee-b36f-a764bb7427dc" />
<img width="762" height="747" alt="image" src="https://github.com/user-attachments/assets/c8d67fb8-75d7-44cd-8e36-cbf2bfa96561" />
<img width="725" height="375" alt="image" src="https://github.com/user-attachments/assets/8fd2f0ad-8f8e-452f-82f1-9c87b0244633" />
<img width="761" height="682" alt="image" src="https://github.com/user-attachments/assets/5869a816-2245-416e-9648-96b844dc2d15" />
<img width="726" height="325" alt="image" src="https://github.com/user-attachments/assets/a8282bc7-657c-46b5-b19c-c1e6d8189b44" />
<img width="802" height="647" alt="image" src="https://github.com/user-attachments/assets/78b27508-b8ad-4761-8a03-8e6455ee5dca" />
<img width="722" height="467" alt="image" src="https://github.com/user-attachments/assets/44121229-9999-48e4-850f-3165f733b3c6" />
<img width="787" height="517" alt="image" src="https://github.com/user-attachments/assets/0ddc7ba8-0058-4091-9a39-aeba8a9dc6d1" />
<img width="756" height="497" alt="image" src="https://github.com/user-attachments/assets/af7d8f19-dbe8-44fb-a4f9-a981dcb24545" />
<img width="785" height="492" alt="image" src="https://github.com/user-attachments/assets/5ad6ed9c-3776-44f2-a573-39f90a8d2cc8" />
<img width="777" height="482" alt="image" src="https://github.com/user-attachments/assets/88d9ae49-c64a-4ac2-b753-9c4fc9d3a95b" />
<img width="730" height="750" alt="image" src="https://github.com/user-attachments/assets/6c016a10-491c-4d31-8418-5a4a5b53cd60" />
<img width="837" height="802" alt="image" src="https://github.com/user-attachments/assets/9c6e4d6e-51ef-4e18-99eb-35b70a20cf3f" />
<img width="792" height="517" alt="image" src="https://github.com/user-attachments/assets/ada625a6-cf00-4c7b-b8c3-8e82473c9ad5" />
<img width="802" height="542" alt="image" src="https://github.com/user-attachments/assets/01c6740f-2223-4de8-b295-e7ab542f0a78" />
<img width="735" height="492" alt="image" src="https://github.com/user-attachments/assets/e924d223-4ad5-4128-a6c9-031570b1c083" />
<img width="727" height="462" alt="image" src="https://github.com/user-attachments/assets/e253ebff-a552-415a-b865-249fc2a92178" />
<img width="722" height="502" alt="image" src="https://github.com/user-attachments/assets/9edb3e96-6efc-4ae7-8b4c-781975c351d1" />
<img width="727" height="497" alt="image" src="https://github.com/user-attachments/assets/50e11093-0074-447f-8f15-95f30ef2585e" />
<img width="732" height="496" alt="image" src="https://github.com/user-attachments/assets/21f94be3-5369-4025-acb6-60535b7df212" />
<img width="736" height="477" alt="image" src="https://github.com/user-attachments/assets/c027426e-eee7-460f-b1e0-0a537e31bdab" />
<img width="717" height="481" alt="image" src="https://github.com/user-attachments/assets/0092c22f-6abb-406d-baaa-65a97633cde8" />
<img width="711" height="472" alt="image" src="https://github.com/user-attachments/assets/6601a890-bd3e-4c48-b062-9276139c6a35" />
<img width="722" height="500" alt="image" src="https://github.com/user-attachments/assets/9d787e05-94b8-4700-892c-d1c7f74da388" />
<img width="717" height="497" alt="image" src="https://github.com/user-attachments/assets/edcb0548-ff8b-4a94-a297-5f5783133425" />
<img width="730" height="487" alt="image" src="https://github.com/user-attachments/assets/fb6a28f3-491e-434c-ba80-468928b34bb3" />
<img width="727" height="472" alt="image" src="https://github.com/user-attachments/assets/bb833dd4-7e7d-4aa7-be46-6337db1e7bfc" />
<img width="692" height="510" alt="image" src="https://github.com/user-attachments/assets/3dbd1208-a7db-417c-a2e0-96470dcd6f02" />
<img width="1027" height="722" alt="image" src="https://github.com/user-attachments/assets/e1cfde0b-36e8-4184-8ed5-284dd00579c0" />
<img width="1245" height="711" alt="image" src="https://github.com/user-attachments/assets/b4fa1afb-9646-4e26-b34a-180332b0cbdc" />

	
## CONCLUSION
Thus, Linear Regression and other regression models were successfully applied for house price prediction, and their performance was compared using standard regression evaluation metrics.

