import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/danielhernandez/Downloads/Watershed_Characteristics.csv', index_col=0)
###############
# define the feature set X and the target variable y
# 'Annual Avg. River Flow' is the column we want to predict
drop = ['Annual Avg. River Flow', 'Runoff Ratio', 'Basin ID']

def remove_outliers_iqr(df):
    df_cleaned = df.copy()
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        df_cleaned = df_cleaned[(df_cleaned[column] >= (Q1 - 1.5 * IQR)) & (df_cleaned[column] <= (Q3 + 1.5 * IQR))]
    return df_cleaned

df_cleaned_iqr = remove_outliers_iqr(data)

X = df_cleaned_iqr.drop(drop, axis=1)  # Features
y = df_cleaned_iqr['Annual Avg. River Flow']  # Target variable


def random_forest(x, y2):
    # split the data into training and testing sets
    #0.2 means that 20 percent will be used for testing and
    #80 percent will be used for training the model
    #42 because for consistent basis for comparing models or approaches
    X_train, X_test, y_train, y_test = train_test_split(x, y2, test_size=0.2, random_state=42)

    #randomForestRegressor is used to train the model, make predictions
    #and then used later to evaluate
    # n_estimators=100 sets the number of trees in the forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    #train the model on the training data
    model.fit(X_train, y_train)

    #use the trained model to make predictions on the test set
    y_pred = model.predict(X_test)

    #convert the predictions to a DataFrame for easier comparison
    #y_pred_df = pd.DataFrame(y_pred, columns=['Predicted'])

    #combine the actual and predicted values into a single DataFrame for comparison
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(f'Results for {y2.name}')

    #print the DataFrame with actual and predicted values
    print(comparison_df)

    #evaluate the model's performance using Mean Squared Error and R^2 Score
    mse = mean_squared_error(y_test, y_pred)  # mean Squared Error
    r2 = r2_score(y_test, y_pred)              # calculate R^2 Score

    #print the evaluation metrics
    print(f'Results for {y2.name}')

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    x_name = x.columns[0]
    #plot a scatter plot to visualize the relationship between actual and predicted values
    plt.scatter(comparison_df['Actual'], comparison_df['Predicted'])
    #plt.xlabel('Actual Annual Avg. River Flow (mm)')
    plt.xlabel(f'Actual {y2.name} (mm)')
    #plt.ylabel('Predicted Annual Avg. River Flow (mm)')
    plt.ylabel(f'Predicted {y2.name} (mm)')
    plt.title('Actual vs. Predicted Annual Avg. River Flow')
    plt.show()

    print(f'Results for {y2.name}')
    importances = model.feature_importances_
    feature_names = x.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print(importance_df.sort_values(by='Importance', ascending=False))

random_forest(X, y)