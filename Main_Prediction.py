#!/usr/bin/env python
#coding: utf-8

input_max_temp = input("Please input maximum of temperature: ")
input_min_temp = input("Please input minimum of temperature: ")
input_meandew = input("Please input mean dew point: ")
input_meanhum = input("Please input mean humidity: ")
input_pressure = input("Please input mean pressure: ")

if (True):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as seabornInstance
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn import metrics

    dataset = pd.read_csv('data.csv')

    dataset.shape

    dataset.describe()

    dataset.isnull().any()

    dataset = dataset.fillna(method='ffill')

    dataset.plot(x='pressure', y='mean_temp', style='o')
    plt.title('Pressure vs Mean Temperature')
    plt.xlabel('pressure')
    plt.ylabel('mean_temp')
    plt.savefig("figures/pressure.png")
    plt.show()
    dataset.plot(x='max_temp', y='mean_temp', style='o')
    plt.title('Maximum Temperature vs Mean Temperature')
    plt.xlabel('max_temp')
    plt.ylabel('mean_temp')
    plt.savefig("figures/max_temp.png")
    plt.show()
    dataset.plot(x='min_temp', y='mean_temp', style='o')
    plt.title('Minimum Temperature vs Mean Temperature')
    plt.xlabel('min_temp')
    plt.ylabel('mean_temp')
    plt.savefig("figures/min_temp.png")
    plt.show()
    dataset.plot(x='meandew', y='mean_temp', style='o')
    plt.title('Mean Dew Point vs Mean Temperature')
    plt.xlabel('meandew')
    plt.ylabel('mean_temp')
    plt.savefig("figures/meandew.png")
    plt.show()
    dataset.plot(x='meanhum', y='mean_temp', style='o')
    plt.title('Mean Humidity vs Mean Temperature')
    plt.xlabel('meanhum')
    plt.ylabel('mean_temp')
    plt.savefig("figures/meanhum.png")
    plt.show()

    X = dataset[['pressure', 'max_temp', 'min_temp', 'meandew', 'meanhum']]
    y = dataset['mean_temp']

    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    seabornInstance.distplot(dataset['mean_temp'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("Linear Regression Prediction: ")

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
    coeff_df.sort_values(by='Coefficient', ascending=False)


    pos_coeffs_df = coeff_df[(coeff_df['Coefficient'] >= 0)].sort_values(by='Coefficient', ascending=False)
    pos_coeffs_df = coeff_df[(coeff_df['Coefficient'] < 0)].sort_values(by='Coefficient', ascending=True)

    y_pred = regressor.predict(X_test)



    import seaborn as sns

    g = sns.regplot(x=y_pred, y=y_test, fit_reg=True)
    g.set(xlabel='Predicted Mean Temperature', ylabel='Actual Mean Temperature', title='Model Predictions')
    plt.title('Regression Plot for Actual vs Predicted Values')

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)

    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig("figures/linear_regression_comparison.png")
    plt.show()


    # R2 for train and test data
    R2_reg_train = regressor.score(X_train, y_train)
    R2_reg_test = regressor.score(X_test, y_test)
    print('R squared for train data is: %.3f' % (R2_reg_train))
    print('R squared for test data is: %.3f' % (R2_reg_test))


    from math import sqrt

    RMSE_reg_train = sqrt(np.mean((y_train - regressor.predict(X_train)) ** 2))
    RMSE_reg_test = sqrt(np.mean((y_test - regressor.predict(X_test)) ** 2))
    print('Root mean squared error for train data is: %.3f' % (RMSE_reg_train))
    print('Root mean sqaured error for test data is: %.3f' % (RMSE_reg_test))

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


    estimated_temp = regressor.predict([[float(input_pressure),float(input_max_temp),float(input_min_temp),float(input_meandew),float(input_meanhum)]])

    print("The expected mean of temperature is", estimated_temp)

    print(" ")
    print("K-Nearest Neighbors Prediction: ")

    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(X_train, y_train)

    pred_knn = knn.predict(X_test)
    y_pred = knn.predict(X_test)



    import seaborn as sns

    g = sns.regplot(x=y_pred, y=y_test, fit_reg=True)
    g.set(xlabel='Predicted Mean Temperature', ylabel='Actual Mean Temperature', title='Model Predictions')
    plt.title('Regression Plot for Actual vs Predicted Values')


    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)


    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig("figures/KNN_comparison.png")
    plt.show()


    # R2 for train and test data
    R2_reg_train = knn.score(X_train, y_train)
    R2_reg_test = knn.score(X_test, y_test)
    print('R squared for train data is: %.3f' % (R2_reg_train))
    print('R squared for test data is: %.3f' % (R2_reg_test))


    from math import sqrt

    RMSE_reg_train = sqrt(np.mean((y_train - knn.predict(X_train)) ** 2))
    RMSE_reg_test = sqrt(np.mean((y_test - knn.predict(X_test)) ** 2))
    print('Root mean squared error for train data is: %.3f' % (RMSE_reg_train))
    print('Root mean sqaured error for test data is: %.3f' % (RMSE_reg_test))


    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


    estimated_temp = knn.predict([[float(input_pressure),float(input_max_temp),float(input_min_temp),float(input_meandew),float(input_meanhum)]])
    print ("The expected mean of temperature is", estimated_temp)


    print(" ")
    print("Random Forest Regression Prediction: ")

    rf = RandomForestRegressor(random_state=5, n_estimators=20)
    rf.fit(X_train, y_train)


    pred_rf = rf.predict(X_test)
    y_pred = rf.predict(X_test)


    import seaborn as sns

    g = sns.regplot(x=y_pred, y=y_test, fit_reg=True)
    g.set(xlabel='Predicted Mean Temperature', ylabel='Actual Mean Temperature', title='Model Predictions')
    plt.title('Regression Plot for Actual vs Predicted Values')


    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)


    df1.plot(kind='bar', figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.savefig("figures/random_forest_comparison.png")
    plt.show()


    # R2 for train and test data
    R2_reg_train = rf.score(X_train, y_train)
    R2_reg_test = rf.score(X_test, y_test)
    print('R squared for train data is: %.3f' % (R2_reg_train))
    print('R squared for test data is: %.3f' % (R2_reg_test))


    from math import sqrt

    RMSE_reg_train = sqrt(np.mean((y_train - rf.predict(X_train)) ** 2))
    RMSE_reg_test = sqrt(np.mean((y_test - rf.predict(X_test)) ** 2))
    print('Root mean squared error for train data is: %.3f' % (RMSE_reg_train))
    print('Root mean sqaured error for test data is: %.3f' % (RMSE_reg_test))


    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


    estimated_temp = rf.predict([[float(input_pressure),float(input_max_temp),float(input_min_temp),float(input_meandew),float(input_meanhum)]])
    print ("The expected mean of temperature is", estimated_temp)
