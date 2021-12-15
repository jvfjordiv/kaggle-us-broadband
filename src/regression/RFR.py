regressor = skl_ensemble.RandomForestRegressor(n_estimators=config.N_ESTIMATORS,random_state = config.RANDOM_STATE)
regressor.fit(X_train, y_train)
y_prediction_rf = regressor.predict(X_test)
y_prediction_rf = np.round(y_prediction_rf)
plt.scatter(y_test,y_prediction_rf)
plt.title("Prediction with Random Forest Regression")
plt.xlabel("Real percent")
plt.ylabel("Predicted")
plt.savefig(os.path.join(config.OUTPUT_GRAPH_PATH, "regression_RFR.png"))
plt.close()

joblib.dump(
    regressor,
    os.path.join(config.OUTPUT_PATH, "RFR.bin")
)

text_file = open(os.path.join(config.OUTPUT_SCORE_PATH, "RFR.txt"), "w")
text_file.write(str(math.sqrt(skl_metrics.mean_squared_error(y_test, y_prediction_rf))))
text_file.close()