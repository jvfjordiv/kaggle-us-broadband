regressor = skl_tree.DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_prediction_dt = regressor.predict(X_test)
y_prediction_dt = np.round(y_prediction_dt)
plt.scatter(y_test,y_prediction_dt)
plt.title("Prediction with Decision Tree Regression")
plt.xlabel("Real percent")
plt.ylabel("Predicted")
plt.savefig(os.path.join(config.OUTPUT_GRAPH_PATH, "regression_DTR.png"))
plt.close()

joblib.dump(
    regressor,
    os.path.join(config.OUTPUT_PATH, "DTR.bin")
)

text_file = open(os.path.join(config.OUTPUT_SCORE_PATH, "DTR.txt"), "w")
text_file.write(str(math.sqrt(skl_metrics.mean_squared_error(y_test, y_prediction_dt))))
text_file.close()