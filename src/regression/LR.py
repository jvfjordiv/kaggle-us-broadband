regressor = skl_lm.LinearRegression()
regressor.fit(X_train, y_train)
y_prediction_lr = regressor.predict(X_test)
y_prediction_lr = np.round(y_prediction_lr)
plt.scatter(y_test,y_prediction_lr)
plt.title("Prediction with Linear Regression")
plt.xlabel("Real percent")
plt.ylabel("Predicted")
plt.savefig(os.path.join(config.OUTPUT_GRAPH_PATH, "regression_LR.png"))
plt.close()

joblib.dump(
    regressor,
    os.path.join(config.OUTPUT_PATH, "LR.bin")
)

text_file = open(os.path.join(config.OUTPUT_SCORE_PATH, "LR.txt"), "w")
text_file.write(str(math.sqrt(skl_metrics.mean_squared_error(y_test, y_prediction_lr))))
text_file.close()