corr = df.corr()[config.TARGET_ATTR].sort_values(ascending=False)
best = corr[corr > config.FINAL_THRESHOLD]
X = []
for i in range(1,len(best)):
    X.append(best.index[i])

X = df.loc[:,X]
Y = df.loc[:,[config.TARGET_ATTR]]

X_train, X_test, y_train, y_test = skl_ms.train_test_split(X, Y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)