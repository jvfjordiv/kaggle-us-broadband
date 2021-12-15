contents = []
with open(os.path.join(config.OUTPUT_SCORE_PATH, "DTR.txt")) as f:
    contents.append([f.read(), 'Decision Tree Regression'])
with open(os.path.join(config.OUTPUT_SCORE_PATH, "LR.txt")) as f:
    contents.append([f.read(), 'Lineal Regression'])
with open(os.path.join(config.OUTPUT_SCORE_PATH, "RFR.txt")) as f:
    contents.append([f.read(), 'Random Forest Regression'])
contents.sort()
print("Best model: "+contents[0][1])
