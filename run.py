import argparse
import os
import config.config as config

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as skl_ms
import sklearn.linear_model as skl_lm
import sklearn.tree as skl_tree
import sklearn.metrics as skl_metrics
import sklearn.ensemble as skl_ensemble
import math
import joblib

def run(gengraph,regression):
    exec(open("./src/generate_features.py").read())
    if (gengraph != 'no'):
        exec(open("./src/graphics_gen.py").read())
    exec(open("./src/train_model.py").read())
    if (regression == 'LR' or regression == 'lr'):
        exec(open("./src/regression/LR.py").read())
    elif (regression == 'DTR' or regression == 'dtr'):
        exec(open("./src/regression/DTR.py").read())
    elif (regression == 'RFR' or regression == 'rfr'):
        exec(open("./src/regression/RFR.py").read())
    else:
        exec(open("./src/regression/LR.py").read())
        exec(open("./src/regression/DTR.py").read())
        exec(open("./src/regression/RFR.py").read())
        exec(open("./src/score.py").read())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gengraph', type=str)
    parser.add_argument('--regression', type=str)
    args = parser.parse_args()
    run(
        gengraph=args.gengraph,
        regression=args.regression
    )