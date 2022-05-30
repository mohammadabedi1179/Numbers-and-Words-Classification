#python3
import sys
from load import Load
import numpy as np
import pandas as pd

class Inference():

    def __init__(self, path : str):
        self.path = path
        
    def getScores(self, images):
        
        load_model = Load()
        model = load_model.load_model(path=self.path)
        images = load_model.preprocess_image(images)
        print(images.shape)
        predictions = model.predict(images)
        predictions = np.round(predictions, decimals=3)
        numeric_preds = predictions
        alphabetic_preds = 1 - predictions
        scores = {'Alphabetic probability' : list(alphabetic_preds), 'Numeric probability' : list(numeric_preds)}
        scores_df = pd.DataFrame(scores)
        scores_df.to_html("scores.html")

        return predictions