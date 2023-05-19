import pandas as pd
from pathlib import Path
import Levenshtein

def eval_csv(path):
    df = pd.read_csv(path)
    df_gt = df['GT Plate']
    df_sr = df['OCR SR Prediction']
    
    return df_gt, df_sr

def eval_char(gt, sr):
    count = 0
    six = 0
    five = 0
    one = 0
    
    for a, b in zip(gt, sr):
        
        errors = Levenshtein.distance(a, b)
        if errors <= 1:
            six+=1
        
        if errors <= 6:
            one+=1
        
        if errors <= 2:
            five+=1
            
        if errors >0:
            print(a, b, errors)
        count+= errors
        
    total = len(gt)*7
    print(total-count)
    print(total)
    print(six, five, one)
    print('Total correct characters: {}%'.format(((total-count)/(total))*100))


if __name__ == "__main__":
    path = Path(r'C:\Users\VALFRIDE-DESKTOP\Desktop\testing\proposed1\eval.csv')
    df_gt, df_sr = eval_csv(path)
    eval_char(df_gt, df_sr)