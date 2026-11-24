import pandas as pd
import csv
 
data = pd.read_csv("ssgsea_scores.csv",index_col = 0)
#next I need to get all ssgsea scores from each of the columns, add them up, and normalize each one
#I want to add up all the scores for each sample
all_scores = data.sum(axis=1)
#I want to normalized each score
for col in data.columns:
    data[col] = data[col]/all_scores

savescores = data.to_csv("ssgseanomrscore.csv",index = True)

    
#next I need to write the normalized scores
