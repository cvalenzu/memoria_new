import glob

import pandas as pd
import sys
import os 

source = sys.argv[1] #"canela"
#files = glob.glob("../results/params_lstm.back/*{}*".format(source))
files = glob.glob("../results/params_lstm/*{}*".format(source))

dfs = pd.DataFrame()
for file in files:
	df = pd.read_csv(file,index_col = 0)
	dfs = pd.concat([dfs,df])


dfs = dfs[dfs.activation != "linear"]
result = []

for lags, df in dfs.groupby(["input_dim","timesteps"]):
	best = df.sort_values("validation_score").iloc[0]
	result.append(best)

result = pd.DataFrame(result)
print(result)
base= "../results/best_params/"
os.makedirs(base, exist_ok=True)
result.to_csv(base+"{}_best_params_by_lag.csv".format(source))
