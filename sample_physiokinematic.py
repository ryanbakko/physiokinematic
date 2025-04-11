import sys
import pickle
import pandas as pd
import pymc as pm
import importlib
from physiokinematic import model

# Get index from command line
index = int(sys.argv[1])

# Load data
hii_data = pd.read_csv("hii_data.csv")
data = hii_data.iloc[index].copy()

# Build model
my_model = pkmodel(data)

# Sample model
with my_model:
    trace = pm.sample(1000, tune=1000, target_accept=0.95)

# Save trace
with open(f"trace_{index}.pkl", "wb") as f:
    pickle.dump(trace, f)
