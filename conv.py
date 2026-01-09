import pandas as pd

# Load Stata file and save as CSV
df = pd.read_stata('child_food.dta')
df.to_csv('data.csv', index=False)
print("Completed: Saved as data.csv")