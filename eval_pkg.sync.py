import numpy as np
from profanity_filter import ProfanityFilter
from profanity_protector import predict, predict_prob

import pandas as pd

# %%

df = pd.read_csv("user_inputs.csv", sep=",")
print(len(df))
test_texts = df["translated"].astype(str)
preds = predict(test_texts)
df[preds == 1]

# %%
print("Profanity filter")
pf = ProfanityFilter()
preds = np.array([(0 if pf.is_clean(x) else 1) for x in test_texts])
print(preds)
df[preds == 1]

# %%
data = pd.read_csv('./data/test_data.csv')
texts = data['text'].astype(str)
y = data['is_offensive']
