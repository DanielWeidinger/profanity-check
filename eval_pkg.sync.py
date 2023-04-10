from profanity_check import predict, predict_prob
import pandas as pd

# %%

df = pd.read_csv("user_inputs.csv", sep=",")
print(len(df))
test_texts = df["translated"].astype(str)
preds = predict(test_texts)
df[preds == 1]

# %%
data = pd.read_csv('./data/test_data.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

# %%
