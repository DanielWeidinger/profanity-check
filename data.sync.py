import re
import pandas as pd

# %%
clear_data = pd.read_csv('./profanity_check/data/clean_data.csv')
len(clear_data)
offensive = clear_data[clear_data["is_offensive"] == 1]
normal = clear_data[clear_data["is_offensive"] == 0]
print(len(offensive))
print(len(normal))
print(len(normal)/(len(offensive)+len(normal)))


def add_to_df(base, to_add, is_offensive):
    return pd.concat([base, pd.DataFrame(
        [{"text": x, "is_offensive": is_offensive} for x in to_add])])


# %%
df = pd.read_csv("data/surgeai_list.csv", sep=",")
texts_1 = list(df["Canonical Form 1"].astype(str))
texts_2 = list(df[~df["Canonical Form 2"].notna()
                  ]["Canonical Form 2"].astype(str))
texts_3 = list(df[~df["Canonical Form 3"].notna()
                  ]["Canonical Form 3"].astype(str))

texts = texts_1 + texts_2 + texts_3
print(f"Surge adds {len(texts)}")

clear_data = add_to_df(clear_data, texts, 1)
len(clear_data)

# %%

df = pd.read_csv("data/kaggle_hate_speech.csv", sep=",")
pattern = r'@\w+:?'
good = list(df[df["class"] == 2]["tweet"].apply(
    lambda x: re.sub('"', '', x)).apply(lambda x: re.sub(pattern, '', x).strip()))

bad = list(df[df["class"] == 1]["tweet"].apply(
    lambda x: re.sub('"', '', x)).apply(lambda x: re.sub(pattern, '', x).strip()))
print(f"Kaggle adds good {len(good)}")
print(f"Kaggle adds bad {len(bad)}")
clear_data = add_to_df(clear_data, good, 0)
clear_data = add_to_df(clear_data, bad, 1)
len(clear_data)
# %%

words = list(pd.read_csv("data/words.csv", sep=",")["word"])
print(f"MIT adds {len(words)}")
clear_data = add_to_df(clear_data, words, 0)
len(clear_data)

# %%
offensive = clear_data[clear_data["is_offensive"] == 1]
normal = clear_data[clear_data["is_offensive"] == 0]
print(len(offensive))
print(len(normal))
print(len(normal)/(len(offensive)+len(normal)))
# %%
cd_len = len(clear_data)
clear_data = clear_data[clear_data["text"].notna()].sample(
    frac=1).reset_index(drop=True)
print(f"NaN entries: {cd_len-len(clear_data)}")
train_split = 1
test_split = 1-train_split
cd_len = len(clear_data)
clear_data\
    .iloc[:int(cd_len*train_split)].to_csv("data/train_data.csv", index=False)
clear_data\
    .iloc[int(cd_len*test_split):].to_csv("data/test_data.csv", index=False)
clear_data
