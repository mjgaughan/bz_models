import pandas as pd



df_0 = pd.read_csv("temp_final_labeled_body_shuffled_0.csv")
df_1 = pd.read_csv("temp_final_labeled_body_shuffled_1.csv")
print(df_0.shape)
print(df_1.shape)
frames = [df_0, df_1]

result = pd.concat(frames)
print(result.shape)
result.to_csv("test.csv")
