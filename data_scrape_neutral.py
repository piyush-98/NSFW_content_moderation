import urllib.request
import pandas as pd
df=pd.read_table(r"C:\Users\PIYUSH\Desktop\nsfw_data_scraper\raw_data\neutral\urls_neutral.txt",header=None)
for i in range(df.shape[0]):
    try:
        urllib.request.urlretrieve(df.iloc[i][0], r"G:\neutral\{}.jpg".format(i))
    except:
        continue