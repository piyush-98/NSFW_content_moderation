import urllib.request
import pandas as pd
df=pd.read_table(r"C:\Users\PIYUSH\Desktop\nsfw_data_scraper\raw_data\drawings\urls_drawings.txt",header=None)
for i in range(df.shape[0]):
    try:
        urllib.request.urlretrieve(df.iloc[i][0], r"G:\drawing\{}.jpg".format(i))
    except:
        continue