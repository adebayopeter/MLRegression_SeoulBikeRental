import pandas as pd
import chardet

# Detect the encoding
with open('dataset/SeoulBikeData.csv', 'rb') as file:
    encoding_result = chardet.detect(file.read())
    # print(encoding_result['encoding'])


data = pd.read_csv('dataset/SeoulBikeData.csv', encoding='ISO-8859-1')

print(data.head())


