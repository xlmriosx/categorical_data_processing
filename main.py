import pandas as pd
import sklearn.preprocessing as preprocessing

df = pd.read_csv('cars.csv')
print(df)

print(pd.get_dummies(df['engine_type']))

encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')

print(encoder.fit(df[['engine_type']].values))

print(encoder.transform([['gasoline'],['diesel'],['aceite']]).toarray())

print(encoder.fit(df[['year_produced']].values))

print(encoder.transform([[2016],[2009],[190]]).toarray())