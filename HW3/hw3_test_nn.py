import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import normalize
from keras.models import load_model
from sklearn.metrics import  accuracy_score
housingprices_valid = pd.read_csv(
    "housingprices_test.csv",
    names=["CloseDate", "SoldPr","Type", "Zip","Area","Rooms", "FullBaths","HalfBaths ","BsmtBth","Beds","BsmtBeds","GarageSpaces"]
)
priceDenormalizer = 4746395.43

housingprices_valid = housingprices_valid.iloc[1:]
housingprices_valid['Type'] = housingprices_valid['Type'].map({'Townhouse': 0, 'SFH': 1, 'Condo': 2, 'Duplex': 3})
housingprices_valid['BsmtBth'] = housingprices_valid['BsmtBth'].map({'No': 0, 'Yes': 1})


def ConvertDates(date_str):
    parts = date_str.split('/')
    if len(parts) == 3:
        return int(parts[2] + parts[0].zfill(2) + parts[1].zfill(2)) #if I change the int to float the error changes 
    else:
        return None  

housingprices_valid['CloseDate'] = housingprices_valid['CloseDate'].apply(ConvertDates)



print(housingprices_valid)
print(housingprices_valid['Type'])
print(housingprices_valid['BsmtBth'])
print(housingprices_valid['CloseDate'])

print(housingprices_valid)
print(housingprices_valid.shape)


housingprices_valid = housingprices_valid.astype(float)
x = housingprices_valid.max(axis=0)
print(x)
housingprices_valid = housingprices_valid / x
print('normalized')
print(housingprices_valid)
Home_features = housingprices_valid.copy()
Home_labels = Home_features.pop('SoldPr')
Home_features = np.array(Home_features)
print("Home_features.shape")
print(Home_features.shape)


model = load_model('HousingPrices.h5')


every_output = model.predict(Home_features)

my_mae = np.mean(np.abs(every_output.flatten() - Home_labels))
print(every_output * priceDenormalizer)
print("My Average Error: ", my_mae)
accuracy_scores = []
total = 0

"""
for i in range(0,len(every_output)):
    total += abs((every_output[i][0] *  priceDenormalizer) - Home_labels[i+1]* priceDenormalizer)
print("accuracy offset")
print(total / len(Home_labels))
"""


