import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import normalize


housingprices_train = pd.read_csv(
    "housingprices_train.csv",
    names=["CloseDate", "SoldPr","Type", "Zip","Area","Rooms", "FullBaths","HalfBaths ","BsmtBth","Beds","BsmtBeds","GarageSpaces"]
)


housingprices_train = housingprices_train.iloc[1:]
housingprices_train['Type'] = housingprices_train['Type'].map({'Townhouse': 0, 'SFH': 1, 'Condo': 2, 'Duplex': 3})
housingprices_train['BsmtBth'] = housingprices_train['BsmtBth'].map({'No': 0, 'Yes': 1})


def ConvertDates(date_str):
    parts = date_str.split('/')
    if len(parts) == 3:
        return int(parts[2] + parts[0].zfill(2) + parts[1].zfill(2)) #if I change the int to float the error changes 
    else:
        return None  

housingprices_train['CloseDate'] = housingprices_train['CloseDate'].apply(ConvertDates)



print(housingprices_train)
print(housingprices_train['Type'])
print(housingprices_train['BsmtBth'])
print(housingprices_train['CloseDate'])
#housingprices_train = housingprices_train.drop(columns=['Type','CloseDate','BsmtBth','Zip','GarageSpaces', 'FullBaths','Rooms','Beds'])
#housingprices_train = housingprices_train.drop(columns=['Type'])
print(housingprices_train)
print(housingprices_train.shape)
print(housingprices_train['GarageSpaces'])
print('type')
print(housingprices_train['GarageSpaces'].dtype)
housingprices_train = housingprices_train.astype(float)
x = housingprices_train.max(axis=0)
print("Denormalizer")
print(x)
print("Denormalizer")
housingprices_train = housingprices_train / x
Home_features = housingprices_train.copy()
Home_labels = Home_features.pop('SoldPr')
Home_features = np.array(Home_features)
print("Home_features.shape")
print(Home_features.shape)


model = Sequential([
    Dense(1, activation='linear',input_shape=(Home_features.shape[1],)),
    Dense(11, activation='ReLU'),
    Dense(1, activation='linear')
])


model.compile(loss='mse', optimizer='sgd', metrics=['mae'])


model.fit(Home_features, Home_labels, epochs=200)


every_output = model.predict(Home_features)
print("every_output.flatten()")
print(every_output.flatten())
print("Home_labels")
print(Home_labels)
my_mae = np.mean(np.abs(every_output.flatten() - Home_labels))

print("My Average Error: ", my_mae)


model.save('HousingPrices.h5', include_optimizer=True)
