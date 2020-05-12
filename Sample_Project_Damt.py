# import scikit-learn as sklearn
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score


le = preprocessing.LabelEncoder()
# Converting string labels into numbers.

col_list = ["Sunrise_Sunset","Normalised Temp","Normalised Wind_Chill","Weather_Condition","Normalised Visibility","Bump","Junction","Traffic_Signal","Nor_weather","Accident"]
df = pd.read_csv("Dataset_main_2.csv", usecols=col_list)
#print(df.shape)
Sun_Con=le.fit_transform(df["Sunrise_Sunset"]) # day=0,Night=1
#print(Sun_Con)
Nor_temp=le.fit_transform(df["Normalised Temp"]) # cold=0,hot=1,mild=2
#print(Nor_temp)
Nor_WindChill=le.fit_transform(df["Normalised Wind_Chill"]) # High=0,Low=1
#print(Nor_WindChill)
Nor_Vis=le.fit_transform(df["Normalised Visibility"]) # bad=0,clear=1,Good=2
#print(Nor_Vis)
Nor_Wea=le.fit_transform(df["Nor_weather"]) # clear weather=0,cloudy=1,Rainy=2
#print(Nor_Wea)
Nor_Accident=le.fit_transform(df["Accident"]) # No=0,Yes=1
#print(Nor_Accident)

#Nor_wcon=le.fit_transform(df["Weather_Condition"]) # false=0, true =1
#print(Nor_wcon)
#Nor_Bump=le.fit_transform(df["Bump"]) # false=0, true =1
#print(Nor_Bump)
#Nor_Crossing=le.fit_transform(df["Crossing"]) # false=0, true =1
# print(Nor_Crossing)
#Nor_Junction=le.fit_transform(df["Junction"]) # false=0, true=1
# print(Nor_Junction)
#Nor_trafficSignal=le.fit_transform(df["Traffic_Signal"]) # false=0, true=1
# print(Nor_trafficSignal)

features=list(zip(Sun_Con,Nor_temp,Nor_WindChill,Nor_Wea,Nor_Vis))
#print(features)
X_train, X_test, y_train, y_test = train_test_split(features, Nor_Accident, test_size=0.33)
#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets

model.fit(X_train,y_train)


predicted= model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
print('Accuracy:{}'.format(accuracy_score(y_test,predicted)))


# print ("Predicted Value:", predicted) # No accident=0, accident happens=1