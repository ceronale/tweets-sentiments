
import pandas as pd
import numpy as np 
import re 
from sklearn.feature_extraction.text  import CountVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.preprocessing import LabelEncoder


#DATA
df = pd.read_csv('./training-1600000-processed-noemoticon.csv',    
                 encoding='ISO-8859-1',   
                 names=[   
                        'target',  
                        'id',    
                        'date',
                        'flag',
                        'user',
                        'text' ])    


#Agrupacion de tweets
df.target.unique()


#Training
x = df.text.values
y = df.target.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=32)

#(Convert a collection of text documents to a matrix of token counts)
vectorizer = CountVectorizer()
#Vocabulary
vectorizer.fit(x_train)
X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)


# Limpiar String
d = ",.!?/&-:;@'..."
"["+"\\".join(d)+"]"
s = x_train[0]
s = ' '.join(w for w in re.split("["+"\\".join(d)+"]", s) if w)


#Logistic Regression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("Probability:", score*100,"%")

#Test 1
tweet = 'This futuristic street light design is surprisingly eco-friendly'
vectTweet = vectorizer.transform(np.array([tweet]))  #
prediction = model.predict(vectTweet)  
print('The tweet is ', 'positive' if prediction[0]==4 else 'negative')


#Test 2
tweet = 'Fast and furious 9 is hands down the worst movie I’ve ever watched...'
vectTweet = vectorizer.transform(np.array([tweet]))  
prediction = model.predict(vectTweet) 
print('The tweet is ', 'positive' if prediction[0]==4 else 'negative')