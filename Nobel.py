import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


st.write(''' # Nobel Prize Category Prediction ''')
st.image("premio-nobel.webp", caption="Every Nobel Prize diploma is a unique, custom-made work of art designed by Swedish and Norwegian artists")

st.header('Texto')

def user_input_features():
  # Entrada
  texto = st.text_input("Enter the text to evaluate")

  user_input_data = {'Text': texto}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

nobel =  pd.read_csv('nobel_final.csv', encoding='latin-1')
X = nobel.clean_motivation
y = nobel.Label

vect = CountVectorizer()
X_dtm = vect.fit_transform(X)

logreg = LogisticRegression()
logreg.fit(X_dtm, y)

df_dtm = vect.transform(df['Text'])
prediction = logreg.predict(df_dtm)

#{'physics':0, 'medicine':1, 'peace':2, 'literature':3, 'chemistry':4, 'economics':5}
#'Physics', 'Medicine', 'Peace', 'Literature', 'Chemistry', 'Economics'
st.subheader('Predicción')
if prediction == 0:
  st.write('Physics')
elif prediction == 1:
  st.write('Medicine')
elif prediction == 2:
  st.write('Peace')
elif prediction == 3:
  st.write('Literature')
elif prediction == 4:
  st.write('Chemistry')
elif prediction == 5:
  st.write('Economics')
else:
  st.write('Sin predicción')
