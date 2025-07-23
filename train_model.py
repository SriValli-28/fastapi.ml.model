import pandas as pandas
import sklearn.model_selection import train_test_split
from sklearn.linear_model import logisticregression
import joblib 
#creating data
data={
    "Math":[78,45,90,35,65,88,76,54,89,67],
    "Science":[68,55,80,45,70,85,60,50,90,75],
    "English":[76,65,88,40,70,80,72,55,85,60],
    "Result":["Pass","Fail","Pass","Fail","Pass","Pass","Pass","Fail","Pass","Fail"]
}
df=pd.DataFrame(data)
df['Result'] = df['Result'].map({'Pass': 1, 'Fail': 0})
print(df)
#features and labels
x= df[['Math', 'Science', 'English']]
y=df['Result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#model training 
model = LogisticRegression()
model.fit(x_train, y_train)
#save dump(pkl)
joblib.dump(model, "model.pkl")                                                                                           