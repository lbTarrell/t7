from flask import Flask, render_template, request
import pickle
import numpy as np

app=Flask(__name__)

loaded_model = pickle.load(open("titanic.pkl","rb"))
loaded_model1 = pickle.load(open("minmaxx2.pkl","rb"))
loaded_model2 = pickle.load(open("onehot.pkl","rb"))

@app.route('/')
def home():
    return render_template("home.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,8)
    to_predict1 = loaded_model1.transform(to_predict)
    result = loaded_model.predict(to_predict1)
    return result[0]

@app.route('/result',methods = ['POST'])
def result():
    prediction=''
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        print(to_predict_list.values())

        to_predict_list['embarked']=loaded_model2.transform([[to_predict_list['embarked']]]).toarray().tolist()
        to_predict_list['Q'],to_predict_list['S']=str(to_predict_list['embarked'][0][0]),str(to_predict_list['embarked'][0][1])
        to_predict_list.pop('embarked', None)
        b={'pclass': '1', 'age': '20', 'sibsp': '1', 'parch': '1', 'fare': '7', 'z': '1', 'Q': '3.0', 'S': '0.0'}
        c={}
        for i,p in b.items():
            for e,f in to_predict_list.items():
                 if e==i:
                      c.update({e:f})
        print(c)
        to_predict_list=list(c.values())
        to_predict_list = list(map(float, to_predict_list))
        print("Before sending to model", to_predict_list)
        result = ValuePredictor(to_predict_list)
        print('Scaling',loaded_model1.transform(np.array(to_predict_list).reshape(1,8)))
        print("result from model", result)
        if int(result)==0:
            prediction='can not survive'
        else:
            prediction='survive'
        print(prediction)
        return render_template("result.html",prediction=prediction)

@app.route('/',methods = ['POST'])
def home1():
    if request.method == 'POST':
        return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
