from flask import Flask,request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('iris.pkl','rb'))
list = ['Iris-setosa','Iris-versicolor','Iris-virginica']

@app.route('/')
def hello_world():
    return render_template("iris.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)
    return render_template('iris.html',pred='{}'.format(list[prediction[0]]))


if __name__ == '__main__':
    app.run(debug=True)