import pickle
import numpy as np

from flask import Flask, redirect, render_template, url_for, request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField, FloatField
from wtforms.validators import Email, InputRequired, Length, ValidationError
# Loading the saved model
rfRegCO_model = pickle.load(open('RandomForest_CO_pred.pkl','rb'))

# Function to predict CO emmision

def CO_pred(model,AT=0, AFDP=0, GTEP=0, TIT=0, TAT=0, TEY=0, CDP=0):
    
    x_in = np.array([AT, AFDP, GTEP, TIT, TAT, TEY, CDP]).reshape(1,-1)
    
    pred = model.predict(x_in)

    return round(pred[0],3)

# print(CO_pred(rfRegCO_model,23.056,4.2547,30.505,1100.0,542.30, 150.94, 13.379))


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey123'

class PredictionForm(FlaskForm):
    AT = FloatField('AT',validators=[InputRequired()])
    AFDP = FloatField('AFDP',validators=[InputRequired()])
    GTEP = FloatField('GTEP',validators=[InputRequired()])
    TIT = FloatField('TIT',validators=[InputRequired()])
    TAT = FloatField('TAT',validators=[InputRequired()])
    TEY = FloatField('TEY',validators=[InputRequired()])
    CDP = FloatField('CDP',validators=[InputRequired()])
    submit = SubmitField("Predict")


@app.route("/")
def home():
    return render_template('home.html')

@app.route('/prediction', methods=["GET","POST"])
def prediction():
    form = PredictionForm()
    print('outside validate')
    if form.validate_on_submit():
        print('inside validate')
        AT = float(form.AT.data)
        AFDP = float(form.AFDP.data)
        GTEP = float(form.GTEP.data)
        TIT = float(form.TIT.data)
        TAT = float(form.TAT.data)
        TEY = float(form.TEY.data)
        CDP = float(form.CDP.data)
        pred = CO_pred(rfRegCO_model,AT,AFDP,GTEP,TIT,TAT,TEY,CDP)

        return render_template('predicted.html',pred=pred)
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)