
"""
@author - Darshjyot
"""

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import warnings
import model
from model import predict
warnings.filterwarnings("ignore")

app = Flask(__name__)  # intitialize the flaks app  # common 

#Read metadata file
product_info = pd.read_csv('dataset/sample30.csv', usecols = ['id', 'name', 'brand', 'categories'])
product_info.drop_duplicates(subset='id', keep='first', inplace=True)
product_info = product_info.set_index('id')




@app.route('/')
@app.route('/index')
def index():
        # home page to render the form
    return  render_template('index.html')




# http:baseurl/recommendations

@app.route('/recommendations', methods=['GET', 'POST'])
def recommend():
    # call the model for prediction
    username = request.form['username']
    prediction = predict(username)
    df_pred = pd.DataFrame(prediction, columns=['id'])

    # Set product information
    df_pred['name'] = df_pred['id'].apply(lambda x : str(product_info.loc[str(x), 'name']))
    df_pred['brand'] = df_pred['id'].apply(lambda x : str(product_info.loc[str(x), 'brand']))
    df_pred['categories'] = df_pred['id'].apply(lambda x : str(product_info.loc[str(x), 'categories']))
    
    return  render_template('view.html',tables=[df_pred.to_html(classes='product')], titles = ['Id', 'Name', 'Brand', 'Category'])





# Any HTML template in Flask App render_template

if __name__ == '__main__' :
    app.run(debug=True )  # this command will enable the run of your flask app or api
    
    #,host="0.0.0.0")






