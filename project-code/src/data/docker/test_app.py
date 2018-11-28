from flask import Flask, request
import json
import pandas as pd

app = Flask(__name__) #create the Flask app

 

 

@app.route('/')

def welcome_page():

    print('Credit Scoring Algorithm')
             

              

              

@app.route('/json-get-results', methods=['POST']) #GET requests will be blocked

def json_get_results():

    req_data = request.get_json()
 
    df = pd.DataFrame.from_dict(req_data, orient='columns')
    
    print(df)

    return 'JSON returned'

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=80)



























