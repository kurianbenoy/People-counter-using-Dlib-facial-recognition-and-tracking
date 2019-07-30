from flask import Flask
from flask_cors import CORS
from flask import request
app= Flask(__name__)

cors=CORS(app,resources={"/"})

@app.route('/')

def hello_count():
    #content=request.get_json()
    #count=content.get('head_count','')
    return "Hello world"
    


#@app.route('/user/<username>')
#def show_user_profile(username):
#    return 'User %s' % escape(username)
