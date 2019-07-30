from flask import Flask
from flask import request
app= Flask(__name__)

@app.route('/')

def hello_count():
    content=request.get_json()
    #count=content.get('head_count','')
    return content
    


#@app.route('/user/<username>')
#def show_user_profile(username):
#    return 'User %s' % escape(username)
