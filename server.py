from flask import Flask
from multiprocessing import Process
from threading import Thread
from merged_code import *
app = Flask(__name__)


@app.route('/count/')
def hello_world():
    Thread(target=peoplereg).start()
    Thread(target=retinanet).start()


#@app.route('/user/<username>')
#def show_user_profile(username):
#    return 'User %s' % escape(username)
