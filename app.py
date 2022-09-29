from flask import Flask, render_template, request,redirect,url_for
import os
UPLOAD_FOLDER=os.getcwd()+'\\upload'
app = Flask(__name__)
from user import routes
@app.route('/createAccount')
def createAccount():
    return render_template('createAcc.html')
@app.route('/')
def doLogin():
    return render_template('login.html')
@app.route('/index')
def upload():
    return render_template("index.html")

if __name__ == '__main__':
    app.run()
