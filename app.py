from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def doLogin():
    return render_template('login.html')
@app.route('/index')
def upload():
    return render_template("uploadRes.html")
# def hello_world():  # put application's code here
#     # doLogin()
#     # return 'Hello World!'


if __name__ == '__main__':
    app.run()
