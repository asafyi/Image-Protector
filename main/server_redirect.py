from flask import Flask, request, redirect
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1


@app.before_request
def before_request():
    print("redirecting from http to https")
    url = request.url.replace('http://', 'https://', 1)
    code = 301
    return redirect(url, code=code)


