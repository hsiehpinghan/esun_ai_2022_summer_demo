from api import bp
from flask import Flask
from flask_ngrok import run_with_ngrok

app = Flask(__name__)

app.register_blueprint(bp)
run_with_ngrok(app)
app.run()
