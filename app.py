from flask import Flask, render_template
from flask_socketio import SocketIO, send
from chat import main


app = Flask(__name__)


app.config["SECRET"] = "secret"
socketio = SocketIO(app, cors_allowed_origins="*")

# Receives message from user, gives bot's response
@socketio.on("message")
def handle_message(message):
    print("Received message: "+ message)
    if message != "User connected!":
        bot_response = main(message)
        send(bot_response, broadcast=True)

@app.route("/")
def index():
    return render_template("index_2.html")


if __name__ == "__main__":
    socketio.run(app, host="localhost")
