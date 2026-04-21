from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({
        "message": "Flask running on Vercel (api/index.py)"
    })

@app.route("/api/hello", methods=["GET"])
def hello():
    name = request.args.get("name", "world")
    return jsonify({
        "message": f"Hello, {name}!"
    })

@app.route("/api/echo", methods=["POST"])
def echo():
    data = request.get_json(silent=True) or {}
    return jsonify({
        "you_sent": data
    })