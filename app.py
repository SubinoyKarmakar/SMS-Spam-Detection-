from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load models
try:
    svc = pickle.load(open("svc_model.pkl", "rb"))
    cv = pickle.load(open("cv.pkl", "rb"))
except FileNotFoundError:
    print("Error: Model files not found.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form["message"]
        vec = cv.transform([text]).toarray()
        pred = svc.predict(vec)[0]
        
        # Keyword extraction for logic explanation
        spam_triggers = ['urgent', 'free', 'click', 'winner', 'cash', 'prize', 'offer', 'loan', 'verify', 'account', 'suspended', 'claim', 'inherited']
        detected_keywords = [word for word in spam_triggers if word in text.lower()]
        
        return jsonify({
            "result": "Spam" if pred == 1 else "Not Spam",
            "keywords": detected_keywords,
            "error": False
        })
    except Exception as e:
        return jsonify({"result": "Error", "error": True, "message": str(e)})

@app.route("/ping")
def ping():
    return "ok", 200
    
if __name__ == "__main__":
    app.run(debug=True)
