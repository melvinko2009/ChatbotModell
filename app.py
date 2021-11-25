from flask import Flask, render_template, request, jsonify
import IRmodel
import Utils

model = IRmodel.Model()
app = Flask(__name__)


def get_response(query):
    #queryNoSpellingMistakes = Utils.checkSpelling(query.lower())
    output = model.get_prediction(query)
    return output


@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {'answer': response}
    return jsonify(message)


if __name__ == '__main__':
    app.run(debug=True)
