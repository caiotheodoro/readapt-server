from flask_cors import CORS, cross_origin
from flask import Flask, request
import os
from predict import predict_age
from spark import filterByClassification
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
CORS(app)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/upload", methods=['POST'])
@cross_origin(origin='localhost', headers=['Content- Type',   'Authorization'])
def upload_image():
    file = request.files['image']
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    try:
        age = predict_age(save_path)
        classification = filterByClassification(age)
        jsonFormatted = {
            'age': age,
            'books': classification.collect()
        }

        return jsonFormatted
    except Exception as e:
        print(e)
        age = "Error"
    return age


if __name__ == "__main__":
    app.run(debug=True)