from flask_cors import CORS, cross_origin   # cors para chamadas de outros domínios
from flask import Flask, request       # flask para criar a aplicação
import os                          # os para acesso aos arquivos
from predict import predict_age    # função do openCV
from spark import filterByClassification    # spark para a filtragem
app = Flask(__name__)          # cria a aplicação   
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')    # pasta de uploads
CORS(app)                     
@app.route("/")                                          # rota raiz
def hello():
    return "Hello World!"

@app.route("/upload", methods=['POST'])                # rota de upload
@cross_origin(origin='localhost', headers=['Content- Type',   'Authorization']) 
def upload_image():                                    # função de upload
    file = request.files['image']                    # pega o arquivo
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)  # caminho do arquivo
    file.save(save_path)                        # salva o arquivo
    try:                                    # tenta executar a função
        age = predict_age(save_path)    # chama a função do openCV para predizer a idade
        classification = filterByClassification(age)    # chama a função do spark com a idade
        jsonFormatted = {                             # formata o json
            'age': age,
            'books': classification.collect()
        }

        return jsonFormatted                     # retorna o json
    except Exception as e:
        print(e)                           # caso ocorra algum erro, imprime o erro
        age = "Error"
    return age


if __name__ == "__main__":
    app.run(debug=True)                        # executa o app