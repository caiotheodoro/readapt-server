from concurrent.futures import process
# cors para chamadas de outros domínios
from flask_cors import CORS, cross_origin
from flask import Flask, request       # flask para criar a aplicação
import os                          # os para acesso aos arquivos
from predict import predict_age    # função do openCV
from spark import filterByClassification    # spark para a filtragem
app = Flask(__name__)          # cria a aplicação
UPLOAD_FOLDER = os.path.join(os.path.dirname(
    __file__), 'uploads')    # pasta de uploads
CORS(app)


@app.route("/")                                          # rota raiz
def hello():
    return "Hello World!"


@app.route("/upload", methods=['POST'])                # rota de upload
@cross_origin(origin='localhost', headers=['Content- Type',   'Authorization'])
def upload_image():                                    # função de upload
    file = request.files['image']                    # pega o arquivo
    save_path = os.path.join(
        UPLOAD_FOLDER, file.filename)  # caminho do arquivo
    file.save(save_path)                        # salva o arquivo
    try:                                    # tenta executar a função
        # chama a função do openCV para predizer a idade
        age = predict_age(save_path)
        # chama a função do spark com a idade
        classification = filterByClassification(age)
        jsonFormatted = {                             # formata o json
            'age': age,
            'books': classification.collect()
        }

        return jsonFormatted                     # retorna o json
    except Exception as e:
        # caso ocorra algum erro, imprime o erro
        print(e)
        age = "Error"
    return age


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=process.env.PORT |
            5000)                  # executa o app
