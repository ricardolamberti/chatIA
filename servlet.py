from flask import  Flask, jsonify, request
import b_backend
import b_backendjson
app=Flask(__name__)


@app.route("/")
def root():
    return "root"

@app.route('/question', methods=['POST'])
def handle_post():
    if request.is_json:
        jsonquestion = request.get_json()
        question=jsonquestion["text"]
        
    if request.method == 'POST':
        question = request.form['question']
        
    print(question)
    respuesta = b_backend.consulta(question)
    
    #print(respuesta)
    #json = b_backendjson.consulta(respuesta)
    #print(json)

    response = app.response_class(
        response = respuesta,
        mimetype = 'application/json'
    )
    return response,200
 

if __name__=='__main__':
    app.run(debug=True)

