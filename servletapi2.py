

from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from b_backendjson import consulta,singleConsulta
from b_backend import consultaSql
import requests
from config import Config
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import traceback

import re
app=Flask(__name__)



@app.route("/request_reportgpt", methods=["POST"])
def request_reportgpt():
    # Obtener el ID del usuario y la pregunta del cuerpo de la solicitud
    user_id = request.json.get("user_id")
    question = request.json.get("question")
    system_question = request.json.get("system_query")
    company = request.json.get("company")
    clearHistory = request.json.get("clear_history")

    response = consulta(user_id,system_question,question,company,clearHistory)
    
    return jsonify({"question": question, "response": response})



@app.route("/request_gpt", methods=["POST"])
def request_gpt():
    # Obtener el ID del usuario y la pregunta del cuerpo de la solicitud
    user_id = request.json.get("user_id")
    question = request.json.get("question")
    company = request.json.get("company")
    clearHistory = request.json.get("clear_history")

    response = consultaSql(user_id,question,company,clearHistory)
    return response
  


# Almacén de conversaciones en memoria
conversations = {}


@app.route("/add_object", methods=["POST"])
def add_object():
    """Agregar un objeto con su definición completa."""
    data = request.json
    object_name = data.get("object")
    brief = data.get("brief")
    fullbrief = data.get("fullbrief")
    api_list = data.get("list", [])

    # Validación de parámetros obligatorios
    if not object_name or not api_list:
        return jsonify({"error": "El nombre del objeto y al menos una API son obligatorios."}), 400

    # Estructura del objeto a almacenar
    objeto = {
        "brief": brief,
        "fullbrief": fullbrief,
        "apis": api_list
    }
    conversations[object_name] = objeto  # Guardar en memoria

    # Respuesta JSON adecuada
    return jsonify({
        "message": f"Objeto '{object_name}' agregado con éxito.",
        "object": objeto
    }), 201  # Código de estado 201: Created

@app.route("/response_question", methods=["POST"])
def response_question():
    """Responder una pregunta analizando con ChatGPT y consultando las APIs necesarias."""
    data = request.json
    user_id = data.get("user_id", "default")
    question = data.get("question")
    extraInfo = data.get("extrainfo")
    context = data.get("context", {})
    parametros = data.get("parametros", {})
    dictionary = data.get("dictionary", {})

    if not question or not context:
        return jsonify({"error": "La pregunta y el contexto son obligatorios."}), 400

    object_name = context.get("object")
    user_name = context.get("user")
    vision = context.get("vision")
    if not object_name:
        return jsonify({"error": "El objeto en el contexto es obligatorio."}), 400

    try:
        collected_data = {}
        actions_to_collect = []
        objects_to_explore = [{"name": object_name, "filters": []}]
        processed_objects = set()
        cycle_count = 0
        max_cycles = 5
        info = {}

        # Analizar inicialmente el objeto raíz
        analysis_result = analyze_question(
            question, collected_data, object_name, user_name, vision, parametros, dictionary, user_id, context, [],extraInfo
        )
        new_actions, new_objects, info = analysis_result
        actions_to_collect.extend(new_actions or [])
        objects_to_explore.extend(new_object for new_object in new_objects if new_object["name"] not in processed_objects)

        # Bucle principal de análisis y recolección
        while (objects_to_explore or actions_to_collect) and cycle_count < max_cycles:
            cycle_count += 1
            # Recolectar información de acciones pendientes con los filtros específicos
            if actions_to_collect:
                new_data = collect_information(user_name, actions_to_collect, vision, parametros, dictionary, [])
                if not isinstance(new_data, dict):
                    print("Advertencia: 'collect_information' devolvió un valor inesperado.")
                    break
                collected_data.update(new_data)
                actions_to_collect.clear()

            # Recolectar información de cada objeto con sus filtros específicos
            for obj in objects_to_explore[:]:  # Copia la lista para evitar modificación concurrente
                obj_name = obj["name"]
                obj_filters = obj.get("filters", [])

                # Evitar reprocesamiento de objetos con los mismos filtros
                if obj_name in processed_objects:
                    continue

                # Obtener y recolectar datos del objeto actual con sus filtros
                obj_data = collect_information(user_name, [obj_name], vision, parametros, dictionary, obj_filters)
                if not isinstance(obj_data, dict):
                    print(f"Advertencia: 'collect_information' devolvió un valor inesperado para el objeto '{obj_name}'.")
                    break
                collected_data.update(obj_data)
                processed_objects.add(obj_name)

            objects_to_explore.clear()  # Limpiar después de recolección para evitar reprocesamiento

            # Reanalizar la pregunta con la información actualizada
            analysis_result = analyze_question(
                question, collected_data, object_name, user_name, vision, parametros, dictionary, user_id, context, [], extraInfo
            )

            if not analysis_result or not isinstance(analysis_result, tuple):
                print(f"Advertencia: 'analyze_question' devolvió un valor inesperado tras recolectar información.")
                break

            new_actions, new_objects, info = analysis_result
            actions_to_collect.extend(new_actions or [])
            objects_to_explore.extend(
                new_object for new_object in new_objects if new_object["name"] not in processed_objects
            )

        if cycle_count >= max_cycles:
            print("Se alcanzó el máximo de ciclos permitidos. Generando respuesta parcial...")

        if not isinstance(info, dict):
            print(f"Advertencia: 'info' no es un diccionario antes de la generación de la respuesta final.")
            info = {}

        final_response = generate_final_response(user_id, question, collected_data, info, parametros)
        if cycle_count >= max_cycles:
            final_response["note"] = "Respuesta generada con la información parcial disponible debido a un límite de ciclos alcanzado."

        return final_response

    except Exception as e:
        # Capturar el stack trace completo y registrarlo
        error_trace = traceback.format_exc()
        print(f"Error interno: {str(e)}\nTraceback:\n{error_trace}")
        return jsonify({"error": f"Error interno: {str(e)}", "trace": error_trace}), 500




def collect_information(user_name, actions_to_collect, vision, parametros, dictionary, filters):
    """ETAPA 2: Ejecutar las acciones seleccionadas y recolectar la información, excluyendo los filtros del resultado."""
    collected_data = {}

    if isinstance(actions_to_collect, list) and actions_to_collect:
        for action_name in actions_to_collect:
            try:
                # Crear una representación de los filtros como parte de la clave
                filter_str = "_".join(f"{flt['field']}={flt['value']}" for flt in filters)
                unique_action_key = f"{action_name}_{filter_str}" if filter_str else action_name
                
                # Obtener la información de acuerdo con los filtros
                response_data = get_info(user_name, action_name, vision, parametros, dictionary, filters)
                
                # Excluir el campo 'filters' de la respuesta si existe
                if isinstance(response_data, dict):
                    collected_data[unique_action_key] = {
                        key: value for key, value in response_data.items() if key != "filters"
                    }
            except Exception as e:
                print(f"Error al ejecutar la acción '{action_name}': {e}")
                collected_data[unique_action_key] = {"error": str(e)}
    else:
        print("Error: 'actions_to_collect' no es una lista válida de acciones.")

    return collected_data


def generate_final_response(user_id, question, collected_data, info, parametros):
    """ETAPA 3: Generar la respuesta final a partir de la información recolectada, incluyendo datos de manuales adicionales."""
    
    # Obtener contexto adicional del manual y ayuda
    manual_context_manual = find_most_relevant_chunks(question, manual_chunks_siti, chunk_embeddings_siti)
    manual_context_ayuda = find_most_relevant_chunks(question, manual_chunks_ayuda, chunk_embeddings_ayuda)
       
    # Filtrar 'filters' de los datos recolectados y de info anidado
    def remove_unwanted_keys(data, unwanted_keys=("filters", "actions")):
        """Remueve las claves no deseadas de un diccionario o lista de manera recursiva."""
        if isinstance(data, dict):
            return {key: remove_unwanted_keys(value, unwanted_keys) for key, value in data.items() if key not in unwanted_keys}
        elif isinstance(data, list):
            return [remove_unwanted_keys(item, unwanted_keys) for item in data]
        return data

    # Aplicación de la función a collected_data y info
    collected_data_without_filters_and_actions = remove_unwanted_keys(collected_data)
    info_without_filters_and_actions = remove_unwanted_keys(info)


    # Construir el input para la consulta con ambos contextos adicionales
    input_usuario = (
        f"Pregunta: {question} | "
        f"Información recolectada: {collected_data_without_filters_and_actions} | "
        f"Información adicional del manual: {manual_context_manual} | "
        f"Información helpdesk usar solo si aplica a la pregunta: {manual_context_ayuda} | "
        f"Información del contexto: {info_without_filters_and_actions}"
    )

    # Registro de la pregunta y el input para depuración
    print("Pregunta:", question)
    print("Input enviado a ChatGPT:", input_usuario)

    # Generar la respuesta final llamando a singleConsulta
    final_response = singleConsulta(
        user_id=user_id,
        systemQuestion="Generar una respuesta basada en la información recolectada y el contexto. (los codigos de objeto obj#xxxxxx son internos y no tienen que aparecer en las respuestas)",
        input_usuario=input_usuario,
        clearHistory="0"
    )

    # Registro de la respuesta para depuración
    print("Respuesta de ChatGPT:", final_response.get("response", "Sin respuesta"))

    return final_response




def extract_json_from_response(response_text):
    """Extrae las listas 'actions_to_collect' y 'objects_to_explore' desde la respuesta de ChatGPT."""
    # Busca el JSON dentro de la respuesta usando expresiones regulares
    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    
    if json_match:
        json_text = json_match.group(1)  # Extrae el bloque de texto JSON

        try:
            # Intenta cargar el texto JSON en un diccionario de Python
            response_data = json.loads(json_text)
            
            # Verificar y retornar las claves esperadas
            actions_to_collect = response_data.get("actions_to_collect", [])
            objects_to_explore = response_data.get("objects_to_explore", [])
            return {
                "actions_to_collect": actions_to_collect,
                "objects_to_explore": objects_to_explore
            }
        except json.JSONDecodeError as e:
            print(f"Error al decodificar JSON: {e}")
            return {"actions_to_collect": [], "objects_to_explore": []}
    else:
        print("No se encontró JSON en la respuesta.")
        return {"actions_to_collect": [], "objects_to_explore": []}


def analyze_question(question, collected_data, object_name, user_name, vision, parametros, dictionary, user_id, context, filters, extraInfo):
    """ETAPA 1: Obtener estructura y determinar qué acciones ejecutar."""
    # Obtener la jerarquía de acciones, el mapa de acciones y datos recopilados
    actions_hierarchy, actions_map, collected_data_initial = get_actions_hierarchy(
        user_name, object_name, vision, parametros, dictionary, filters
    )
    
    # Agregar los datos iniciales recolectados a collected_data
    collected_data.update(collected_data_initial)

    # Iterar por los elementos de context y modificar si el valor comienza con "act___op1"
    # for key, value in context.items():
    #     if isinstance(value, str) and value.startswith("act___op1"):
    #         # Agregar al mapa de acciones en el formato [object_name, key, original_value]
    #         actions_map.append(['modulo', key, value])
    #         # Mantener la clave en context con el valor vacío
    #         context[key] = ""

    # Preparar la entrada para la IA, incluyendo solo las descripciones
    input_usuario = prepare_input_for_chatgpt(question, collected_data, actions_hierarchy, context, extraInfo)
    
    # Solicitar la decisión de la IA sobre qué acciones ejecutar y qué objetos explorar
    action_decision = request_action_decision(user_id, input_usuario, extraInfo)

    # Validar que la respuesta sea un diccionario
    if not isinstance(action_decision, dict):
        action_decision = {}

    # Obtener las descripciones de acciones y objetos a explorar
    actions_to_collect_descriptions = action_decision.get("actions_to_collect", [])
    objects_to_explore = action_decision.get("objects_to_explore", [])

    # Convertir descripciones a códigos usando actions_map
    actions_to_collect = resolve_action_codes(actions_to_collect_descriptions, actions_map)

    # Asegurarse de que la lista de objetos a explorar sea válida
    if not isinstance(objects_to_explore, list):
        objects_to_explore = []

    return actions_to_collect, objects_to_explore, actions_hierarchy




def request_action_decision(user_id, input_usuario, extraInfo):
    """Solicitar a ChatGPT que decida qué acciones ejecutar y qué objetos explorar en formato JSON."""
    action_decision_response = singleConsulta(
        user_id=user_id,
        systemQuestion=f"Seleccionar las acciones necesarias para responder a la pregunta y devolver una lista de 'actions_to_collect' y 'objects_to_explore'. {extraInfo}",
        input_usuario=input_usuario,
        clearHistory="0"
    )

    # Extraer JSON con estructura { "actions_to_collect": [...], "objects_to_explore": [...] }
    return extract_json_from_response(action_decision_response.get("response", ""))

def get_actions_hierarchy(user_name, object_name, vision, parametros, dictionary, filters):
    """Obtener la jerarquía de acciones de un objeto y sus relaciones con filtros específicos."""
    info = get_info(user_name, object_name, vision, parametros, dictionary, filters)
    actions_hierarchy = {
        "object": object_name,
        "actions": {},  
        "related_objects": {}
    }
    actions_map = []  # Almacenar [object, description, action_code]
    collected_data = {}  # Almacenar datos recopilados excluyendo filtros
    title = info.get("objectName","")
    unique_action_key = f"{title} filtrado por {filters}"
    collected_data[unique_action_key] = info.get("fields",{})
    
    # Mapeo de acciones sin códigos y almacenamiento en actions_map
    for action_code, action_description in info.get("actions", {}).items():
        # Dividir `action_description` en `action_descr` y `action_help` usando '|' como delimitador
        action_descr, _, action_help = action_description.partition('|')
        
        # Agregar solo `action_descr` en `actions_map`
        actions_map.append([object_name, action_descr.strip(), action_code])
        
        # En `actions_hierarchy`, usar solo `action_descr` para la clave, y mantener la descripción completa
        actions_hierarchy["actions"]['Acción ' + action_descr.strip()] = action_description


    # Procesar objetos relacionados y recolectar datos
    objects_childs = info.get("records", {})
    for related_object, relation_filters in objects_childs.items():
        related_structure = get_info(user_name, related_object, "", parametros, dictionary, relation_filters)
        
        if related_structure:
            # Mapeo de acciones del objeto relacionado en actions_map
            for rel_action_code, rel_action_description in related_structure.get("actions", {}).items():
                # Dividir `rel_action_description` en `rel_action_descr` y `rel_action_help` usando '|' como delimitador
                rel_action_descr, _, rel_action_help = rel_action_description.partition('|')
                
                # Agregar solo `rel_action_descr` en `actions_map`
                actions_map.append([related_object, rel_action_descr.strip(), rel_action_code])
                
                # En `actions_hierarchy`, usar solo `rel_action_descr` para la clave y mantener la descripción completa
                actions_hierarchy["related_objects"].setdefault(related_object, {"info": {"actions": {}}})
                actions_hierarchy["related_objects"][related_object]["info"]["actions"]['Acción ' + rel_action_descr.strip()] = rel_action_description

            # Recolectar datos del objeto relacionado excluyendo filtros
            if isinstance(related_object, dict):
                title = related_object.get("objectName", "")
            else:
                title = related_object
            unique_action_key = f"{title} filtrado por {relation_filters}"
            collected_data[unique_action_key] = info.get("fields",{}) 


    return actions_hierarchy, actions_map, collected_data

def resolve_action_codes(actions_descriptions, actions_map):
    """Convierte descripciones de acciones de vuelta a sus códigos usando actions_map."""
    actions_to_collect = []
    
    for action in actions_descriptions:
        action_obj = action.get("name")
        action_desc = action.get("action")
        
        if isinstance(action_desc, str):
            # Buscar en actions_map para convertir la descripción a su código
            found = False
            for obj, desc, code in actions_map:
                if obj in action_obj and (action_desc in  desc or  action_desc in code ):
                    actions_to_collect.append(code)  # Añadir el código correspondiente
                    found = True
                    break
            if not found:
                # Si no se encontró en el mapa, conservar la descripción como está
                actions_to_collect.append(action_desc)

    return actions_to_collect









def prepare_input_for_chatgpt(question, collected_data, actions_hierarchy, context, extraInfo):
    """Construir el mensaje que se enviará a ChatGPT con la jerarquía de acciones."""
    return (
        f"Pregunta: {question} | Información recolectada: {collected_data} | "
        f"Estructura de acciones y relaciones: {actions_hierarchy}. "
        "Con base en la pregunta y la información proporcionada:\n\n"
        "- Si puedes responder directamente con la información suministrada, devuelve un JSON vacío.\n"
        f"- Si necesitas realizar acciones adicionales para obtener información, completa actions_to_collect como: {extraInfo} \n"
        "  ```json\n"
        "  {\n"
        "    \"actions_to_collect\": [ { \"name\": \"NombreDelObjeto(formato obj#xxxx-xxxxx-xxxx-xxxxxxx)\", \"action\": \"action\"} ,  { \"name\": \"Módulo xxxx\", \"action\": \"act___op1=xxxxxxxxxxxxxxxxx\"},  { \"name\": \"TRX xxxxxxxxxx\", \"action\": \"GET_TRX:xxxxxx\"}]\n"
        "  }\n"
        "  ```\n"
        "- Si necesitas explorar más información sobre ciertos objetos relacionados, devuélvelos en `objects_to_explore` en formato JSON con la siguiente estructura para incluir filtros:\n"
        "  ```json\n"
        "  {\n"
        "    \"objects_to_explore\": [\n"
        "      {\n"
        "        \"name\": \"NombreDelObjeto(formato obj#xxxx-xxxxx-xxxx-xxxxxxx)\",\n"
        "        \"filters\": [\n"
        "          {\"field\": \"filter\", \"value\": \"valor\", \"operation\": \"=\"},\n"
        "          {\"field\": \"filter\", \"value\": \"otroValor\", \"operation\": \">\"}\n"
        "        ]\n"
        "      },\n"
        "      {\n"
        "        \"name\": \"OtroObjeto\",\n"
        "        \"filters\": [\n"
        "          {\"field\": \"filter\", \"value\": \"valor\", \"operation\": \"<=\"}\n"
        "        ]\n"
        "      }\n"
        "    ]\n"
        "  }\n"
        "  ```\n"
        "- Asegúrate de que los filtros en `objects_to_explore` no se contradigan o se solapen entre sí (por ejemplo, rangos de fechas superpuestos o condiciones opuestas).\n"
        "- Combina y optimiza filtros para que todos sean consistentes entre sí antes de devolverlos.\n"
        " Devuelve ambas listas (`actions_to_collect` y `objects_to_explore`) en un único bloque JSON. Asegúrate de que `objects_to_explore` contenga el nombre del objeto y"
        " cualquier filtro necesario en el formato indicado. Fechas en formato DD/MM/YYYY.  "
        f" | Otras opciones de contexto general: {context} "
    )








def consulta_api(api_url):
    """Consulta la API y devuelve la respuesta."""
    headers = {
        "Authorization": f"Bearer {Config.API_TOKEN}"
    }

    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            print("Error: Token no válido o expirado.")
            return None
        else:
            print(f"Error en la API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error al llamar a la API: {str(e)}")
        return None

@app.route("/list_objects", methods=["GET"])
def list_objects():
    """Listar los objetos definidos actualmente."""
    return jsonify(conversations)


@app.route("/api/clientes/<int:id>", methods=["GET"])
def get_cliente(id):
    """Simular la API de clientes."""
    # Datos simulados para diferentes IDs de cliente
    clientes = {
        1: {"nombre": "Juan Pérez", "estado": "activo", "deuda": 1500.0},
        2: {"nombre": "María Gómez", "estado": "inactivo", "deuda": 0.0}
    }
    cliente = clientes.get(id)

    if cliente:
        return jsonify(cliente), 200
    else:
        return jsonify({"error": "Cliente no encontrado"}), 404


import requests

import json

# Diccionario global para almacenar caché de consultas
query_cache = {}

def get_info(user_name, object_name, vision, parametros, dictionary, filters=None):
    """Obtener información de un objeto con filtros específicos, con almacenamiento en caché."""
    url = f"{Config.SERVER_URL}/rest/bot/getInfo"

    # Verificación del formato de `filters`
    if not isinstance(filters, list) or not all(
        isinstance(f, dict) and "field" in f and "value" in f and "operation" in f
        for f in filters
    ):
        filters = [{'field': '', 'operation': 'any', 'value': filters}]


    # Generar una clave única para la consulta
    cache_key = f"{user_name}-{object_name}-{vision}-{dictionary}-{json.dumps(filters, sort_keys=True)}"

    # Comprobar si la consulta ya está en caché
    if cache_key in query_cache:
        print("Resultado encontrado en caché. Evitando consulta repetida.")
        return query_cache[cache_key]

    data = {
        "user": {
            "usuario": user_name,
            "password": "botAccess",
        },
        "action": object_name,
        "showDetails": True,
        "vision": vision,
        "dictionary": dictionary,
        "filters": filters
    }

    # Log de depuración para ver el contenido de la solicitud
    print("Solicitud a getInfo - URL:", url)
    print("Datos enviados en la solicitud (JSON):")
    print(json.dumps(data, indent=4))

    response = requests.post(
        url,
        json=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {Config.API_TOKEN}"
        }
    )

    if response.status_code == 200:
        
        result = response.json()
        query_cache[cache_key] = result  # Almacenar en caché el resultado
        print("Result ---> ", result)
   
        return result
    elif response.status_code == 401:
        print("Error: Token no válido o expirado.")
    else:
        print(f"Error al obtener la información: {response.status_code} - {response.text}")
    
    return None

import os
import pickle
import re
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Inicializar HuggingFaceEmbeddings
print("Inicializando HuggingFaceEmbeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

import os
import re
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Función para cargar contenido del manual
def load_manual_content(filepath):
    with open(filepath, 'r', encoding='ISO-8859-1') as file:
        return file.read()

# Función para dividir 'ayuda.txt' en fragmentos de preguntas y respuestas
def split_manual_by_question_answer(manual_content):
    print("Dividiendo 'ayuda.txt' en preguntas y respuestas completas...")
    # Usar expresiones regulares para encontrar pares de pregunta y respuesta
    pattern = r'("Pregunta \d+:.*?")\s*("Respuesta: \d+:.*?")'
    matches = re.findall(pattern, manual_content, re.DOTALL)
    chunks = [f"{question} {answer}" for question, answer in matches]
    print(f"Total de fragmentos de 'ayuda.txt': {len(chunks)}")
    return chunks

# Función para dividir el contenido de 'siti.txt' en fragmentos
def split_manual_content(manual_content, chunk_size=200):
    sentences = re.split(r'(?<=[.!?])\s+', manual_content)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        if current_length + len(sentence) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    print(f"Total de fragmentos de 'siti.txt': {len(chunks)}")
    return chunks

# Guardar fragmentos y embeddings en un archivo
def save_embeddings(file_path, chunks, embeddings):
    with open(file_path, 'wb') as f:
        pickle.dump({'chunks': chunks, 'embeddings': embeddings}, f)
    print(f"Embeddings guardados en {file_path}")

# Cargar fragmentos y embeddings desde un archivo
def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Embeddings cargados desde {file_path}")
    return data['chunks'], data['embeddings']

# Función para cargar o generar embeddings con barra de progreso, ajustada para los diferentes formatos de contenido
def get_or_generate_embeddings(file_path, manual_content, split_function, chunk_size=200):
    if os.path.exists(file_path):
        return load_embeddings(file_path)
    else:
        manual_chunks = split_function(manual_content, chunk_size=chunk_size) if split_function == split_manual_content else split_function(manual_content)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        chunk_embeddings = []
        print(f"Generando embeddings para {file_path} con barra de progreso...")
        for chunk in tqdm(manual_chunks, desc="Progreso de embeddings"):
            embedding = embeddings.embed_query(chunk)
            chunk_embeddings.append(embedding)

        save_embeddings(file_path, manual_chunks, chunk_embeddings)
        return manual_chunks, chunk_embeddings

# Rutas y contenidos de los archivos
manual_path_siti = "siti.txt"
manual_path_ayuda = "ayuda.txt"
full_manual_content_siti = load_manual_content(manual_path_siti)
full_manual_content_ayuda = load_manual_content(manual_path_ayuda)

# Obtener o generar embeddings para ambos archivos
manual_chunks_siti, chunk_embeddings_siti = get_or_generate_embeddings("siti_embeddings.pkl", full_manual_content_siti, split_manual_content)
manual_chunks_ayuda, chunk_embeddings_ayuda = get_or_generate_embeddings("ayuda_embeddings.pkl", full_manual_content_ayuda, split_manual_by_question_answer)

# Función para encontrar el fragmento más relevante
from sklearn.metrics.pairwise import cosine_similarity

def find_most_relevant_chunks(question, manual_chunks, chunk_embeddings, top_n=10):
    """
    Encuentra las N chunks más relevantes para la pregunta basada en las similitudes de coseno.

    Args:
        question (str): La pregunta para generar el embedding.
        manual_chunks (list): Lista de chunks de texto manuales.
        chunk_embeddings (list): Lista de embeddings de los chunks.
        top_n (int): Número de chunks más relevantes a devolver.

    Returns:
        list: Lista de los chunks más relevantes en orden de relevancia.
    """
    # Generar el embedding de la pregunta
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    question_embedding = embeddings.embed_query(question)
    
    # Calcular similitudes de coseno
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    
    # Obtener los índices de las N similitudes más altas
    most_relevant_indices = similarities.argsort()[-top_n:][::-1]  # Orden descendente
    
    # Devolver los chunks correspondientes
    most_relevant_chunks = [manual_chunks[i] for i in most_relevant_indices]
    return most_relevant_chunks







if __name__ == "__main__":
    app.run(debug=True)
    
   