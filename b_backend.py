
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from logging.handlers import RotatingFileHandler
import logging
from pathlib import Path

# 1. Cargar la bbdd con langchain
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate


logger = logging.getLogger(__name__)


def _configure_logging():
    """Configura el sistema de logs para el módulo."""
    if logger.handlers:
        return

    logger.setLevel(logging.INFO)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_dir / "backend.log",
        maxBytes=1_048_576,  # 1 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


_configure_logging()

class MyQueryChecker:
    def check_query(self, query):
        """Valida y controla el acceso a las consultas SQL generadas."""
        logger.debug("Verificando consulta generada: %s", query)

        if "DELETE" in query.upper():
            logger.error("Consulta bloqueada por intento de eliminación: %s", query)
            raise ValueError("DELETE statements are not allowed")

        if "TUR_PNR_BOLETO" in query.upper():
            if "COMPANY='TEST'" not in query.upper().strip():
                logger.error(
                    "Consulta bloqueada por falta de filtro de compañía permitido: %s",
                    query,
                )
                raise ValueError("No puede consultar otras companias")

        logger.info("Consulta SQL verificada correctamente")
        return query

# Create an instance of your query checker

pg_uri = f"postgresql+psycopg2://pss:Nhrm7167@tkm7.cln23hrnqquq.us-west-2.rds.amazonaws.com:5432/tkm5"
db = f""



# 2. Importar las APIs
import a_env_vars
import os
os.environ["OPENAI_API_KEY"] = a_env_vars.OPENAI_API_KEY

# 3. Crear el LLM
# from langchain.chat_models import ChatOpenAI
query_checker = MyQueryChecker()

llm = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo')



# 4. Crear la cadena
#from langchain_experimental.sql import SQLDatabaseChain
#cadena = SQLDatabaseChain(llm = llm, database = db, verbose=True)
#agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# 5. Formato personalizado de respuesta
formatoSys = """
Dada una base de boletos de avion:
0. Es imperativo siempre incluir un filtro company='{company}' con la tabla tur_pnr_boleto, si se pide info de otras company devolver error.
1. El campo tarifa y tarifa_factura contiene el importe del boleto con la comision incluida
2. Si se piden los tickets emitidos usa tarifa_factura, neto_factura e impuesto_factura y sin filtro void ya que solo intersa quee fueron emitidos
3. Si se piden los tickets volados usar los que tengan fecha de viaje anterior al dia actual (departure_date<DATE()), quitar los anulados (void='N') y usar tarifa, neto o impuestos si se requiere alguno de estos importes
4. El campo neto contiene el importe del boleto sin la comision
5. El campo impuestos contiene los impuestos del boleto
6. El campo codigoaerolinea es la aerolinea que opera el vuelo
7. Para buscar por país ir a tur_airport desde el aeropuerto origen o destino y usar el pais de esa tabla
8. Cuando se pide una ubicacion y no se especifica si es origen o destino, usar destino.
9. En la tabla tur_airport se encuentra las localizacion y pais al que pertenece cada aeropuerto
10. No usar ciudad_destino ni ciudad_origen usar aeropuerto_destino y aeropuerto origen
11. No usar el campo pais.
12. Los boletos anulados tienen el campo void = N en la tabla tur_pnr_boleto 
13. crea una consulta de postgresql
14. revisa los resultados
15. Asegurarse haber cumplido las regla
16. devuelve el dato y mostrar la consulta realizada y dejar claro os parametros que se utilizaron
17. si tienes que hacer alguna aclaración o devolver cualquier texto que sea siempre en español 

"""

formato = """
Eres un experto en análisis de datos y dada una pregunta del usuario sobre una base de boletos de avion:
0. Es imperativo siempre incluir un filtro company='{company}' con la tabla tur_pnr_boleto, si se pide info de otras company devolver error.
#{question}
"""
# Diccionario para almacenar el historial de conversaciones por usuario
conversations = {}
def eliminar_contenido(cadena):
  inicio = cadena.find("```")
  if inicio != -1:
    final = cadena.find("```", inicio + 3)
    if final != -1:
      return cadena[:inicio] + cadena[final+3:]
  return cadena
# 6. Función para hacer la consulta

def consultaSql(user_id,question,company,clearHistory):
    db= SQLDatabase.from_uri(pg_uri,include_tables=['tur_pnr_boleto', 'tur_carrier', 'tur_airport'])
    db_chain = ''#'#SQLDatabaseChain.from_llm(llm, db,use_query_checker=True, verbose=True)
 # qObtener o inicializar el historial de conversación para este usuario
    logger.info(
        "Iniciando consulta SQL para user_id=%s, company=%s, reiniciar_historial=%s",
        user_id,
        company,
        clearHistory,
    )

    try:
        if clearHistory == '1':
            logger.info("Reinicio de historial de conversación solicitado")
            conversations[user_id] = []
        else:
            logger.debug("Se conserva el historial de conversación existente")

        conversation_history = conversations.get(user_id, [])

        # Generar una respuesta utilizando ChatGPT
        input_text = ' '.join(conversation_history + [question])
        logger.debug("Contexto de la conversación: %s", input_text)

        consulta = formato.format(question=input_text, company=company)
        system = formatoSys.format(company=company)

        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", consulta)]
        ).partial(dialect=db.dialect)

        logger.info("Ejecutando cadena de base de datos")
        # response = agent_executor.run(prompt)
        response = db_chain.run(prompt)

        # Agregar la pregunta y la respuesta al historial de conversación
        conversation_history.append(consulta)
        conversation_history.append(response)
        conversations[user_id] = conversation_history

        logger.info("Consulta completada correctamente para user_id=%s", user_id)
        return jsonify({"question": question, "response": response})
    except Exception as exc:
        logger.exception("Error al procesar la consulta SQL para user_id=%s", user_id)
        error_message = {
            "error": "No se pudo procesar la solicitud.",
            "details": str(exc),
        }
        return jsonify(error_message), 500
