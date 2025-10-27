import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import OutputParserException
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

import a_env_vars

os.environ["OPENAI_API_KEY"] = a_env_vars.OPENAI_API_KEY


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configura el sistema de logging del módulo JSON."""
    if logger.handlers:
        return

    logger.setLevel(logging.INFO)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_dir / "backendjson.log",
        maxBytes=1_048_576,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


_configure_logging()

# Define a new Pydantic model with field descriptions and tailored for Twitter.
class Command(BaseModel):
    object: str = Field(description="Objeto al que se aplica la instruccion: campo, columna, fila, filtro")
    field: str = Field(description="Campo de la base al que aplica")
    function:  Optional[str] = Field(description="Funcion que se aplica al object de tipo campo puede ser SUM, MAX, MIN,AVG ")
    operator: Optional[str] = Field(description="Operador que se aplica si es un Filtro puede ser: >,<,=,<>, in")
    value: Optional[str] = Field(description="Valor de la condicion del filtro")
    value_to: Optional[str] = Field(description="Valor para el operador in, valor desde")
    value_from: Optional[str] = Field(description="Valor para el operador in, valor hasta")

  
class Report(BaseModel):
    type: str = Field(description="Tipo de reporte: grilla, agrupado, detalle")
    list_command: List[Command] = Field(description="Lista de comandos")

# Instantiate the parser
parser = PydanticOutputParser(pydantic_object=Report)

# Verifica si `get_format_instructions` está disponible; de lo contrario, usa `schema()`
try:
    format_instructions = parser.get_format_instructions()
except AttributeError:
    format_instructions = parser.pydantic_object.schema()

# Construir el prompt con las instrucciones de formato
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(
            """{system_question}\n{format_instructions}\n{question}"""
        )
    ],
    input_variables=["question"],
    partial_variables={
        "format_instructions": format_instructions,
    },
)


chat_model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=1000
)
conversations = {}


def consulta(user_id, systemQuestion, input_usuario, company, clearHistory):
    logger.info(
        "Procesando consulta JSON para user_id=%s, company=%s, reiniciar_historial=%s",
        user_id,
        company,
        clearHistory,
    )

    try:
        if clearHistory == '1':
            logger.info("Reinicio de historial solicitado para user_id=%s", user_id)
            conversations[user_id] = []
        else:
            logger.debug("Se mantiene el historial existente para user_id=%s", user_id)

        conversation_history = conversations.get(user_id, [])
        logger.debug(
            "Historial actual para user_id=%s: %s",
            user_id,
            conversation_history,
        )

        if not conversation_history:
            input_text = ' '.join([input_usuario])
        else:
            input_text = 'Historial conversación: '.join(
                conversation_history + ['| Nueva pregunta:' + input_usuario]
            )

        logger.debug("Entrada construida para user_id=%s: %s", user_id, input_text)

        _input = prompt.format_prompt(
            question=input_text,
            system_question=systemQuestion,
        )
        logger.debug("Prompt generado para user_id=%s", user_id)

        output = chat_model(_input.to_messages())
        logger.info("Respuesta generada correctamente para user_id=%s", user_id)

        conversation_history.append(input_usuario)
        conversations[user_id] = conversation_history

        return output.content
    except Exception:
        logger.exception(
            "Error al procesar la consulta JSON para user_id=%s",
            user_id,
        )
        raise

        
    try:
        parsed = parser.parse(output.content)
    except OutputParserException:
        new_parser = OutputFixingParser.from_llm(
            parser=parser,
            llm=ChatOpenAI()
        )
        parsed = new_parser.parse(output.content)
    logger.debug("Resultado parseado para user_id=%s: %s", user_id, parsed)
    return parsed


def singleConsulta(user_id, systemQuestion, input_usuario, clearHistory):
    """Generar una respuesta utilizando ChatGPT con manejo de historial."""

    # Manejo del historial según el parámetro clearHistory
    if clearHistory == '1':
        logger.info("Reinicio de historial solicitado para user_id=%s", user_id)
        conversations[user_id] = []
    else:
        logger.debug("Manteniendo historial existente para user_id=%s", user_id)

    # Obtener el historial de conversación actual
  #  conversation_history = conversations.get(user_id, [])

    # Formatear la entrada en función del historial
 #   if not conversation_history:
        input_text = input_usuario
#    else:
#        input_text = ' '.join(conversation_history + ['| Nueva pregunta: ' + input_usuario])

    # Crear un prompt adecuado para enviar al modelo
    prompt_template = ChatPromptTemplate.from_template(
        "Sistema: {system_question}\nPregunta: {question}"
    )
    _input = prompt_template.format_prompt(
        system_question=systemQuestion,
        question=input_text
    )

    # Generar la respuesta del modelo
    try:
        output = chat_model(_input.to_messages())
        logger.info("Respuesta de singleConsulta generada para user_id=%s", user_id)
    except OutputParserException:
        logger.exception(
            "Error al procesar la respuesta del modelo para user_id=%s",
            user_id,
        )
        return {"error": "Error al generar la respuesta del modelo."}
    except Exception:
        logger.exception("Error inesperado en singleConsulta para user_id=%s", user_id)
        return {"error": "Ha ocurrido un error inesperado."}

    # Actualizar el historial de conversación
    #conversation_history.append(input_usuario)
    #conversations[user_id] = conversation_history

    # Devolver la respuesta del modelo
    return {"response": output.content}
        
