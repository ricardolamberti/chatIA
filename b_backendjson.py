import os
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models import ChatOpenAI
from typing import Optional
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import OutputFixingParser
from langchain.schema import OutputParserException
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

import a_env_vars
import os
os.environ["OPENAI_API_KEY"] = a_env_vars.OPENAI_API_KEY

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
    if (clearHistory=='1'):
        print("Borrando historico")
        conversations[user_id] = []
    else:
        print("Mantener historico")
     
    conversation_history = conversations.get(user_id, [])

    # Generar una respuesta utilizando ChatGPT
    if (conversation_history==[]):
        input_text = ' '.join([input_usuario]) 
    else:
        input_text = 'Historial conversación: '.join(conversation_history +   ['| Nueva pregunta:'+input_usuario])

    _input = prompt.format_prompt(question=input_text, system_question=systemQuestion)
    output = chat_model(_input.to_messages())
    
    conversation_history.append(input_usuario)
   
    conversations[user_id] = conversation_history

    return (output.content)

        
    try:
        parsed = parser.parse(output.content)
    except OutputParserException as e:
        new_parser = OutputFixingParser.from_llm(
            parser=parser,
            llm=ChatOpenAI()
        )
        parsed = new_parser.parse(output.content)
    print(parsed)
    return parsed

def singleConsulta(user_id, systemQuestion, input_usuario, clearHistory):
    """Generar una respuesta utilizando ChatGPT con manejo de historial."""
    
    # Manejo del historial según el parámetro clearHistory
    if clearHistory == '1':
        print("Borrando histórico de conversación.")
        conversations[user_id] = []
    else:
        print("Manteniendo histórico de conversación.")
    
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
    except OutputParserException as e:
        print(f"Error al procesar la respuesta: {e}")
        return {"error": "Error al generar la respuesta del modelo."}
    except Exception as e:
        print(f"Error inesperado: {e}")
        return {"error": "Ha ocurrido un error inesperado."}

    # Actualizar el historial de conversación
    #conversation_history.append(input_usuario)
    #conversations[user_id] = conversation_history

    # Devolver la respuesta del modelo
    return {"response": output.content}
        
