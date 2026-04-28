from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

import json
import io
import re

from .print_pydantic_markdown import pydantic_to_markdown

def storeGetList(key, store):
    l = store.get( ("ns",),key)
    if l == None:
        return []
    ll = l.value
    assert isinstance(ll, list)
    return ll

def storeListAppend(key, value, store):
    ll = storeGetList(key,store)
    ll.append(value)
    store.put(("ns",), key, ll)


##Chatbot IO controls
def cmdlinePrint(*args, **kwargs):
    print(*args, *kwargs)
def cmdlineInput(query):
    return input(query + " : ").strip()

output_style = "plain" #supports markdown for prettier printing


#Control which functions are used for text input and output to the chatbot
print_func = cmdlinePrint
input_func = cmdlineInput
   
def Print(*args, **kwargs):
    global print_func
    print_func(*args, *kwargs)
        
def Input(query):
    global input_func
    return input_func(query)


def prettyPrintPydantic(instance)->str:
    global output_style
    if output_style == "plain":
        if isinstance(instance, list):
            out = ""
            for i, r in enumerate(instance):
                out = out+str(r)
                if i != len(instance)-1:
                    out = out + "\n"
            return out
        else:
            return str(instance) #default repr 
    elif output_style == "markdown":
        #sval = f"```json\n{instance.model_dump_json(indent=4)}\n```"
        return pydantic_to_markdown(instance, mode="table")
    else:
        assert 0
        
@tool
def getUserInput(query: str) -> str:
    """
    Get a response from the user to a query.
    Args:
       query: The question to pose to the user.
    Return:
       Return the user's response
    """
    query = re.sub(r'[\s:;\$]*$', "", query)   
    return Input(query)

@tool
def provideInformationToUser(description: str):
    """
    Provide some text information to the user
    """
    Print(description)

def queryYesNo(query: str)->bool:
    result = ""
    while(result not in ["y","n"]):    
        result = Input(query + " [y/n]")
        print("QUERY YES/NO RECEIVED",result,"VALID ?", result in ["y","n"] )

    print("QUERY YES/NO GOT VALID RESPONSE")
    return True if result == "y" else False
        

def callModelWithStructuredOutput(model, sys_prompt : str, other_messages : list[BaseMessage], schema : BaseModel, use_langchain_structured_output_method = True)-> BaseModel:
    """
    Wrapper function for calling an LLM model with structured output
    This should not be necessary but either LangChain or the model providers can be very flakey
    """
    if use_langchain_structured_output_method:
        messages = [ SystemMessage(sys_prompt) ] + other_messages
        ret = None
        retries=0
        while ret == None:
            if retries > 10:
                raise Exception(f"Obtained a null result from the model. Attempted 10 times. If this happens it might be because the last message is an AIMessage not a HumanMessage. Models seem to implicitly rely on this! Message history: {messages}") 
                
                
            #ret = model.with_structured_output(schema, method="function_calling").invoke(messages) #using method=json_schema (default I think) produces garbled output for the oss-120b model for some reason; function_calling seems more reliable

            ret = model.with_structured_output(schema).invoke(messages)
            retries+=1
            
        return ret
        
    else:
        sys_prompt += """
Your output must be in JSON format. Use the following schema:
""" + json.dumps(schema.model_json_schema())
        
        messages = [ SystemMessage(sys_prompt) ] + other_messages
        obj = None
        valid=False
        tries=0
        while(valid == False):
            tries += 1
            if tries == 10:
                break
            
            try:
                response = model.invoke(messages)
                obj = schema.model_validate_json(response.content)
                valid=True
            except Exception as e:
                messages.append(HumanMessage(f"Your previous response did not parse correctly for the following reason: {str(e)}"))

        if valid == False:
            raise Exception("Not able to parse output correctly within 10 tries")

        return obj

    
def spaceSeparateSeq(seq):
    r = ""
    for i, v in enumerate(seq):
        r = r + str(v) + ("" if i==len(seq)-1 else " ")
    return r
