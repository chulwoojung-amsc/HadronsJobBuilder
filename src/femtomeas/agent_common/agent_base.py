from typing import Tuple

from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter, PositiveFloat, PositiveInt, create_model, model_validator
from typing import Literal, Union, List, Optional, Tuple, Any
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState, dynamic_prompt, ModelRequest
import json
from .common import getUserInput, provideInformationToUser, queryYesNo, prettyPrintPydantic, Print as AgentPrint, Input as AgentInput
from femtomeas.workflow_manager.api_general import getKnownMachines, getUserAccountProjects, getMachineQueues
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime
import re

from typing import Callable
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    AgentState,
    ExtendedModelResponse
)
from langgraph.types import Command
from typing_extensions import NotRequired

def indentExceptFirst(text, prefix):
    lines = text.splitlines(True)
    return lines[0] + ''.join(prefix + line for line in lines[1:])

def promptStringList(lines : List[str], indent : int = 0):
    if len(lines) == 0:
        return ""
    b = ' ' * indent
    b2 = ' ' * (indent + 2) #for lines below the first in multiline rules
    out = b + '- ' + indentExceptFirst(lines[0], b2)
    for i in range(1, len(lines)):
        out = out + '\n' + b + '- '+  indentExceptFirst(lines[i], b2)
    return out


scratch = []

@tool
def scratchPadWrite(content : str)->None:
    """Append text to the agent scratchpad. The existing content is preserved by this operation.
Args:
    content: The content to add to the scratchpad
    """
    print("SCRATCH WRITE",content)
    scratch.append(content)

def listEnumerateStr(lst)->str:
    out = ""
    for i,v in enumerate(lst):
        out = out + f"{i} : {v}\n"
    return out

    
@tool
def scratchPadRead()->str:
    """Read back the agent scratchpad in the format of a numbered list"""
    out = listEnumerateStr(scratch)
    print("SCRATCH READ",out)
    return out

main_agent_done = False

@tool
def mainAgentDone()->None:
    """Call this tool to signal that the agent workflow is complete. Ensure all parameters have been specified before calling"""
    print("AGENT SIGNALED DONE")
    global main_agent_done
    main_agent_done = True

class ParameterCheck(BaseModel):
    missing_parameters: List[str] =  Field(..., description="The list of parameters for which the users has not specified a value.")
    
def invokeMainAgent(agent, user_interactions, config):
    """Invoke the agent, updating the message list with the output and returning the response message content
    If an error occured, None is returned indicating to continue to the next iteration of the main while loop
    (the messages will contain details of the error for the next iteration)
    """
    try:
        resp = agent.invoke({ "messages": user_interactions }, config=config)
        resp_msg = resp['messages'][-1]

        #gpt-oss-120b with the AmSC LLM services sometimes runs ahead of itself with intervening blocks of reasoning output directly into the content in xml-like tags.
        #However it seems that the content before the first tag is the intended user output.
        if "<reasoning>" in resp_msg.content:
            resp_msg = AIMessage(content=resp_msg.content[:resp_msg.content.find('<reasoning>')])               

        user_interactions.append(resp_msg)

        resp_content = resp_msg.content

        if len(resp_content) == 0:
            user_interactions.append(HumanMessage(f"Your previous had no content, try again"))
            return None

        return resp_content
        
    except Exception as e:
        user_interactions.append(HumanMessage(f"Encountered an error: {e}"))
        return None


# def invokeAgentWithStructuredOutput(agent, messages, output_format):
#     """It is assumed that the system prompt contains instructions to output in this format"""
#     internal_context = messages.copy()
    
#     parsed=False
#     while not parsed:
#         try:
#             resp = agent.invoke({"messages" : internal_context})
#             content = resp["messages"][-1].content
            
#             print("invokeAgentWithStructuredOutput got",content)
#             obj = output_format.model_validate_json(content)
            
#             #obj = resp['structured_response']
#             parsed=True
                
#         except Exception as e:
#             print(f"invokeAgentWithStructuredOutput to { output_format.__name__ } PARSE ERROR {e}")
#             internal_context.append( HumanMessage(f"Encountered an error: {e}") )
    
#     return obj
            
def checkAllParametersSpecified(check_complete_agent, user_interactions):
    """
    Check all the parameters have been specified
    If so, return True
    If not, user_interactions will be appended with a message specifying the missing parameters, and False will be returned
    """

    internal_context = user_interactions.copy()
    
    #Use an agent to check that it really is done
    
    parsed=False
    while not parsed:
        try:
            resp = check_complete_agent.invoke({"messages" : internal_context})
            obj = resp['structured_response']
            parsed=True
                
        except Exception as e:
            print("CHECK COMPLETE AGENT PARSE ERROR",e)
            internal_context.append( HumanMessage(f"Encountered an error: {e}") )

    #obj = invokeAgentWithStructuredOutput(check_complete_agent, user_interactions, ParameterCheck)
    
    if len(obj.missing_parameters) > 0:
        print("CHECK COMPLETE AGENT FOUND MISSING PARAMETERS:",obj.missing_parameters)
        user_interactions.append(HumanMessage(f"The following parameters have not yet been specified by the user: { obj.missing_parameters }. Work with the user to determine these parameters."))
        return False
    else:
        return True


def finalAgentStructuredOutputParse(final_output_agent, user_interactions, structured_output_model):
    #Formally parse the message chain into structured output
    #return invokeAgentWithStructuredOutput(final_output_agent, user_interactions, structured_output_model)
    
    parsed=False
    internal_context = user_interactions.copy()

    while not parsed:
        try:
            resp = final_output_agent.invoke({ "messages": internal_context })
            obj = resp['structured_response']
            parsed=True
        except Exception as e:
            print("FINAL OUTPUT AGENT PARSE ERROR",e)
            internal_context.append( HumanMessage(f"Encountered the following error. Try again: {e}") )
    return obj


    
    
def parameterAgent(llm_model, structured_output_model : BaseModel,
                   role: str, tools,
                   tool_rules : List[str] = [],
                   parameter_rules : List[str] = [],
                   input_messages = [ HumanMessage("Start your workflow") ],
                   additional_user_query_rules = []
                   ):

    global main_agent_done
    main_agent_done = False
    scratch.clear()
    
    output_type_name = type(structured_output_model).__name__
    param_rules_header = """    -------------------------------------------
    Additional rules for specific parameters:   
    -------------------------------------------
    """ if len(parameter_rules) > 0 else ""

    
    
    """
    role: System prompt text describing the agent's role. Follows from "You are a conversational agent responsible for..."
    tool_rules: Rules for using explicit tools as a string list.
    parameter_rules: Rules for handling specific parameters

    For rules, do *not* include any formatting for the beginning of the first line. Rules can be multi-line and will be indented appropriately
    """

    sys = f"""
    You are a conversational agent responsible for {role}

    To output text to the user, output a message containing the text for the user (questions, answers). The user's response will be contained in the next message. Your output *must* include either
    1) ONE question
    2) ONE answer to the user's previous question AND ONE further question
    NEVER output text not intended for the user such as notes-to-self. See the output rules below.
    
    The overall goal of your conversation is to aid the user in choosing values for each for the fields in the schema {output_type_name} (provided below).
  
    To formulate the response, you are free to call appropriate tools to obtain extra information to help the user.

    To identify if the user has chosen a value, confirm that the user's response is a statement describing a valid value for the parameter.

    You must keep notes of choices made by the user using the scratchPadWrite tool. Refer to the scratchpad rules below.
    
    Obtain the values for the parameters in the order they appear in {output_type_name}
    
    If the user asks a question, you must answer it before asking any further questions

    Once the user has specified all parameters, call the mainAgentDone tool to signal completion. Never output the completed JSON. Never ask the user to confirm the complete set of parameters.

    DO NOT PERFORM ANY PLANNING STEPS

    You must adhere to the following rules:      
    -----------------------------------------
    Output rules
    -----------------------------------------
    - Never ask more than one question at a time. Always wait for the user to respond before asking your next question.
    - In your response to the user, *never* include any reasoning, chain-of-thought or notes-to-self. Only ever include a single question or an answer followed by a question. For example, never output "We need to wait for user response."
    - If you do decide to include reasoning in your output despite these explicit instructions *not to*, you may receive an error message. Do not apologize, simply generate the correct output
    - If the user has not responsed to your question, do not think ahead to the next question. Wait for the user to respond.
    - Do not output the completed JSON schema. Simply terminate your workflow by calling the mainAgentDone tool when all the parameters have been chosen by the user.
    
    -------------------------------------------
    General Parameter Rules:
    -------------------------------------------
    - If the parameter rules specify that *you* should choose or set the value of a specific parameter yourself never ask the user about this parameter.
    - **Never** guess a parameter that should be provided by the user. These values should always be obtained from the user. Never record such a parameter value unless it has been explicitly provided by the user. Follow the User Query rules below for questions to the user.
    - If a parameter has a default, you may suggest that value to the user but you must not assume a value without asking.    

    -------------------------------------------
    Tool Rules:    
    -------------------------------------------
    - Use scratchPadWrite to store user choices and other notes.
    - The current scratchpad content is included in your system prompt
    - If a tool provides a list of valid responses, only accept values from among that list as valid choices by the user. If you list the values, ensure you only list those returned by the tool; never make up entries.
{promptStringList(tool_rules,4)}

    -------------------------------------------
    Scratchpad rules:
    -------------------------------------------
    - Use the scratchpad to take notes of all choices made by the user.
    - You must write a note for *every* user response that contains a choice
    - Do not call scratchPadWrite if the user asks you a question.
    - Do not record questions that the user asks to you or notes on your responses. Only record choices.
    - The note must include all choices that the user has made that are not currently noted in the scratchpad
    - Always write a note if the user has made a choice of a parameter value, even if you don't yet have all the parameter values. Just record what you have
    - Ensure that you also take note of the context. For example, if you are recording parameter choices for a specific propagator, note which propagator thse values correspond to.
    - Do not repeat information in the scratchpad. Before calling scratchPadWrite, check the current scratchpad content and only add information if the scratchpad either does not contain it or the new content supercedes the existing.
    - The scratchpad is also available to the automated validation steps performed after your workflow. You can thus use the scratchpad to respond to validation or missing parameter errors to clarify
    
    -------------------------------------------
    User Query rules:
    -------------------------------------------
    - Never ask if the user wants to specify a parameter; assume that the user wants to specify all parameters
    - Be brief and to the point with your question, and do not ask for more than one value in a single question.
    - If you ask a question where the user is asked to choose between a set of known options, first obtain the list of options (calling any appropriate tools) then list those options alongside the question in your response. If there are more than 6 choices, list only the first 6 and indicate that there are more options.
    - If the user responds to a query with an invalid response, your response should explain that the choice is invalid and ask the question again. Never ask a question about the next field without a valid response to the current field.
    - Instead of answering your question about a parameter, the user might respond to your query with a question of their own. If this occurs:
         - On the first line of your response, answer the user's question
         - On a separate line repeat the original question about the parameter but include a statement indicating that they can ask follow-up questions
{promptStringList(additional_user_query_rules,4)}
    
{param_rules_header}    

{promptStringList(parameter_rules,4)}

    -------------------------------------------
    Rules for dealing with validation failures
    -------------------------------------------
    If you receive a message saying "Your previous response failed validation", perform the following:
    -You must explain that you received an error
    -If the reason is due to an obvious typing error by the user, correct that error and explain to the user what you corrected, then terminate your workflow. Do not repeat questions for parameters not associated with the validation error.
    -If the solution to the validation error is not clear, explain to the user the nature of the error and ask them to provide a solution
    
    -------------------------------------------
    Schema for fields you must populate
    -------------------------------------------

    The fields you must obtain values for are listed in the following schema:
    """ + json.dumps(structured_output_model.model_json_schema())

    
    config = {"configurable": {"thread_id": "1", "stream" : False}}

    @dynamic_prompt
    def system_prompt(request: ModelRequest) -> str:
        prompt = sys + f"""
---------------------------
Current scratchpad contents
---------------------------        
{listEnumerateStr(scratch)}"""
        #print("SCRATCHPAD", listEnumerateStr(scratch))
        return prompt
    
    all_tools = tools.copy() + [scratchPadWrite,mainAgentDone]
    agent = create_agent(model=llm_model, tools=all_tools, middleware=[system_prompt])


    check_param_rules_header = """    -------------------------
    Specific parameter rules
    -------------------------
    These rules apply to specific parameters. If the rule for a parameter states that the value should be chosen by the agent, do not include this parameter in your checks.
    """ if len(parameter_rules) > 0 else ""

    
    check_complete_sys = """
    You must check the message history and the scratchpad contents to determine if the user has provided answers to all fields in the following schema:
    """ + json.dumps(structured_output_model.model_json_schema()) + f"""
    Identify all parameters that the user has not specified and output them into the missing_parameters field of your output.
    If the user has specified all parameters, set missing_parameters to an empty list

    -----------------------
    Scratchpad content
    -----------------------
    {json.dumps(scratch)}
    
{check_param_rules_header}    

{promptStringList(parameter_rules,4)}

    ------------------------
    Output rules
    ------------------------       
    Your output must be provided according to the following schema:
    """ + json.dumps(ParameterCheck.model_json_schema())
    

    check_complete_agent = create_agent(model=llm_model, system_prompt=check_complete_sys, response_format=ParameterCheck)
   
    output_sys = f"""
    You are an agent responsible for inserting information from the user into a structured output with the schema below.

    Use your message history and the scratchpad content to identify the user's decision for a parameter

    Determine whether the user has chosen a value by identifying whether the user's response is a statement describing a valid value for the parameter.

    You must identify the user's decision for all parameters
    - If the rules for a specific parameter state that you must choose or specify a value, do this now based on the rule and the message history
    - Otherwise, **never** guess a parameter. These values should always be obtained from the user's responses.

    -----------------------
    Scratchpad content
    -----------------------
    {json.dumps(scratch)}
    
{check_param_rules_header}    

{promptStringList(parameter_rules,4)}
    
    ------------------------
    Output rules
    ------------------------
    You must return JSON formated output in the following schema:
    """ + json.dumps(structured_output_model.model_json_schema()) 

    
    final_output_agent = create_agent(model=llm_model, system_prompt=output_sys, response_format=structured_output_model)

    user_interactions = input_messages.copy()
    accepted = False
    obj = None
    
    while(accepted == False):
        resp_content = invokeMainAgent(agent, user_interactions, config)
                    
        if main_agent_done:
            #Use an agent to check that it really is done
            if not checkAllParametersSpecified(check_complete_agent, user_interactions):
                main_agent_done = False
                continue
    
            #Formally parse the message chain into structured output
            obj = finalAgentStructuredOutputParse(final_output_agent, user_interactions, structured_output_model)

            #Automatic validation
            try:            
                valid = obj.check()
                if not valid[0]:
                    print("VALIDATION FAIL",valid)
                    user_interactions.append(HumanMessage(f"Your previous response failed validation due to: {valid[1]}"))
                    continue
            except Exception as e:
                if not isinstance(e, AttributeError):
                    raise Exception(f"Validation threw an error, {e}")
                
            #Human validation
            output = f"Obtained:\n" + prettyPrintPydantic(obj)
            AgentPrint(output)
            
            accepted = queryYesNo("Is this correct?")
            
            if(accepted == False):
                reason = AgentInput("Explain what is wrong: ")
                main_agent_done = False
                user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))
                continue
            else:
                break
            
        elif resp_content == None:
            continue
            
        else:
            #Obtain the user response
            user_resp = AgentInput(resp_content)
            user_interactions.append(HumanMessage(user_resp))
    return obj
