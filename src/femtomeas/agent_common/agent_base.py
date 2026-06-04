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


class AgentState:
    def reset(self,schema):
        self.schema = schema
        self.done = False
        self.result = None
        self.params_struct = ""
        self.scratch = []
        
    def __init__(self):
        self.reset(None)

agent_state = AgentState()                

@tool
def scratchPadWrite(content : str)->None:
    """Append text to the agent scratchpad. The existing content is preserved by this operation.
Args:
    content: The content to add to the scratchpad
    """
    print("SCRATCH WRITE",content)
    agent_state.scratch.append(content)

def listEnumerateStr(lst)->str:
    out = ""
    for i,v in enumerate(lst):
        out = out + f"{i} : {v}\n"
    return out

    
@tool
def scratchPadRead()->str:
    """Read back the agent scratchpad in the format of a numbered list"""
    out = listEnumerateStr(agent_state.scratch)
    print("SCRATCH READ",out)
    return out




@tool
def setParamsStruct(contents: str)->None:
    """Set the contents of the output params struct
Args:
  contents: The structure in JSON format
"""
    agent_state.params_struct = contents
    print("PARAMS STRUCT WRITE",contents)

    
@tool
def mainAgentDone(result: str)->None:
    """Call this tool once with the final JSON output structure. Ensure all parameters have been specified before calling
Args:
  result: The JSON-formatted output structure"""

    print("AGENT SIGNALED DONE", result)

    agent_state.result = agent_state.schema.model_validate_json(result)
    print("SCHEMA",agent_state.schema,"OUTPUT",agent_state.result)
    agent_state.done = True


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

           
    
def parameterAgent(llm_model, structured_output_model : BaseModel,
                   role: str, tools,
                   tool_rules : List[str] = [],
                   parameter_rules : List[str] = [],
                   input_messages = [ HumanMessage("Start your workflow") ],
                   additional_user_query_rules = [],
                   additional_workflow_termination_rules: str | None = None,
                   output_check_kwargs = {}
                   ):

    agent_state.reset(structured_output_model)
   
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

    To output text to the user, return a message containing the text for the user (questions, answers). The user's response will be contained in the following message. Your responses to the user *must* include either:
    1) ONE question
    2) ONE answer to the user's previous question AND ONE further question
    3) The statement "<DONE>" and nothing else, but only if all parameters have been specified (refer to the rules below). You MUST call the 'mainAgentDone' tool before outputing this response. Never respond with "<DONE>" without also calling the "mainAgentDone" tool.

    NEVER output text not intended for the user such as notes-to-self. See the output rules below.
    
    The overall goal of your conversation is to aid the user in choosing values for each for the fields in the schema {output_type_name} (provided below).

    You must always keep notes of choices made by the user using the scratchPadWrite tool. Refer to the scratchpad rules below.

    You must always record the currently-known contents of the output {output_type_name} JSON structure using the setParamsStruct tool. When your workflow begins the output structure should be empty. Refer to the "Recording JSON output" rules below
    
    Obtain the values for the parameters in the order they appear in {output_type_name}
    
    If the user asks a question, you must answer it before asking any further questions
    
    ---------------------------------------------------------
    Actions to perform once user has specified all parameters
    ---------------------------------------------------------

    Each time you receive a message from the user that is not a question, review your current knowledge of the required parameters for the output structure. If all required parameters have been set, { (additional_workflow_termination_rules + "\nThen ") if additional_workflow_termination_rules is not None else "" }you MUST call the mainAgentDone tool with the complete output structure and respond with "DONE". Perform no other actions. Do NOT inform the user that all parameters have been set prior to calling this tool. Do not wait for confirmation before calling this tool.
        
    You must adhere to the following rules:      
    -----------------------------------------
    Output rules
    -----------------------------------------
    - Your output to the user must include a question.
    - Never ask more than one question at a time. Always wait for the user to respond before asking your next question.
    - In your response to the user, *never* include any reasoning, chain-of-thought or notes-to-self. Only ever include a single question or an answer followed by a question. For example, never output "We need to wait for user response."
    - If you do decide to include reasoning in your output despite these explicit instructions *not to*, you may receive an error message. Do not apologize, simply generate the correct output
    - If the user has not responsed to your question, do not think ahead to the next question. Wait for the user to respond.
    
    -------------------------------------------
    General Parameter Rules:
    -------------------------------------------
    - If the parameter rules specify that *you* should choose or set the value of a specific parameter yourself never ask the user about this parameter.
    - **Never** guess a parameter that should be provided by the user. These values should always be obtained from the user. Never record such a parameter value unless it has been explicitly provided by the user. Follow the User Query rules below for questions to the user.
    - If a parameter has a default, you may suggest that value to the user but you must not assume a value without asking.    

    -------------------------------------------
    Tool Rules:    
    -------------------------------------------
    - Use scratchPadWrite to store user choices and other notes. The current scratchpad content is included in your system prompt
    - Use setParamsStruct to document the current best knowledge of the output struct. The current params struct is included in your system prompt.
    - If a tool provides a list of valid responses, only accept values from among that list as valid choices by the user. If you list the values, ensure you only list those returned by the tool; never make up entries.
{promptStringList(tool_rules,4)}

    -------------------------------------------
    Scratchpad rules:
    -------------------------------------------
    - Use the scratchpad to take notes of all choices made by the user.
    - You must write a note for *every* user decision
    - Do not call scratchPadWrite if the user asks you a question.
    - Do not record questions that the user asks to you or notes on your responses. Only record choices.
    - The note must include all choices that the user has made that are not currently noted in the scratchpad
    - Always write a note if the user has made a choice of a parameter value, even if you don't yet have all the parameter values. Just record what you have
    - Ensure that you also take note of the context. For example, if you are recording parameter choices for a specific propagator, note which propagator thse values correspond to.
    - Do not repeat information in the scratchpad. Before calling scratchPadWrite, check the current scratchpad content and only add information if the scratchpad either does not contain it or the new content supercedes the existing.
    - You can also use the scratchpad to record TODO notes for yourself to help you plan.
    - The scratchpad is also available to the automated validation steps performed after your workflow. You can thus use the scratchpad to respond to validation or missing parameter errors to clarify

    -------------------------------------------
    Recording JSON output rules
    ------------------------------------------    
    - As your conversation with the user progresses, you must record the current state of the output {output_type_name} JSON using the setParamsStruct. Never specify the parameters of a different data structure, only {output_type_name}.
    - The structure must follow the {output_type_name} schema for all fields and types, with the exception of unknown parameters. You must include all fields, even if they can have default values.
    - For parameters that the user has not yet specified you must include the field but assign the value "<UNKNOWN>", even if the parameter is not a string parameter.
    - You must call this tool with the updated structure *every time* the user makes a choice or specifies/decides a parameter value, even if some parameters are still unknown. Never respond to the user without calling this tool first UNLESS you are answering a question from the user.
    - Always call this tool when the user specifies a parameter value.
    - You must follow all rules (general and specific) provided in this prompt regarding the parameters you record. 
    - If recording a parameter that belongs to one of a list of structure instances and you don't yet know how many instances will be needed, instantiate a single instance and record the parameter there.
    
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
    - If you are asking the user to confirm a suggestion you have made, do not ask them to confirm their choice.
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
{listEnumerateStr(agent_state.scratch)}

-----------------------------
Current output params struct
-----------------------------
{agent_state.params_struct}        
        """
        return prompt
    
    all_tools = tools.copy() + [scratchPadWrite,mainAgentDone,setParamsStruct]
    agent = create_agent(model=llm_model, tools=all_tools, middleware=[system_prompt])

    user_interactions = input_messages.copy()
    accepted = False
    obj = None

    while(accepted == False):
        resp_content = invokeMainAgent(agent, user_interactions, config)

        print("AGENT DONE STATE", agent_state.done)
        if agent_state.done:
            obj = agent_state.result
            
            #Automatic validation
            try:            
                valid = obj.check(**output_check_kwargs)
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
                agent_state.done = False
                user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))
                continue
            else:
                break
            
        elif resp_content == None:
            continue

        elif "<DONE>" in resp_content:
            print("REMINDER TO CALL MAINAGENTDONE")
            user_interactions.append(HumanMessage(f"You must call the mainAgentDone tool when you are done"))
        
        else:
            #Obtain the user response
            user_resp = AgentInput(resp_content)
            user_interactions.append(HumanMessage(user_resp))
    return obj
