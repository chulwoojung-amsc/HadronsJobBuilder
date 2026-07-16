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
from .common import getUserInput, provideInformationToUser, queryYesNo, prettyPrintPydantic, getStructuredResponse, Print as AgentPrint, Input as AgentInput
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

class AgentOutput(BaseModel):
    """Structured output for the agent"""    
    question_to_user: str = Field("", description="A question posed to the user")
    answer_to_user: str = Field("", description="An answer to a question posed by the user")
    scratchpad_note: str = Field("", description="Notes kept by the agent, appended to the current scratchpad")
    params_struct: str = Field(..., description="The current draft of the output JSON structure")
    done: bool = Field(..., description="The agent's workflow is complete")


class AgentState:
    def reset(self):                
        self.scratch = []
        self.params_struct = ""
        
    def __init__(self):
        self.reset()

agent_state = AgentState()                

def listEnumerateStr(lst)->str:
    out = ""
    for i,v in enumerate(lst):
        out = out + f"{i} : {v}\n"
    return out
          
    
def parameterAgent(llm_model, structured_output_model : BaseModel,
                   role: str, tools,
                   tool_rules : List[str] = [],
                   parameter_rules : List[str] = [],
                   input_messages = [ HumanMessage("Start your workflow") ],
                   additional_user_query_rules = [],
                   additional_workflow_termination_rules: str | None = None,
                   validator : Callable | None = None                   
                   ):

    agent_state.reset()
   
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
    
    The overall goal of your conversation is to aid the user in choosing values for each for the fields in the {output_type_name} JSON structure. 
      - The schema is provided in the "Schema for {output_type_name}" section below.
      - Your current working draft of this structure is provided in the "Current {output_type_name} params struct" section below. 

    On each turn of the conversation you must respond with structured output in the AgentOutput schema:
        {AgentOutput.model_json_schema() }
    
    In this output you must always record the currently-known contents of the output {output_type_name} JSON structure in the "params_struct" field. Refer to the "Recording JSON output" rules below for how to format this parameter.
    
    Each time you receive a message from the user that is not a question, review your current knowledge of the required parameters for the output {output_type_name} structure. 
        - If all required parameters have been specified, set the "done" parameter in your output to True indicating that your workflow is complete
        - Otherwise set the "done" parameter to False. 

    If your workflow is not complete (i.e. not all required parameters have been specified) you must continue to ask questions to obtain the missing parameters:    
        - Use the output field "answer_to_user" to answer a question that the user posed, if any. If the user asks a question, your response must contain an answer.
        - Use the output field "question_to_user" to ask a question.
      These outputs will be sent to the user and their response will be contained in the next message you receive. Follow the "Question/answer output rules" below when formating these outputs.
      
    Use the "scratchpad_note" field of your output to record notes to yourself (these are not visible to the user). Refer to the scratchpad rules below for appropriate content.
            
    -----------------------------------------
    Question and answer output rules
    -----------------------------------------
    - If your workflow is done and you have set the "done" parameter in your output, never output an answer or a question.
    - If your workflow is not done, your output to the user *must* include either:
      1) ONE question
      2) ONE answer to the user's previous question AND ONE further question    
    - NEVER output text not intended for the user such as notes-to-self.    
    - Never ask more than one question at a time. Always wait for the user to respond before asking your next question.
    - If the user has not responsed to your question, do not think ahead to the next question. Wait for the user to respond.
    
    - Never ask if the user wants to specify a parameter; assume that the user wants to specify all parameters
    - If the user asks you to choose a value for a parameter, explain to them via the "answer_to_user" field that you cannot decide on parameter values, you can only make suggestions, then repeat the question in the "question_to_user" field.
    - Be brief and to the point with your question, and do not ask for more than one value in a single question.
    - If you ask a question where the user is asked to choose between a set of known options, first obtain the list of options (calling any appropriate tools) then list those options alongside the question in your response. If there are more than 6 choices, list only the first 6 and indicate that there are more options.
    - If the user responds to a query with an invalid response, your response should explain that the choice is invalid and ask the question again. Never ask a question about the next field without a valid response to the current field.
    - Instead of answering your question about a parameter, the user might respond to your query with a question of their own. If this occurs:
         - Answer the user's question in the "answer_to_user" field of your output
         - In the "question_to_user" field you must repeat the original question about the parameter but include a statement indicating that they can ask follow-up questions
{promptStringList(additional_user_query_rules,4)}

    ------------------------------------------------------------------------------------------------------------
    General Parameter Rules:
    These rules describe how you should obtain values for parameters in the output {output_type_name} structure
    ------------------------------------------------------------------------------------------------------------
    - Obtain the values for the parameters in the order they appear in {output_type_name}    
    - If the parameter rules specify that *you* should choose or set the value of a specific parameter yourself never ask the user about this parameter.
    - **Never** guess a parameter that should be provided by the user. These values should always be obtained from the user. Never record such a parameter value unless it has been explicitly provided by the user.
    - If a parameter has a default, you must suggest that value to the user when asking your question about the parameter. Never assume a value without asking.    
    - If there is only one option for a parameter you must use that value. The first time this choice appears in your output you MUST also tell the user that you have made this choice in the "answer_to_user" field.

    -------------------------------------------
    Tool Rules:    
    -------------------------------------------
    - If a tool provides a list of valid responses, only accept values from among that list as valid choices by the user. If you list the values, ensure you only list those returned by the tool; never make up entries.
{promptStringList(tool_rules,4)}

    -------------------------------------------
    Scratchpad rules:
    -------------------------------------------
    - Use the scratchpad to take notes of all choices made by the user.
    - You must write a note for *every* user decision
    - Do not record a note if the user asks you a question.
    - Do not record questions that the user asks to you or notes on your responses. Only record choices.
    - The note must include all choices that the user has made that are not currently noted in the scratchpad
    - Always write a note if the user has made a choice of a parameter value, even if you don't yet have all the parameter values. Just record what you have
    - Ensure that you also take note of the context. For example, if you are recording parameter choices for a specific propagator, note which propagator thse values correspond to.
    - Do not repeat information in the scratchpad. Before populating "scratchpad_note" , check the current scratchpad content and only add information if the scratchpad either does not contain it or the new content supercedes the existing.
    - You can also use the scratchpad to record TODO notes for yourself to help you plan.
    - The scratchpad is also available to the automated validation steps performed after your workflow. You can thus use the scratchpad to respond to validation or missing parameter errors to clarify

    -------------------------------------------
    Recording JSON output rules:
    These rules apply to the {output_type_name} structure you must record in the "params_struct" field of your output
    ------------------------------------------    
    - As your conversation with the user progresses, you must record the current state of the output {output_type_name} JSON using in the "params_struct" field. Never specify the parameters of a different data structure, only {output_type_name}.
    - The structure must follow the {output_type_name} schema for all fields and types, with the exception of unknown parameters. You must include all fields, even if they can have default values.
    - For parameters that the user has not yet specified you must include the field but assign the value "<UNKNOWN>", even if the parameter is not a string parameter.
    - You must output the updated structure in every response, even if some parameters are still unknown.
    - You must follow all rules (general and specific) provided in this prompt regarding the {output_type_name} parameters you record. 
    - If recording a parameter that belongs to one of a list of structure instances and you don't yet know how many instances will be needed, instantiate a single instance and record the parameter there.
    
{param_rules_header}    

{promptStringList(parameter_rules,4)}

    -------------------------------------------
    Rules for dealing with validation failures
    -------------------------------------------
    If you receive a message saying "Your previous response failed validation", perform the following:
    -You must explain that you received an error
    -If the reason is due to an obvious typing error by the user, correct that error and explain to the user what you corrected, then terminate your workflow. Do not repeat questions for parameters not associated with the validation error.
    -If the solution to the validation error is not clear, explain to the user the nature of the error and ask them to provide a solution
    
    ------------------------------
    Schema for {output_type_name} 
    ------------------------------
    Follow this schema for your "params_struct" output
    """ + json.dumps(structured_output_model.model_json_schema())
    
    config = {"configurable": {"thread_id": "1", "stream" : False}}

    @dynamic_prompt
    def system_prompt(request: ModelRequest) -> str:
        prompt = sys + f"""
---------------------------
Current scratchpad contents
---------------------------        
{listEnumerateStr(agent_state.scratch)}

-----------------------------------------
Current {output_type_name} params struct
-----------------------------------------
{agent_state.params_struct}        
        """
        return prompt
    
    all_tools = tools.copy()
    agent = create_agent(model=llm_model, tools=all_tools, middleware=[system_prompt], response_format=AgentOutput)

    user_interactions = input_messages.copy()
    accepted = False
    obj = None

    while(accepted == False):
        #Invoke the agent
        try:
            resp = agent.invoke({ "messages": user_interactions }, config=config)
            resp_struct = getStructuredResponse(resp, AgentOutput)
        except Exception as e:
            user_interactions.append(HumanMessage(f"Encountered an error: {e}"))
            continue

        #Check it followed the rules about questions/answers
        if resp_struct.done and ( len(resp_struct.question_to_user) > 0 or  len(resp_struct.answer_to_user) > 0 ):
            user_interactions.append(HumanMessage("You cannot answer or ask a question if 'done' is set to True"))
            print("DONE TRUE BUT QUESTION",resp_struct.question_to_user,"OR ANSWER",resp_struct.answer_to_user)
            continue
        if not resp_struct.done and len(resp_struct.question_to_user) == 0:
            user_interactions.append(HumanMessage("Your response must include a question unless you are done"))
            print("NO QUESTION IN RESPONSE")
            continue
        if len(resp_struct.params_struct) == 0:
            user_interactions.append(HumanMessage(f"Your response must include your current working draft of the {output_type_name} schema in 'params_struct'"))
            print("NO PARAMS_STRUCT IN RESPONSE")
            continue

        print("OUTPUT", prettyPrintPydantic(resp_struct), "\n\n" )

        if resp_struct.done:
            obj = structured_output_model.model_validate_json(resp_struct.params_struct)            
            
            #Automatic validation
            if validator is not None:                                                           
                valid = validator(obj)
                if not valid[0]:
                    print("VALIDATION FAIL",valid)
                    user_interactions.append(HumanMessage(f"Your previous response failed validation due to: {valid[1]}"))
                    continue
                
            #Human validation            
            accepted = queryYesNo("Is the following correct?", "\n" + prettyPrintPydantic(obj))
            
            if(accepted == False):
                reason = AgentInput("Explain what is wrong: ")
                agent_state.done = False
                user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))
                continue
            else:
                break
        
        else:
            #Append any notes to the scratchpad
            if len(resp_struct.scratchpad_note) > 0:
                agent_state.scratch.append(resp_struct.scratchpad_note)

            #Set the working params struct for the next round of prompting
            agent_state.params_struct = resp_struct.params_struct

            #Compose the AI message to the user
            ai_msg = resp_struct.answer_to_user + ("\n\n" if len(resp_struct.answer_to_user) > 0 else "") + resp_struct.question_to_user

            #Add it to the message history
            user_interactions.append(AIMessage(ai_msg))

            #Obtain the user response
            user_resp = AgentInput(ai_msg)
            user_interactions.append(HumanMessage(user_resp))
    return obj
