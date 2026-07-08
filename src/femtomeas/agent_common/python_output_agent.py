from typing import Tuple

from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter, PositiveFloat, PositiveInt, create_model, model_validator
from typing import Literal, Union, List, Optional, Tuple, Any, Callable
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, AgentState, dynamic_prompt, ModelRequest
import json
from .common import queryYesNo, prettyPrintPydantic, prettyPrintPythonCode, getStructuredResponse, callModelWithStructuredOutput, Print as AgentPrint, Input as AgentInput
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
import asteval

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
    code: str = Field(..., description="The current output code")
    done: bool = Field(..., description="The agent's workflow is complete")


class AgentState:
    def reset(self):                
        self.scratch = []
        self.code = ""
        
    def __init__(self):
        self.reset()

agent_state = AgentState()                

def listEnumerateStr(lst)->str:
    out = ""
    for i,v in enumerate(lst):
        out = out + f"{i} : {v}\n"
    return out
          
    
def executeCode(code):
    """Execute the code, returning the symbol table and error list"""
    aeval = asteval.Interpreter()
    aeval.eval(code)
    errors = []
    if len(aeval.error)>0:
        for err in aeval.error:
            e = err.get_error()
            errors.append(f"{e[0]}:{e[1]}\n")
    return aeval.symtable, errors            
    
def executeCodeAndParse(code, model, output_list_name):
    syms, errors = executeCode(code)
    if len(errors) > 0:
        return ([],errors)
    
    parsed = []
    for r in syms[output_list_name]:
        try:
            mval = model.model_validate(r)
        except Exception as e:
            errors.append(str(e))
            mval = None
        parsed.append(mval)
    return (parsed, errors)
    




def parameterAgent(llm_model, structured_output_model : BaseModel, output_list_name: str,
                   role: str, tools,
                   tool_rules : List[str] = [],
                   parameter_rules : List[str] = [],
                   input_messages = [ HumanMessage("Start your workflow") ],
                   additional_user_query_rules = [],
                   additional_workflow_termination_rules: str | None = None,
                   instance_validator : Callable | None = None,
                   group_validator : Callable | None = None
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
    
    The overall goal of your conversation is to write Python code to generate instances of the {output_type_name} structure according to the rules below.
      - The schema is provided in the "Schema for {output_type_name}" section below.
      - Your current working draft of the code is provided in the "Current {output_type_name} code" section below. 

    On each turn of the conversation you must respond with structured output in the AgentOutput schema:
        {AgentOutput.model_json_schema() }
    
    In this output you must always record the current working draft of the output code in for generating {output_type_name} instances in the "code" field. Refer to the "Recording code output" rules below for how to write this code.
    
    Each time you receive a message from the user that is not a question, review your current knowledge of the required parameters for the output {output_type_name} structures. 
        - If your code generates all required instances and populates them with correct values for all required parameters, set the "done" parameter in your output to True indicating that your workflow is complete
        - Otherwise set the "done" parameter to False. 

    If your workflow is not complete (i.e. your code does not instantiate all required instances or sets all required parameters) you must continue to ask questions to obtain the missing parameters and instances:    
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
    - Be brief and to the point with your question, and do not ask for more than one value in a single question.
    - If you ask a question where the user is asked to choose between a set of known options, first obtain the list of options (calling any appropriate tools) then list those options alongside the question in your response. If there are more than 6 choices, list only the first 6 and indicate that there are more options.
    - If the user responds to a query with an invalid response, your response should explain that the choice is invalid and ask the question again. Never ask a question about the next field without a valid response to the current field.
    - Instead of answering your question about a parameter, the user might respond to your query with a question of their own. If this occurs:
         - Answer the user's question in the "answer_to_user" field of your output
         - In the "question_to_user" field you must repeat the original question about the parameter but include a statement indicating that they can ask follow-up questions
{promptStringList(additional_user_query_rules,4)}

    ------------------------------------------------------------------------------------------------------------
    General Parameter Rules:
    These rules describe how you should obtain values for parameters in the output {output_type_name} structures
    ------------------------------------------------------------------------------------------------------------
    - Obtain the values for the parameters in the order they appear in {output_type_name}    
    - The user may either provide you with one or more explicit parameter choices, or they might provide you with instructions on how to instantiate a set of instances with different values, for example by providing a selection criteria.
        - If the user provides a selection criteria:
            - You must write code to generate *all* values that satisfy this criteria. Do this by writing appropriate loops in your code output. Ask the user for the loop bounds if you do not know.     
            - Do not ask the user to provide values that satisfy the criteria or to confirm specific solutions.
            - Do not try to identify solutions to the criteria yourself.
        - *Do not insist* that the user provide explicit values for parameters. 
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
    Recording code output rules:
    These rules apply to the code for generating the {output_type_name} structures you must record in the "code" field of your output
    ------------------------------------------    
    - The code field must contain correct Python inside a string.
    - The code must produce jsonable Python dictionaries according to schema below.
    - Use Python True/False for boolean fields
    - You cannot use any libraries within the code. 
    - The {output_type_name} structure instances must be stored to a list with name '{output_list_name}' that is initialized to an empty list. This list must contain ONLY {output_type_name} structures.
    - NEVER duplicate code or instances from previous sections. Your list must include ONLY the {output_type_name} structures that YOU create.
    - You must use for loops to iterate over parameter values if there are more than 3 values.
    - Try to make the code as short as possible while still remaininng intelligible. For instance:
        - If you have a loop involving {output_type_name} that share multiple parameters, instantiate a base instance outside the loop with those static parameters set, and take copies within the loop to set the values that differ.
        - Even if not in a loop, if different instances have multiple shared parameters, prefer a base instance with copies versus writing out the full set of parameters again.
    - As your conversation with the user progresses, you must write code to create instances of {output_type_name} and fill their parameters in the "code" field. Never write code for any purpose other than instantiating and populating parameters of {output_type_name}.
    - The structure instances must follow the {output_type_name} schema for all fields and types, with the exception of unknown parameters. You must include all fields, even if they can have default values.
    - For parameters that the user has not yet specified you must include the field but assign the value "<UNKNOWN>", even if the parameter is not a string parameter.
    - You must output the updated code in every response, even if some parameters are still unknown.
    - You must follow all rules (general and specific) provided in this prompt regarding the {output_type_name} parameters you record. 
    - If recording a parameter that belongs to one of a list of structure instances and you don't yet know how many instances will be needed, instantiate a single instance and record the parameter there.
    - The code you write must only pertain to creating {output_type_name} structures.
    - Your code cannot call any of the tools you have access to.
    
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
    Your "code" output must produce jsonable Python dictionaries according to the following schema:
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
Current code for generating {output_type_name} params structs
-----------------------------------------
{agent_state.code}        
        """
        return prompt
    
    all_tools = tools.copy()
    agent = create_agent(model=llm_model, tools=all_tools, middleware=[system_prompt], response_format=AgentOutput)

    user_interactions = input_messages.copy()
    accepted = False
    code = None

    print("AGENT START")
    while(accepted == False):
        #Invoke the agent
        try:
            resp = agent.invoke({ "messages": user_interactions }, config=config)
            resp_struct = getStructuredResponse(resp, AgentOutput)
        except Exception as e:
            print("EXCEPTION",e)
            user_interactions.append(HumanMessage(f"Encountered an error: {e}"))
            continue

        #Check it followed the rules about questions/answers
        # if resp_struct.done and ( len(resp_struct.question_to_user) > 0 or  len(resp_struct.answer_to_user) > 0 ):
        #     user_interactions.append(HumanMessage("You cannot answer or ask a question if 'done' is set to True"))
        #     print("DONE TRUE BUT QUESTION",resp_struct.question_to_user,"OR ANSWER",resp_struct.answer_to_user)
        #     continue
        if resp_struct.done and len(resp_struct.question_to_user) > 0:
            user_interactions.append(HumanMessage("You cannot ask a question if 'done' is set to True"))
            print("DONE TRUE BUT QUESTION",resp_struct.question_to_user)
            continue
        if not resp_struct.done and len(resp_struct.question_to_user) == 0:
            user_interactions.append(HumanMessage("Your response must include a question unless you are done"))
            print("NO QUESTION IN RESPONSE")
            continue
        if len(resp_struct.code) == 0:
            user_interactions.append(HumanMessage(f"Your response must include your current working draft of the code for generating {output_type_name} structures 'code'"))
            print("NO CODE IN RESPONSE")
            continue

        print("OUTPUT", prettyPrintPydantic(resp_struct), "\n\n" )

        if resp_struct.done:
            print("DONE")

            print("RAW CODE",resp_struct.code)
            aeval = asteval.Interpreter()
            aeval.eval(resp_struct.code)
            errors = ""
            if len(aeval.error)>0:
                for err in aeval.error:
                    e = err.get_error()
                    errors = errors + f"{e[0]}:{e[1]}\n"
            if len(errors) > 0:
                print("ERRORS", errors)
                user_interactions.append(HumanMessage(f"Running your code produced error(s): {errors}"))      
                continue
            if output_list_name not in aeval.symtable:
                print(f"SYMTABLE DOESNT CONTAIN {output_list_name}", aeval.symtable.keys())
                user_interactions.append(HumanMessage(f"Your code must create a list named '{output_list_name}'"))
                continue 

            #Automatic validation
            vfail = False
            parsed_results = [] #not kept, just used for validation
            for r in aeval.symtable[output_list_name]:                                
                try:
                    mval = structured_output_model.model_validate(r)
                except Exception as e:
                    print("LIST INSTANCE VALIDATION ERROR",r,e)
                    user_interactions.append(HumanMessage(f"There was an error when validating one of the model instances with Pydantic. Instance {r}, error {e}"))                  
                    vfail = True
                    break
                parsed_results.append(mval)

                if instance_validator is not None:            
                    valid = instance_validator(mval)
                    if not valid[0]:
                        print("AUTO INSTANCE VALIDATION FAIL",valid)
                        user_interactions.append(HumanMessage(f"There was an error when validating one of the model instances for correctness: {valid[1]}"))
                        vfail = True
                        break                

            if vfail:
                continue

            if group_validator is not None:
                valid = group_validator(parsed_results)
                if not valid[0]:
                    print("AUTO GROUP VALIDATION FAIL",valid)
                    user_interactions.append(HumanMessage(f"There was an error when validating the collection of model instances for correctness: {valid[1]}"))
                    continue

            #Allow the agent to make a closing statement, e.g. in response to fixing a previous validation error
            if len(resp_struct.answer_to_user) > 0:                                            
                user_interactions.append(AIMessage(resp_struct.answer_to_user))
                AgentPrint(resp_struct.answer_to_user)

            #Human validation
            accepted = queryYesNo("Is the following code correct?", "\n" + prettyPrintPythonCode(resp_struct.code))            

            if(accepted == False):
                reason = AgentInput("Explain what is wrong: ")
                agent_state.done = False
                user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))
                continue
            else:
                code = resp_struct.code
                break
        
        else:
            print("RESP")

            #Append any notes to the scratchpad
            if len(resp_struct.scratchpad_note) > 0:
                agent_state.scratch.append(resp_struct.scratchpad_note)

            #Set the working params struct for the next round of prompting
            agent_state.code = resp_struct.code

            #Compose the AI message to the user
            ai_msg = resp_struct.answer_to_user + ("\n\n" if len(resp_struct.answer_to_user) > 0 else "") + resp_struct.question_to_user

            #Add it to the message history
            user_interactions.append(AIMessage(ai_msg))

            #Obtain the user response
            user_resp = AgentInput(ai_msg)
            user_interactions.append(HumanMessage(user_resp))
    return code










class ParameterModelCallOutput(BaseModel):
    code: str = Field(..., description="The output code")

def parameterModelCall(llm_model, structured_output_model : BaseModel, output_list_name : str, 
                   role: str,
                   parameter_rules : List[str] = [],
                   input_messages = [ HumanMessage("Start your workflow") ],
                   instance_validator : Callable | None = None,
                   group_validator : Callable | None = None
                   )-> str:
    """This function is similar to the conversational agent but is intended for single-pass model calls

    role: System prompt text describing the agent's role. Follows from "You are responsible for..."
    parameter_rules: Rules for handling specific parameters

    For rules, do *not* include any formatting for the beginning of the first line. Rules can be multi-line and will be indented appropriately
    """

    output_type_name = type(structured_output_model).__name__
    param_rules_header = """    -------------------------------------------
    Additional rules for specific parameters:   
    -------------------------------------------
    """ if len(parameter_rules) > 0 else ""

    sys = f"""
    You are responsible for {role}
    
    Your goal is to write Python code to generate instances of the {output_type_name} structure according to the rules below.
      - The schema is provided in the "Schema for {output_type_name}" section below.      

    You must respond with structured output in the ParameterModelCallOutput schema:
        {ParameterModelCallOutput.model_json_schema() }
    
    The output code for generating {output_type_name} instances must be placed in the "code" field. Refer to the "Recording code output" rules below for how to write this code.
    
    ------------------------------------------------------------------------------------------------------------
    General Parameter Rules:
    These rules describe how you should obtain values for parameters in the output {output_type_name} structures
    ------------------------------------------------------------------------------------------------------------
    - Use the code and other information in the message history to obtain the parameters
    - If the parameter rules specify that *you* should choose or set the value of a specific parameter yourself, do so. 
        - Otherwise, *never* guess a parameter. Never record such a parameter value unless it has been specified in your inputs.
    - If there is only one option for a parameter you must use that value.

    -------------------------------------------
    Recording code output rules:
    These rules apply to the code for generating the {output_type_name} structures you must record in the "code" field of your output
    ------------------------------------------    
    - The code field must contain correct Python inside a string.
    - Use Python True/False for boolean fields
    - You cannot use any libraries within the code. 
    - Your output code must instantiate *all* required instances.
    - The {output_type_name} structure instances must be stored to a list with name '{output_list_name}'.
    - You must use for loops to iterate over parameter values if there are more than 3 values.
    - Try to make the code as short as possible while still remaininng intelligible. For instance:
        - If you have a loop involving {output_type_name} that share multiple parameters, instantiate a base instance outside the loop with those static parameters set, and take copies within the loop to set the values that differ.
        - Even if not in a loop, if different instances have multiple shared parameters, prefer a base instance with copies versus writing out the full set of parameters again.
    - As your conversation with the user progresses, you must write code to create instances of {output_type_name} and fill their parameters in the "code" field. Never write code for any purpose other than instantiating and populating parameters of {output_type_name}.
    - The structure instances must follow the {output_type_name} schema for all fields and types, with the exception of unknown parameters. You must include all fields, even if they can have default values.
    - You must output the updated code in every response, even if some parameters are still unknown.
    - You must follow all rules (general and specific) provided in this prompt regarding the {output_type_name} parameters you record. 
    
{param_rules_header}    

{promptStringList(parameter_rules,4)}
   
    ------------------------------
    Schema for {output_type_name} 
    ------------------------------
    Follow this schema within your "code" output
    """ + json.dumps(structured_output_model.model_json_schema())
    
    user_interactions = input_messages.copy()
    accepted = False
    code = None

    print("PARAMETER MODEL CALL START")
    while(accepted == False):
        #Invoke the agent
        try:
            resp_struct = callModelWithStructuredOutput(llm_model, sys, user_interactions, ParameterModelCallOutput)
        except Exception as e:
            print("EXCEPTION",e)
            user_interactions.append(HumanMessage(f"Encountered an error: {e}"))
            continue

        if len(resp_struct.code) == 0:
            user_interactions.append(HumanMessage(f"Your response must include the code for generating {output_type_name} structures in 'code'"))
            print("NO CODE IN RESPONSE")
            continue

        print("OUTPUT", prettyPrintPydantic(resp_struct), "\n\n" )

        print("RAW CODE",resp_struct.code)
        aeval = asteval.Interpreter()
        aeval.eval(resp_struct.code)
        errors = ""
        if len(aeval.error)>0:
            for err in aeval.error:
                e = err.get_error()
                errors = errors + f"{e[0]}:{e[1]}\n"
        if len(errors) > 0:
            print("ERRORS", errors)
            user_interactions.append(HumanMessage(f"Running your code produced error(s): {errors}"))      
            continue
        if output_list_name not in aeval.symtable:
            print("SYMTABLE DOESNT CONTAIN RESULT", aeval.symtable.keys())
            user_interactions.append(HumanMessage(f"Your code does not create the '{output_list_name}' list"))
            continue 

        #Automatic validation
        vfail = False
        parsed_results = [] #not kept, just used for validation
        for r in aeval.symtable[output_list_name]:                                
            try:
                mval = structured_output_model.model_validate(r)
            except Exception as e:
                print("LIST INSTANCE VALIDATION ERROR",r,e)
                user_interactions.append(HumanMessage(f"There was an error when validating one of the model instances with Pydantic. Instance {r}, error {e}"))                  
                vfail = True
                break
            parsed_results.append(mval)

            if instance_validator is not None:            
                valid = instance_validator(mval)
                if not valid[0]:
                    print("AUTO INSTANCE VALIDATION FAIL",valid)
                    user_interactions.append(HumanMessage(f"There was an error when validating one of the model instances for correctness: {valid[1]}"))
                    vfail = True
                    break                

        if vfail:
            continue

        if group_validator is not None:
            valid = group_validator(parsed_results)
            if not valid[0]:
                print("AUTO GROUP VALIDATION FAIL",valid)
                user_interactions.append(HumanMessage(f"There was an error when validating the collection of model instances for correctness: {valid[1]}"))
                continue

        #Human validation                        
        AgentPrint(f"Obtained:\n" + prettyPrintPydantic(resp_struct.code))
        
        accepted = queryYesNo("Is this correct?")
        
        if(accepted == False):
            reason = AgentInput("Explain what is wrong: ")
            agent_state.done = False
            user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))
            continue
        else:
            code = resp_struct.code
            break
    return code
