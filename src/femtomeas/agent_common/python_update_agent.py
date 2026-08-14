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
from .common import queryYesNo, prettyPrintPydantic, prettyPrintPythonCode, getStructuredResponse, callModelWithStructuredOutput, Print as AgentPrint, Input as AgentInput, InputMulti as AgentInputMulti
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

from .python_output_agent import indentExceptFirst, promptStringList, listEnumerateStr, executeCode, executeCodeAndParse

class InstanceInfo(BaseModel):
    instance_tag: str = Field(..., description="The name/tag of the instance")
    user_info: str = Field(..., description="Additional information (if any) provided by the user on how this instance will be employed")

def checkUsedInstances(used_instance_list_name: str,
                        used_instance_list_code_input: str,
                        structured_output_model : BaseModel,
                        instantiation_list_name: str,
                        instantiation_list_code_input: str,
                        checker: Callable ):
    """Check the used-instance list and remove invalided entries either because the instance they point to no longer exists or it fails another, user-specified validation (e.g. if a dependency no longer exists)"""
    instance_list, e = executeCodeAndParse(instantiation_list_code_input, structured_output_model, instantiation_list_name)
    assert len(e) == 0
    instance_map = {}
    for i in instance_list:
        instance_map[i.name] = i

    orig_use_list, e = executeCodeAndParse(used_instance_list_code_input, InstanceInfo, used_instance_list_name)
    assert len(e) == 0

    regen_list = False
    validated = []
    for o in orig_use_list:
        if o.instance_tag not in instance_map.keys(): #if somehow the instance no longer exists
            regen_list=True            
        elif not checker(instance_map[o.instance_tag]):
            regen_list=True
        else:
            validated.append(o)

    if regen_list:
        #we don't need an LLM to rewrite this list
        out = f"{used_instance_list_name} = ["
        first = True
        for o in validated:
            if not first:
                out = out + ", "
            out = out + str(o.model_dump())
            first = False
        out += ']'                        
        return False, out 
    else:
        return True, ""






class AgentOutput(BaseModel):
    """Structured output for the agent"""    
    question_to_user: str = Field("", description="A question posed to the user")
    answer_to_user: str = Field("", description="An answer to a question posed by the user")
    scratchpad_note: str = Field("", description="Notes kept by the agent, appended to the current scratchpad")
    new_instance_code: str = Field(..., description="The current output code for instantiating the new instances")    
    use_instance_code: str = Field(..., description="Code for generating the final list of instances required")
    done: bool = Field(..., description="The agent's workflow is complete")


class AgentState:
    def reset(self):                
        self.scratch = []
        self.code = ""
        
    def __init__(self):
        self.reset()

agent_state = AgentState()                


def autoValidateNewInstanceCode(code, instance_validator, group_validator, structured_output_model, instantiation_list_name):
    print("RAW CODE",code)
    aeval = asteval.Interpreter()

    #Check to ensure the agent does not create a new list
    aeval.eval(code)
    if instantiation_list_name in aeval.symtable:
        print("AGENT CREATED LIST",code)
        return False, HumanMessage(f"Your code must NOT create the list named '{instantiation_list_name}'")        

    aeval.eval(f"""
{instantiation_list_name} = []                
{code}
""")
    errors = ""
    if len(aeval.error)>0:
        for err in aeval.error:
            e = err.get_error()
            errors = errors + f"{e[0]}:{e[1]}\n"
    if len(errors) > 0:
        print("NEW INSTANCE CODE ERRORS", errors)
        return False, HumanMessage(f"Running your new_instance_code code produced error(s): {errors}")

    #Automatic validation    
    parsed_results = [] #not kept, just used for validation
    for r in aeval.symtable[instantiation_list_name]:                                
        try:
            mval = structured_output_model.model_validate(r)
        except Exception as e:
            print("LIST INSTANCE VALIDATION ERROR",r,e)
            return False, HumanMessage(f"There was an error when validating one of the model instances with Pydantic. Instance {r}, error {e}")
            
        parsed_results.append(mval)

        if instance_validator is not None:            
            valid = instance_validator(mval)
            if not valid[0]:
                print("AUTO INSTANCE VALIDATION FAIL",valid)
                return False, HumanMessage(f"There was an error when validating one of the model instances for correctness: {valid[1]}")

    #Custom group validation                
    if group_validator is not None:
        valid = group_validator(parsed_results)
        if not valid[0]:
            print("AUTO GROUP VALIDATION FAIL",valid)
            return False, HumanMessage(f"There was an error when validating the collection of model instances for correctness: {valid[1]}")
            
    return True, None


def autoValidateUsedInstanceCode(use_instance_code, used_instance_list_name, do_append_to_used_instances, new_instance_code, instantiation_list_name, instantiation_list_code_base, structured_output_model):
    #Get the complete list of names of all instances
    aeval = asteval.Interpreter()
    aeval.eval(f"""
{instantiation_list_code_base}                
{new_instance_code}
""")
    all_names = []
    for r in aeval.symtable[instantiation_list_name]: 
        mval = structured_output_model.model_validate(r)
        all_names.append(mval.name)

    #If a previous used-instance list exists, check to ensure the agent does not create a new list
    if do_append_to_used_instances:
        aeval = asteval.Interpreter()
        aeval.eval(use_instance_code)
        if used_instance_list_name in aeval.symtable:
            print("AGENT CREATED USED INSTANCE LIST",use_instance_code)
            return False, HumanMessage(f"Your 'use_instance_code' code must NOT create the list named '{used_instance_list_name}'")            

    #Check the used-instance list is created and refers to existing instances
    aeval = asteval.Interpreter()
    aeval.eval(f"""
{ used_instance_list_name + " = []\n" if do_append_to_used_instances else "" }
{use_instance_code}""")

    errors = ""
    if len(aeval.error)>0:
        for err in aeval.error:
            e = err.get_error()
            errors = errors + f"{e[0]}:{e[1]}\n"
    if len(errors) > 0:
        print("USED INSTANCE CODE ERRORS", errors)
        return False, HumanMessage(f"Running your use_instance_code code produced error(s): {errors}")        

    if used_instance_list_name not in aeval.symtable:
        print("USED_INSTANCE_LIST_NAME NOT DEFINED")
        return False, HumanMessage(f"Your 'use_instance_code' code output field must generate a list named {used_instance_list_name}")        
    
    if instantiation_list_name in aeval.symtable:
        print("USED INSTANCE CODE REFERS TO LIST", use_instance_code)
        return False, HumanMessage(f"The 'use_instance_code' cannot refer directly to {instantiation_list_name} (e.g. via indices or conditionals)")        

    for r in aeval.symtable[used_instance_list_name]:
        try:
            mval = InstanceInfo.model_validate(r)
        except Exception as e:
            print("USED INSTANCE LIST INSTANCE VALIDATION ERROR",r,e)
            return False, HumanMessage(f"There was an error when validating one of the InstanceInfo instances with Pydantic. Instance {r}, error {e}")
    
        if mval.instance_tag not in all_names:
            print("USED INSTANCE ENTRY",r,"NOT A KNOWN NAME")
            return False, HumanMessage(f"Name {mval.instance_tag} in {used_instance_list_name} is not a known instance name")

    return True, None


def parameterAgentSingleQ(llm_model, structured_output_model : BaseModel,                    
                   instantiation_list_name: str,
                   instantiation_list_code_input: str | None,               
                   used_instance_list_name: str,
                   used_instance_list_code_input: str | None,
                   role: str, tools,
                   tool_rules : List[str] = [],
                   parameter_rules : List[str] = [],
                   user_info_rules : str | None = None, #extra rules for populating the "user_info" field of InstanceInfo
                   input_messages = [ HumanMessage("Start your workflow") ],
                   additional_user_query_rules = [],                   
                   instance_validator : Callable | None = None,
                   group_validator : Callable | None = None,
                   used_instance_list_checker : Callable = lambda m: True #a custom checked for instances referred to in the used-instance list to test whether they remain valid
                   ):

    agent_state.reset()    
    do_append_to_new_instances = instantiation_list_code_input is not None
    invalidate_remaining_workflow = False #whether changes here invalidate later steps in the workflow

    do_append_to_used_instances = False
    if used_instance_list_code_input is not None:
        do_append_to_used_instances = queryYesNo("The used module instance list already exists for this observable, do you wish to append to this list? Answering 'n' will overwrite the list and require later workflow stages to be repeated for this observable.", f"\nExisting code\n{used_instance_list_code_input}")
        if not do_append_to_new_instances: #if we overwrite the used module instance list we need to redo later workflow stages
            invalidate_remaining_workflow = True

    ################################################
    #if we are appending, first check to ensure all the existing used-instance entries remain valid, if not remove them
    if do_append_to_used_instances:
        v, c = checkUsedInstances(used_instance_list_name, used_instance_list_code_input, structured_output_model,  instantiation_list_name, used_instance_list_code_input, used_instance_list_checker)
        if not v:
            AgentPrint(f"""I have detected that some entries from the previous used-instance list have become invalid and have removed them. Ensure that you include replacements for these if needed.
Old used-instance code:
{used_instance_list_code_input}
New used-instance code:
{v}            
"""
            )
            used_instance_list_code_input = c
    ##################################################  

    instantiation_list_code_base = instantiation_list_code_input if instantiation_list_code_input is not None else f"{instantiation_list_name} = []"

    output_type_name = type(structured_output_model).__name__
    param_rules_header = """    -------------------------------------------
    Additional rules for specific parameters:   
    -------------------------------------------
    """ if len(parameter_rules) > 0 else "" 

    sys = f"""
    You are a conversational agent responsible for {role}
    
    On each turn of the conversation you must respond with structured output in the AgentOutput schema:
        {AgentOutput.model_json_schema() }

    The overall goal of your conversation is to write Python code snippets for instantiating any *new* instances of the {output_type_name} structure required. 
    - The schema for the instances is provided in the "Schema for {output_type_name}" section below.
    - Existing instances are generated according to the code snippet provided in the "Existing {output_type_name} instantiation code" section below. 
    - You must only instantiate new instances if no existing instance matches.
    - New instances must be appended to the list '{instantiation_list_name}'. Do not create this list; assume that your code snippet will be appended to the existing code.    
    - New instances must be generated according to the "General Parameter Rules" below.
    - Your current working draft of the code for generating new instances is provided in the "Draft new {output_type_name} instantiation code" section below. 
    - You must always record the current working draft of the code for generating new {output_type_name} instances in the "new_instance_code" field of your output. Refer to the "Recording code output" rules below for how to write this code.
    
    Each time you receive a message from the user that is not a question, review your current knowledge of the required parameters for the output {output_type_name} structures. 
        - If your code generates all required instances and populates them with correct values for all required parameters, perform the "Completion workflow" described below.
        - Otherwise set the "done" parameter to False. 

    If your workflow is not complete (i.e. your code does not instantiate all required instances or sets all required parameters) you must continue to ask questions to obtain the missing parameters and instances:    
        - Use the output field "answer_to_user" to answer a question that the user posed, if any. If the user asks a question, your response must contain an answer.
        - Use the output field "question_to_user" to ask a question.
      These outputs will be sent to the user and their response will be contained in the next message you receive. Follow the "Question/answer output rules" below when formating these outputs.
      
    Use the "scratchpad_note" field of your output to record notes to yourself (these are not visible to the user). Refer to the scratchpad rules below for appropriate content.
            
    ----------------------------------------------
    Existing {output_type_name} instantiation code
    ----------------------------------------------
    {instantiation_list_code_base}

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
    These rules apply to the code for generating the {output_type_name} structures you must record in the "new_instance_code" field of your output
    ------------------------------------------    
    - The code field must contain correct Python inside a string.
    - The code must produce jsonable Python dictionaries according to schema below.
    - Use Python True/False for boolean fields
    - Use Python None instead of "null"
    - You cannot use any libraries within the code. 
    - The {output_type_name} structure instances must be appended to a list with name '{instantiation_list_name}'. This list must contain ONLY {output_type_name} structures.
    - You must use for loops to iterate over parameter values if there are more than 3 values.
    - Try to make the code as short as possible while still remaininng intelligible. For instance:
        - If you have a loop involving {output_type_name} that share multiple parameters, instantiate a base instance outside the loop with those static parameters set, and take copies within the loop to set the values that differ.
        - Even if not in a loop, if different instances have multiple shared parameters, prefer a base instance with copies versus writing out the full set of parameters again.
    - As your conversation with the user progresses, you must write code to create instances of {output_type_name} and fill their parameters in the "new_instance_code" field. Never write code for any purpose other than instantiating and populating parameters of {output_type_name}.
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
    Your "new_instance_code" output must produce jsonable Python dictionaries according to the following schema:
{json.dumps(structured_output_model.model_json_schema())}

    ------------------------------
    Completion workflow
    ------------------------------
    When your conversation workflow is complete, perform the following actions:
    
    1) Write a Python code snippet in your "use_instance_code" output field that {"appends to an existing" if do_append_to_used_instances else "creates a"} list named {used_instance_list_name} of InstanceInfo objects. Use the following schema:
{json.dumps(InstanceInfo.model_json_schema())}     
        
       For the "instance_tag"s insert the "name" fields of the set of instances required.
         - These names must belong to instances within {instantiation_list_name}; do not create any new instances.
         - List only the names associated with the set of instances required by the user in the current conversation. Do not include other instances from {instantiation_list_name}.
         - Do not use if statements to isolate instances within {instantiation_list_name} as this list may grow in the future.

       For the "user_info" field, insert any information about how/where this instance will be used.     
{promptStringList([user_info_rules],7) if user_info_rules is not None else ""}

    2) Set the "done" parameter in your output to True indicating that your workflow is complete.
    """
    config = {"configurable": {"thread_id": "1", "stream" : False}}

    @dynamic_prompt
    def system_prompt(request: ModelRequest) -> str:
        prompt = sys + f"""
------------------------------------------------
Draft new {output_type_name} instantiation code
------------------------------------------------
{agent_state.code}

---------------------------
Current scratchpad contents
---------------------------        
{listEnumerateStr(agent_state.scratch)}   
        """
        return prompt
    
    all_tools = tools.copy()
    agent = create_agent(model=llm_model, tools=all_tools, middleware=[system_prompt], response_format=AgentOutput)

    user_interactions = input_messages.copy()
    accepted = False
    new_instance_code = None
    use_instance_code = None

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

        if resp_struct.done and len(resp_struct.question_to_user) > 0:
            user_interactions.append(HumanMessage("You cannot ask a question if 'done' is set to True"))
            print("DONE TRUE BUT QUESTION",resp_struct.question_to_user)
            continue
        if not resp_struct.done and len(resp_struct.question_to_user) == 0:
            user_interactions.append(HumanMessage("Your response must include a question unless you are done"))
            print("NO QUESTION IN RESPONSE")
            continue        

        print("OUTPUT", prettyPrintPydantic(resp_struct), "\n\n" )

        if resp_struct.done:
            print("DONE")

            #Automatic validation
            valid, fail_message = autoValidateNewInstanceCode(resp_struct.new_instance_code, instance_validator, group_validator, structured_output_model, instantiation_list_name)
            if not valid:
                user_interactions.append(fail_message)
                continue 

            valid, fail_message = autoValidateUsedInstanceCode(resp_struct.use_instance_code, used_instance_list_name, do_append_to_used_instances, resp_struct.new_instance_code, instantiation_list_name, instantiation_list_code_base, structured_output_model)
            if not valid:
                user_interactions.append(fail_message)
                continue             

            #Allow the agent to make a closing statement, e.g. in response to fixing a previous validation error
            if len(resp_struct.answer_to_user) > 0:                                            
                user_interactions.append(AIMessage(resp_struct.answer_to_user))
                AgentPrint(resp_struct.answer_to_user)

            #Human validation
            accepted = queryYesNo("Is the following code for generating *new* instances correct?", "\n" + prettyPrintPythonCode(resp_struct.new_instance_code) + "\n")            

            if(accepted == False):
                reason = AgentInput("Explain what is wrong: ")
                user_interactions.append(HumanMessage(f"Your previous new_instance_code output was not accepted for the following reason: {reason}"))
                continue

            accepted = queryYesNo("Is the following code for listing the names of the used instances correct?", "\n" + prettyPrintPythonCode(resp_struct.use_instance_code) + "\n")      

            if(accepted == False):
                reason = AgentInput("Explain what is wrong: ")
                user_interactions.append(HumanMessage(f"Your previous use_instance_code output was not accepted for the following reason: {reason}"))
                continue
            
            new_instance_code = resp_struct.new_instance_code
            use_instance_code = resp_struct.use_instance_code
            break
        
        else:
            print("RESP")

            #Append any notes to the scratchpad
            if len(resp_struct.scratchpad_note) > 0:
                agent_state.scratch.append(resp_struct.scratchpad_note)

            #Set the working params struct for the next round of prompting
            agent_state.code = resp_struct.new_instance_code

            #Compose the AI message to the user
            ai_msg = resp_struct.answer_to_user + ("\n\n" if len(resp_struct.answer_to_user) > 0 else "") + resp_struct.question_to_user

            #Add it to the message history
            user_interactions.append(AIMessage(ai_msg))

            #Obtain the user response
            user_resp = AgentInput(ai_msg)
            user_interactions.append(HumanMessage(user_resp))
    return instantiation_list_code_base + "\n" + new_instance_code, \
           used_instance_list_code_input + "\n" + use_instance_code if do_append_to_used_instances else use_instance_code,  \
           invalidate_remaining_workflow



class AgentOutputMultiQ(BaseModel):
    """Structured output for the agent"""    
    questions_to_user: List[str] = Field(..., description="Questions posed to the user")
    answer_to_user: str = Field(..., description="An answer to a question posed by the user")
    scratchpad_note: str = Field(..., description="Notes kept by the agent, appended to the current scratchpad")
    new_instance_code: str = Field(..., description="The current output code for instantiating the new instances")    
    use_instance_code: str = Field(..., description="Code for generating the final list of instances required")
    done: bool = Field(..., description="The agent's workflow is complete")



def parameterAgentMultiQ(llm_model, structured_output_model : BaseModel,                    
                   instantiation_list_name: str,
                   instantiation_list_code_input: str | None,               
                   used_instance_list_name: str,
                   used_instance_list_code_input: str | None,
                   role: str, tools,
                   tool_rules : List[str] = [],
                   parameter_rules : List[str] = [],
                   user_info_rules : str | None = None, #extra rules for populating the "user_info" field of InstanceInfo
                   input_messages = [ HumanMessage("Start your workflow") ],
                   additional_user_query_rules = [],                   
                   instance_validator : Callable | None = None,
                   group_validator : Callable | None = None,
                   used_instance_list_checker : Callable = lambda m: True #a custom checked for instances referred to in the used-instance list to test whether they remain valid
                   ):

    agent_state.reset()    
    do_append_to_new_instances = instantiation_list_code_input is not None
    invalidate_remaining_workflow = False #whether changes here invalidate later steps in the workflow

    do_append_to_used_instances = False
    if used_instance_list_code_input is not None:
        do_append_to_used_instances = queryYesNo("The used module instance list already exists for this observable, do you wish to append to this list? Answering 'n' will overwrite the list and require later workflow stages to be repeated for this observable.", f"\nExisting code\n{used_instance_list_code_input}")
        if not do_append_to_new_instances: #if we overwrite the used module instance list we need to redo later workflow stages
            invalidate_remaining_workflow = True

    ################################################
    #if we are appending, first check to ensure all the existing used-instance entries remain valid, if not remove them
    if do_append_to_used_instances:
        v, c = checkUsedInstances(used_instance_list_name, used_instance_list_code_input, structured_output_model,  instantiation_list_name, used_instance_list_code_input, used_instance_list_checker)
        if not v:
            AgentPrint(f"""I have detected that some entries from the previous used-instance list have become invalid and have removed them. Ensure that you include replacements for these if needed.
Old used-instance code:
{used_instance_list_code_input}
New used-instance code:
{v}            
"""
            )
            used_instance_list_code_input = c
    ##################################################  

    instantiation_list_code_base = instantiation_list_code_input if instantiation_list_code_input is not None else f"{instantiation_list_name} = []"

    output_type_name = type(structured_output_model).__name__
    param_rules_header = """    -------------------------------------------
    Additional rules for specific parameters:   
    -------------------------------------------
    """ if len(parameter_rules) > 0 else "" 

    sys = f"""
    You are a conversational agent responsible for {role}
    
    On each turn of the conversation you must respond with structured output in the AgentOutputMultiQ schema:
        {AgentOutputMultiQ.model_json_schema() }

    The overall goal of your conversation is to write Python code snippets for instantiating any *new* instances of the {output_type_name} structure required. 
    - The schema for the instances is provided in the "Schema for {output_type_name}" section below.
    - Existing instances are generated according to the code snippet provided in the "Existing {output_type_name} instantiation code" section below. 
    - You must only instantiate new instances if no existing instance matches.
    - New instances must be appended to the list '{instantiation_list_name}'. Do not create this list; assume that your code snippet will be appended to the existing code.    
    - New instances must be generated according to the "General Parameter Rules" below.
    - Your current working draft of the code for generating new instances is provided in the "Draft new {output_type_name} instantiation code" section below. 
    - You must always record the current working draft of the code for generating new {output_type_name} instances in the "new_instance_code" field of your output. Refer to the "Recording code output" rules below for how to write this code.
    
    Each time you receive a message from the user that is not a question, review your current knowledge of the required parameters for the output {output_type_name} structures. 
        - If your code generates all required instances and populates them with correct values for all required parameters, perform the "Completion workflow" described below.
        - Otherwise set the "done" parameter to False. 

    If your workflow is not complete (i.e. your code does not instantiate all required instances or sets all required parameters) you must continue to ask questions to obtain the missing parameters and instances:    
        - Use the output field "answer_to_user" to answer a question that the user posed, if any. If the user asks a question, your response must contain an answer.
        - Use the output field "questions_to_user" to ask questions.
      These outputs will be sent to the user and their response(s) will be contained in the next message you receive. Follow the "Question/answer output rules" below when formating these outputs.
      
    Use the "scratchpad_note" field of your output to record notes to yourself (these are not visible to the user). Refer to the scratchpad rules below for appropriate content.
            
    ----------------------------------------------
    Existing {output_type_name} instantiation code
    ----------------------------------------------
    {instantiation_list_code_base}

    -----------------------------------------
    Question and answer output rules
    -----------------------------------------
    - If your workflow is done and you have set the "done" parameter in your output, never output an answer or a question.
    - If your workflow is not done, your output to the user *must* include either:
      1) One or more questions
      2) ONE answer to the user's previous question AND one or more further questions   
    - NEVER output text not intended for the user such as notes-to-self.
    - The user's message may contain multiple responses. These will be separated by a dashed line break.         
    - If a field requires choosing a Union subtype with multiple options, you must first ask the user to choose the type before asking about any parameters of that subtype.
    - If the user has not responsed to your questions, do not think ahead to the next group of questions. Wait for the user to respond.    
    - Never ask if the user wants to specify a parameter; assume that the user wants to specify all parameters
    - Be brief and to the point with your questions.
    - Each question should only be about a single parameter. Ask a separate question for each parameter.
    - Prefer to ask multiple questions at once rather than one at a time, apart from when asking the user to choose between multiple types.
    - When asking the user to choose between multiple types, list only the types and not their parameters.
    - If you ask a question where the user is asked to choose between a set of known options, first obtain the list of options (calling any appropriate tools) then list those options alongside the question in your response. If there are more than 6 choices, list only the first 6 and indicate that there are more options.
    - If the user responds to a query with an invalid response, your response should explain that the choice is invalid and ask the question again. Never ask a question about the next field without a valid response to the current field.
    - If the user asks you a question about a parameter:
         - Answer the user's question in the "answer_to_user" field of your output
         - In the "questions_to_user" field you must repeat the original question about the parameter but include a statement indicating that they can ask follow-up questions instead of answering.
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
    - Never choose the default value for a parameter without asking the user. Instead, suggest the default value to the user and let them decide whether to use it.
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
    These rules apply to the code for generating the {output_type_name} structures you must record in the "new_instance_code" field of your output
    ------------------------------------------    
    - The code field must contain correct Python inside a string.
    - The code must produce jsonable Python dictionaries according to schema below.
    - Use Python True/False for boolean fields
    - Use Python None instead of "null"
    - You cannot use any libraries within the code. 
    - The {output_type_name} structure instances must be appended to a list with name '{instantiation_list_name}'. This list must contain ONLY {output_type_name} structures.
    - You must use for loops to iterate over parameter values if there are more than 3 values.
    - Try to make the code as short as possible while still remaining intelligible. For instance:
        - If you have a loop involving {output_type_name} that share multiple parameters, instantiate a base instance outside the loop with those static parameters set, and take copies within the loop to set the values that differ.
        - Even if not in a loop, if different instances have multiple shared parameters, prefer a base instance with copies versus writing out the full set of parameters again.
    - As your conversation with the user progresses, you must write code to create instances of {output_type_name} and fill their parameters in the "new_instance_code" field. Never write code for any purpose other than instantiating and populating parameters of {output_type_name}.
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
    Your "new_instance_code" output must produce jsonable Python dictionaries according to the following schema:
{json.dumps(structured_output_model.model_json_schema())}

    ------------------------------
    Completion workflow
    ------------------------------
    When your conversation workflow is complete, perform the following actions:
    
    1) Write a Python code snippet in your "use_instance_code" output field that {"appends to an existing" if do_append_to_used_instances else "creates a"} list named {used_instance_list_name} of InstanceInfo objects. Use the following schema:
{json.dumps(InstanceInfo.model_json_schema())}     
        
       For the "instance_tag"s insert the "name" fields of the set of instances required.
         - These names must belong to instances within {instantiation_list_name}; do not create any new instances.
         - List only the names associated with the set of instances required by the user in the current conversation. Do not include other instances from {instantiation_list_name}.
         - Do not use if statements to isolate instances within {instantiation_list_name} as this list may grow in the future.

       For the "user_info" field, insert any information about how/where this instance will be used.     
{promptStringList([user_info_rules],7) if user_info_rules is not None else ""}

    2) Set the "done" parameter in your output to True indicating that your workflow is complete.
    """
    config = {"configurable": {"thread_id": "1", "stream" : False}}

    @dynamic_prompt
    def system_prompt(request: ModelRequest) -> str:
        prompt = sys + f"""
------------------------------------------------
Draft new {output_type_name} instantiation code
------------------------------------------------
{agent_state.code}

---------------------------
Current scratchpad contents
---------------------------        
{listEnumerateStr(agent_state.scratch)}   
        """
        return prompt
    
    all_tools = tools.copy()
    agent = create_agent(model=llm_model, tools=all_tools, middleware=[system_prompt], response_format=AgentOutputMultiQ)

    user_interactions = input_messages.copy()
    accepted = False
    new_instance_code = None
    use_instance_code = None

    print("AGENT START")
    while(accepted == False):
        #Invoke the agent
        try:
            resp = agent.invoke({ "messages": user_interactions }, config=config)
            print("RESP TEST ", resp)
            resp_struct = getStructuredResponse(resp, AgentOutputMultiQ)
        except Exception as e:
            print("EXCEPTION",e)
            user_interactions.append(HumanMessage(f"Encountered an error: {e}"))
            continue

        if resp_struct.done and len(resp_struct.questions_to_user) > 0:
            user_interactions.append(HumanMessage("You cannot ask a question if 'done' is set to True"))
            print("DONE TRUE BUT QUESTION",resp_struct.questions_to_user)
            continue
        if not resp_struct.done and len(resp_struct.questions_to_user) == 0:
            user_interactions.append(HumanMessage("Your response must include a question unless you are done"))
            print("NO QUESTION IN RESPONSE")
            continue        

        print("OUTPUT", prettyPrintPydantic(resp_struct), "\n\n" )

        if resp_struct.done:
            print("DONE")

            #Automatic validation
            valid, fail_message = autoValidateNewInstanceCode(resp_struct.new_instance_code, instance_validator, group_validator, structured_output_model, instantiation_list_name)
            if not valid:
                user_interactions.append(fail_message)
                continue 

            valid, fail_message = autoValidateUsedInstanceCode(resp_struct.use_instance_code, used_instance_list_name, do_append_to_used_instances, resp_struct.new_instance_code, instantiation_list_name, instantiation_list_code_base, structured_output_model)
            if not valid:
                user_interactions.append(fail_message)
                continue             

            #Allow the agent to make a closing statement, e.g. in response to fixing a previous validation error
            if len(resp_struct.answer_to_user) > 0:                                            
                user_interactions.append(AIMessage(resp_struct.answer_to_user))
                AgentPrint(resp_struct.answer_to_user)

            #Human validation
            accepted = queryYesNo("Is the following code for generating *new* instances correct?", "\n" + prettyPrintPythonCode(resp_struct.new_instance_code) + "\n")            

            if(accepted == False):
                reason = AgentInput("Explain what is wrong: ")
                user_interactions.append(HumanMessage(f"Your previous new_instance_code output was not accepted for the following reason: {reason}"))
                continue

            accepted = queryYesNo("Is the following code for listing the names of the used instances correct?", "\n" + prettyPrintPythonCode(resp_struct.use_instance_code) + "\n")      

            if(accepted == False):
                reason = AgentInput("Explain what is wrong: ")
                user_interactions.append(HumanMessage(f"Your previous use_instance_code output was not accepted for the following reason: {reason}"))
                continue
            
            new_instance_code = resp_struct.new_instance_code
            use_instance_code = resp_struct.use_instance_code
            break
        
        else:
            print("RESP")

            #Append any notes to the scratchpad
            if len(resp_struct.scratchpad_note) > 0:
                agent_state.scratch.append(resp_struct.scratchpad_note)

            #Set the working params struct for the next round of prompting
            agent_state.code = resp_struct.new_instance_code

            #Compose the AI message to the user
            ai_msg = resp_struct.answer_to_user + ("\n\n" if len(resp_struct.answer_to_user) > 0 else "")
            for i, q in enumerate(resp_struct.questions_to_user):
                ai_msg += f"{i+1}) {q}\n"            

            #Add it to the message history
            user_interactions.append(AIMessage(ai_msg))

            #Obtain the user response
            user_resp = AgentInputMulti(resp_struct.questions_to_user, preamble=resp_struct.answer_to_user if len(resp_struct.answer_to_user) else None)
            user_msg = ""
            for r in user_resp:
                user_msg += "---------------------------------------------\n" + r + "\n"
            user_interactions.append(HumanMessage(user_msg))

    return instantiation_list_code_base + "\n" + new_instance_code, \
           used_instance_list_code_input + "\n" + use_instance_code if do_append_to_used_instances else use_instance_code,  \
           invalidate_remaining_workflow




def parameterAgent(llm_model, structured_output_model : BaseModel,                    
                   instantiation_list_name: str,
                   instantiation_list_code_input: str | None,               
                   used_instance_list_name: str,
                   used_instance_list_code_input: str | None,
                   role: str, tools,
                   tool_rules : List[str] = [],
                   parameter_rules : List[str] = [],
                   user_info_rules : str | None = None, #extra rules for populating the "user_info" field of InstanceInfo
                   input_messages = [ HumanMessage("Start your workflow") ],
                   additional_user_query_rules = [],                   
                   instance_validator : Callable | None = None,
                   group_validator : Callable | None = None,
                   used_instance_list_checker : Callable = lambda m: True, #a custom checked for instances referred to in the used-instance list to test whether they remain valid
                   multi_question_mode: bool = True #the agent can ask multiple questions at the same time (user still answers consecutively)
                   ):
    if multi_question_mode:
        return parameterAgentMultiQ(llm_model, structured_output_model, instantiation_list_name,
                                    instantiation_list_code_input, used_instance_list_name, used_instance_list_code_input,
                                    role, tools, tool_rules, parameter_rules, user_info_rules, input_messages, additional_user_query_rules,
                                    instance_validator, group_validator, used_instance_list_checker)
    else:
        return parameterAgentSingleQ(llm_model, structured_output_model, instantiation_list_name,
                                    instantiation_list_code_input, used_instance_list_name, used_instance_list_code_input,
                                    role, tools, tool_rules, parameter_rules, user_info_rules, input_messages, additional_user_query_rules,
                                    instance_validator, group_validator, used_instance_list_checker)




class ParameterModelCallOutput(BaseModel):        
    new_instance_code: str = Field(..., description="The current output code for instantiating the new instances")    
    use_instance_code: str = Field(..., description="Code for generating the final list of instances required")
    

def parameterModelCall(llm_model, structured_output_model : BaseModel,                    
                   instantiation_list_name: str,
                   instantiation_list_code_input: str | None,               
                   used_instance_list_name: str,
                   used_instance_list_code_input: str | None,
                   role: str,                  
                   parameter_rules : List[str] = [],
                   user_info_rules : str | None = None, #extra rules for populating the "user_info" field of InstanceInfo
                   input_messages = [ HumanMessage("Start your workflow") ],                   
                   instance_validator : Callable | None = None,
                   group_validator : Callable | None = None,
                   used_instance_list_checker : Callable = lambda m: True #a custom checked for instances referred to in the used-instance list to test whether they remain valid
                   ):

    do_append_to_new_instances = instantiation_list_code_input is not None
    invalidate_remaining_workflow = False #whether changes here invalidate later steps in the workflow

    do_append_to_used_instances = False
    if used_instance_list_code_input is not None:
        do_append_to_used_instances = queryYesNo("The used module instance list already exists for this observable, do you wish to append to this list? Answering 'n' will overwrite the list and require later workflow stages to be repeated for this observable.", f"\nExisting code\n{used_instance_list_code_input}")
        if not do_append_to_new_instances: #if we overwrite the used module instance list we need to redo later workflow stages
            invalidate_remaining_workflow = True

    ################################################
    #if we are appending, first check to ensure all the existing used-instance entries remain valid, if not remove them
    if do_append_to_used_instances:
        v, c = checkUsedInstances(used_instance_list_name, used_instance_list_code_input, structured_output_model,  instantiation_list_name, used_instance_list_code_input, used_instance_list_checker)
        if not v:
            AgentPrint(f"""I have detected that some entries from the previous used-instance list have become invalid and have removed them. Ensure that you include replacements for these if needed.
Old used-instance code:
{used_instance_list_code_input}
New used-instance code:
{v}            
"""
            )
            used_instance_list_code_input = c
    ##################################################  

    instantiation_list_code_base = instantiation_list_code_input if instantiation_list_code_input is not None else f"{instantiation_list_name} = []"

    output_type_name = type(structured_output_model).__name__
    param_rules_header = """    -------------------------------------------
    Additional rules for specific parameters:   
    -------------------------------------------
    """ if len(parameter_rules) > 0 else "" 

    sys = f"""
    You are are responsible for {role}
    
    You must respond with structured output in the ParameterModelCallOutput schema:
        {ParameterModelCallOutput.model_json_schema() }

    Your goals are as follows:
     
    1) Write a Python code snippet in your "new_instance_code" output field for instantiating any *new* instances of the {output_type_name} structure required. 
        - The schema for the instances is provided in the "Schema for {output_type_name}" section below.
        - Existing instances are generated according to the code snippet provided in the "Existing {output_type_name} instantiation code" section below. 
        - You must only instantiate new instances if no existing instance matches.
        - New instances must be appended to the list '{instantiation_list_name}'. Do not create this list; assume that your code snippet will be appended to the existing code.    
        - New instances must be generated according to the "General Parameter Rules" below.        
        - Refer to the "Recording code output" rules below for how to write this code.
    
    2) Write a Python code snippet in your "use_instance_code" output field that {"appends to an existing" if do_append_to_used_instances else "creates a"} list named {used_instance_list_name} of InstanceInfo objects. Use the following schema:
{json.dumps(InstanceInfo.model_json_schema())}     
        
       For the "instance_tag"s insert the "name" fields of the set of instances required.
         - These names must belong to instances within {instantiation_list_name}; do not create any new instances.
         - List only the names associated with the set of instances required by the user in the current context. Do not include other instances from {instantiation_list_name}.
         - Do not use if statements to isolate instances within {instantiation_list_name} as this list may grow in the future.

       For the "user_info" field, insert any information about how/where this instance will be used.     
{promptStringList([user_info_rules],7) if user_info_rules is not None else ""}
            
    ----------------------------------------------
    Existing {output_type_name} instantiation code
    ----------------------------------------------
    {instantiation_list_code_base}

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
    - Never repeat back a user's choice and ask them to confirm it.     
    - If the user has made a decision about a parameter or type, never ask them about it again, never ask them to confirm it.

    -------------------------------------------
    Recording code output rules:
    These rules apply to the code for generating the {output_type_name} structures you must record in the "new_instance_code" field of your output
    ------------------------------------------    
    - The code field must contain correct Python inside a string.
    - The code must produce jsonable Python dictionaries according to schema below.
    - Use Python True/False for boolean fields
    - You cannot use any libraries within the code. 
    - The {output_type_name} structure instances must be appended to a list with name '{instantiation_list_name}'. This list must contain ONLY {output_type_name} structures.
    - You must use for loops to iterate over parameter values if there are more than 3 values.
    - Try to make the code as short as possible while still remaininng intelligible. For instance:
        - If you have a loop involving {output_type_name} that share multiple parameters, instantiate a base instance outside the loop with those static parameters set, and take copies within the loop to set the values that differ.
        - Even if not in a loop, if different instances have multiple shared parameters, prefer a base instance with copies versus writing out the full set of parameters again.
    - Never write code to "new_instance_code" for any purpose other than instantiating and populating parameters of {output_type_name}.
    - The structure instances must follow the {output_type_name} schema for all fields and types, with the exception of unknown parameters. You must include all fields, even if they can have default values.
    - You must follow all rules (general and specific) provided in this prompt regarding the {output_type_name} parameters you record. 
    - The code you write must only pertain to creating {output_type_name} structures.
    - Never ask the user to confirm your code.

{param_rules_header}    

{promptStringList(parameter_rules,4)}

    ------------------------------
    Schema for {output_type_name} 
    ------------------------------
    Your "new_instance_code" output must produce jsonable Python dictionaries according to the following schema:
{json.dumps(structured_output_model.model_json_schema())}
    """

    user_interactions = input_messages.copy()
    accepted = False
    new_instance_code = None
    use_instance_code = None

    print("AGENT START")
    while(accepted == False):
        #Invoke the agent
        try:
            resp_struct = callModelWithStructuredOutput(llm_model, sys, user_interactions, ParameterModelCallOutput)            
        except Exception as e:
            print("EXCEPTION",e)
            user_interactions.append(HumanMessage(f"Encountered an error: {e}"))
            continue

        print("OUTPUT", prettyPrintPydantic(resp_struct), "\n\n" )

        #Automatic validation
        valid, fail_message = autoValidateNewInstanceCode(resp_struct.new_instance_code, instance_validator, group_validator, structured_output_model, instantiation_list_name)
        if not valid:
            user_interactions.append(fail_message)
            continue 

        valid, fail_message = autoValidateUsedInstanceCode(resp_struct.use_instance_code, used_instance_list_name, do_append_to_used_instances, resp_struct.new_instance_code, instantiation_list_name, instantiation_list_code_base, structured_output_model)
        if not valid:
            user_interactions.append(fail_message)
            continue             

        #Human validation
        accepted = queryYesNo("Is the following code for generating *new* instances correct?", "\n" + prettyPrintPythonCode(resp_struct.new_instance_code) + "\n")            

        if(accepted == False):
            reason = AgentInput("Explain what is wrong: ")
            user_interactions.append(HumanMessage(f"Your previous new_instance_code output was not accepted for the following reason: {reason}"))
            continue

        accepted = queryYesNo("Is the following code for listing the names of the used instances correct?", "\n" + prettyPrintPythonCode(resp_struct.use_instance_code) + "\n")      

        if(accepted == False):
            reason = AgentInput("Explain what is wrong: ")
            user_interactions.append(HumanMessage(f"Your previous use_instance_code output was not accepted for the following reason: {reason}"))
            continue
        
        new_instance_code = resp_struct.new_instance_code
        use_instance_code = resp_struct.use_instance_code
        break

    return instantiation_list_code_base + "\n" + new_instance_code, \
           used_instance_list_code_input + "\n" + use_instance_code if do_append_to_used_instances else use_instance_code,  \
           invalidate_remaining_workflow


