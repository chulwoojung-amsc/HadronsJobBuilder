from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)
import json
from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from .common import *
from .hadrons_xml import HadronsXML

class RBPrecCGsolver(BaseModel):
    """red-black preconditioned conjugate gradient (CG) solver"""
    type: Literal["RBPrecCG"] = "RBPrecCG"
    residual: float = Field(...,description="the solver tolerance, residual or stopping condition. Typical values are in the range 1e-6 to 1e-9")
    maxIteration: NonNegativeInt = Field(10000,description="maximum number of solver iterations. DEFAULTABLE")
    guesser: str = Field("", description="guesser instance. DEFAULTABLE")
    
    def setXML(self,name,action,xml):
        opt = xml.addModule(name,"MSolver::RBPrecCG")
        HadronsXML.setValues(opt, [ ("action",action), ("maxIteration", self.maxIteration), ("residual", self.residual), ("guesser", "") ])

    
class SolverConfig(BaseModel):
    name : str = Field(..., description="The name/tag for the solver instance")
    solver_args: Union[RBPrecCGsolver] = Field(..., description="Parameters of the solver. Each item must have a 'type' field. Valid values are: RBPrecCG", discriminator='type')
    action: str = Field(..., description="The name/tag of the action instance to use with the solver.")
    user_info: str = Field(..., description="Additional information (if any) provided by the user on what observables/propagators this solver will be used for")
    
    def setXML(self,xml):
        self.solver_args.setXML(self.name,self.action,xml)
    
class SolversConfig(BaseModel):
    solvers: List[SolverConfig] = Field(...,description="The list of solver instances")

def identifySolvers(model, state, user_interactions: list[BaseMessage]) -> SolversConfig:
    """
    Parse the list of messages to identify a list of solver instances and their associated parameters
    """
    
    sys = """
You are an assistant responsible for identifying the solvers required for computing the lattice QCD propagators for the calculation.

Previous agent interactions have identified a set of observables and their required number of propagators. Solvers are required to compute those propagators. A solver instance has a set of parameters such as stopping conditions and the maximum number of iterations. The instance also has an 'action' field, that must be set to the name of one of the action instances identified previously. Each action instance must have one or more solver instances associated with it.

Workflow:
1) If the user has not already done so in their previous responses, ask the user to specify what solver *types* they wish to use for which propagators. This question should not be specific to one observable or propagator; rather you should allow the user the freedom to specify information that could apply to multiple or even all propagators. In your question, list the solvers that you support but do not list their associated parameters. Do not ask the user to provide parameters at this stage.

For example, "Specify the solvers required for the calculation (supported options: <OPTIONS>)."

If there is only one supported solver you may assume this response and skip this question; however you must explain this to the user using the provideInformationToUser
    
2) Instantiate an instance of SolverConfig for each required solver according to the rules described below.
3) Populate all parameters in conjunction with the user following the user query rules below
4) Insert the SolverConfig into the SolversConfig.solvers list
       
## Solver instance rules:
 - A different instance is required for each unique set of solver parameters. Some examples are as follows:
        a) If the user desires both a "sloppy" (loose tolerance) propagator and an "exact" (tight tolerance) propagator for a given action, create two solver instances with the same action but different residuals. This example is appropriate for an AMA style calculation.
        b) If the user specified the RBPrecCG solver type and there are action instances with names "action_1" and "action_2", create two separate solver instances with different values for the 'action' parameter.
   Note that these examples are just some of many possible workflows. Do not assume that the user desires either of these patterns. In particular, do not confuse the user by mentioning the concepts of sloppy or exact solvers unless the user has indicated that they want to do an AMA workflow.

 - Ensure there is at least one solver instance per action instance. 
 - Create a separate entry for each solver instance, even if the solver appears multiple times with different parameters.
 - Your list must include every solver instance explicitly mentioned, and only those. Do not invent instances. do not combine instances unless the user explicitly describes them as the same.
  
 - If a parameter value is unknown you must ask the user; never guess parameters unless they are specifically described with the word DEFAULTABLE, which indicates that the default value can be chosen.
 - For the "action" parameter, enter the name of the action associated with this solver instance. Each action should have one or more solvers.
 - For the "name" parameter, choose a unique tag/name to the solver instance. Assign this automatically, never ask the user (although they may choose to suggest names if they desire). Never use the same tag for different instances. The tag should include the action name and enough of the parameter values to uniquely distinguish it among the other source instances, prefering shorter tags if possible.    
 - For the "guesser" parameter, use the message history and user input to infer whether any of the existing eigensolver instances can be used to accelerate this solver by correlating this information with the user_info and other parameters of the eigensolvers, and if so use the eigensolver's name for the "guesser" parameter. Pay particular attention to the eigensolver's action_name parameter, which must be the same as the action associated with this propagator solver instance. Ask the user to confirm whether the inferred guesser is correct.
 - For the "user_info" parameter, summarize any information relevant to what observables/propagators this solver will be used for provided by the user. It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon. If the user does now specify any details, use an empty string. For example, if the user specifies that this solver will be used for light quark propagators, enter "use for all light quark propagators" in user_info.

Providing information to the user:
- Use the provideInformationToUser to output information that is not a question. Never include a question in the call to this tool. Questions must use the getUserInput tool instead.
    
User query rules:
- Use the getUserInput tool for queries
- Be brief and to the point with your questions. Prefer asking multiple consecutive questions rather than one question that requires specifying many choices.
- Do not output exhaustive lists of parameters associated with options.
- Do not require a specific format for the user's input. For example, never ask "provide each value separated by a comma" or similar.
- When asking for parameters, phrase your questions to refer to groups of propagators that share the same partial set of parameters rather than specific propagators.
- Do not phrase questions specifically referring to action instances; all questions should refer to groups of observables and/or propagators.
- Do not assume that the solvers associated with propagators in these groups will all have the same parameters
- Do not ask questions to the user that explicitly state that one value can be provided that applies to all instances in the group. If the user wants to specify a parameter that applies to more than one propagator in the group they will do so explicitly.       
- Use plurals for parameter names associated with groups containing more than one propagator
  Examples:
    "Provide the stopping conditions for the solvers associated with the pion's propagators" (plural)
    "Provide the maximum number of iterations for the solver associated with the pion's first propagator" (singular)
- When asking a question referring to a group, ensure your question clearly identifies the group.     
- If the user responds to a query with an invalid response, repeat the query until a valid response is provided. Never accept an invalid response.
- Instead of answering your question, the user might respond to your query with a question. If this occurs, do not try to fill in the parameter; instead, first answer the user's question using provideInformationToUser tool and ensure the user is satisfied with a follow-up call to getUserInput. Once satisfied, repeat the original question.
    
Your output must be in JSON format and adhere to the following schema:    
""" + json.dumps(SolversConfig.model_json_schema())
    
    agent = create_agent(model=model, tools=[getUserInput,provideInformationToUser], system_prompt=sys, response_format=SolversConfig)
    
    accepted = False
    obj = None
    while(accepted == False):    
        resp = agent.invoke({ "messages": user_interactions })
        obj = resp["structured_response"]        
        user_interactions = resp['messages']
        
        #Auto validation
        valid = True
        invalid_why = "Your previous response was invalid for the following reason(s):"
        names = []
        for r in obj.solvers:
            if not state.isValidAction(r.action):
                invalid_why += f"\n-Action instance '{r.action}' does not exist"
                valid = False
            if r.name in names:
                invalid_why += f"\n-Solver name '{r.name}' is not unique"
                valid = False
            names.append(r.name)
                                
        if not valid:
            user_interactions.append(HumanMessage(invalid_why))
            continue        

        #Human validation
        output = "Obtained {len(obj.solvers)} solvers\n" + prettyPrintPydantic(obj.solvers)
        Print(output)
        
        accepted = queryYesNo("Is this correct?")
        if(accepted == False):
            reason = Input("Explain what is wrong: ")
            user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))            
    return obj
