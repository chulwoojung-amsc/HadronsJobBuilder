from langchain_core.messages import BaseMessage
from langchain.messages import (
    SystemMessage,
    HumanMessage,
    ToolCall,
    AIMessage
)

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter, PositiveFloat, PositiveInt
from typing import Literal, Union, List, Optional, Tuple
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from langchain.agents import create_agent
from .hadrons_xml import HadronsXML
import json
from .common import *

class ChebyParams(BaseModel):
    """Parameters of the Chebyshev polynomial"""
    alpha : PositiveFloat = Field(..., description="The lower bound of the eigenvalue window suppresed by the Chebyshev filter.")
    beta : PositiveFloat = Field(..., description="The upper bound of the eigenvalue window suppresed by the Chebyshev filter.")
    Npoly: int = Field(..., ge=3, description="The Chebyshev polynomial order.")

    def validate(self, state):
        if self.beta <= self.alpha:
            return (False, "The upper window bound beta must be larger than the lower window bound alpha")
        elif self.Npoly % 2 == 0:
            return (False, "The polynomial order must be and odd integer")
        else:
            return (True,"")
        
    def setXML(self, chebyParams):
        HadronsXML.setValue(chebyParams, "alpha", self.alpha)
        HadronsXML.setValue(chebyParams, "beta", self.beta)
        HadronsXML.setValue(chebyParams, "Npoly", self.Npoly)
       
        
    
class LanczosEigenSolver(BaseModel):
    """Parameters of the Lanczos eigensolver"""
    type: Literal["LanczosEigenSolver"] = "LanczosEigenSolver"

    name : str = Field(..., description="The name of the eigensolver instance. ") 
    action_name : str = Field(..., description="The name of the associated action")
    
    cheby: ChebyParams = Field(..., description="Parameters of the Chebyshev filter")
    Nstop: PositiveInt = Field(..., description="Stop the solver once this many eigenvectors are within the tolerance")
    Nk: PositiveInt = Field(..., description="The number of eigenvectors to keep on each restart")
    Nextra: PositiveInt = Field(..., description="The number of extra/spare eigenvectors in the Krylov space which are discarded on each restart, i.e. Nm-Nk")
    resid: PositiveFloat = Field(..., description="The eigensolver convergence tolerance")
    MaxIt: PositiveInt = Field(20, description="The maximum number of restart iterations")

    storeEvecs: bool = Field(False, description="Indicate whether the eigenvectors are to be written to disk")
    fileStem: str = Field("", description="The file stem for the eigenvectors. The trajectory index will be appended")    

    def validate(self, state):
        cheby_valid, cheby_why = self.cheby.validate(state)
        if not cheby_valid:
            return (False, cheby_why)
        if not state.isValidAction(self.action_name):
            return (False, "Provided action name is not among the list of actions in the state")
        if self.storeEvecs and self.fileStem == "":
            return (False, "If writing the eigenvectors, a valid file stem must be provided")
        if self.Nstop > self.Nk:
            return (False, "Nk must be greater than or equal to Nstop")
        
        return (True,"")
        
    
    def setXML(self,name,xml):
        #We expose only the guesser to the agents, the other crud can stay internal
        
        lanc_name = name + "_solver"
        op_name = name + "_op"

        #Operator module
        op_opt = xml.addModule(op_name, "MFermion::Operators")
        HadronsXML.setValue(op_opt, "action", self.action_name)

        #Guesser module (create one even if we don't intend to use it as it has no real overhead)
        guesser_opt = xml.addModule(name, "MGuesser::ExactDeflation")
        HadronsXML.setValue(guesser_opt, "eigenPack", lanc_name)
        HadronsXML.setValue(guesser_opt, "size", self.Nstop)

        #Lanczos module
        lanc_opt = xml.addModule(lanc_name,"MSolver::FermionImplicitlyRestartedLanczos")

        #LancParams
        lp = HadronsXML.createSubElement(lanc_opt, "lanczosParams")        
        cheby = HadronsXML.createSubElement(lp, "Cheby")
        self.cheby.setXML(cheby)

        HadronsXML.setValue(lp, "Nstop", self.Nstop)
        HadronsXML.setValue(lp, "Nk", self.Nk)
        HadronsXML.setValue(lp, "Nm", self.Nk + self.Nextra)
        HadronsXML.setValue(lp, "resid", self.resid)
        HadronsXML.setValue(lp, "MaxIt", self.MaxIt)
        HadronsXML.setValue(lp, "betastp", 0.) #this doesn't do anything in Grid's solver anymore, but still must be specified :(
        HadronsXML.setValue(lp, "MinRes", 0) #forces a minimum number of restarts, not sure why this would be useful...

        #Lanczos options
        HadronsXML.setValue(lanc_opt, "op", op_name + "_schur") #Use with even/odd fields
        HadronsXML.setValue(lanc_opt, "output", self.fileStem)
        HadronsXML.setValue(lanc_opt, "redBlack", True)
        HadronsXML.setValue(lanc_opt, "multiFile", False) #I think this stores the evecs to part files per node rather than to one big file, maybe make an option later

class EigenSolverConfig(BaseModel):
    name : str = Field(..., description="The name for the eigensolver instance")  #technically this is the name of the guesser but that is irrelevant to the functioning of the agents
    solver_args: Union[LanczosEigenSolver] = Field(..., description="Parameters of the eigensolver. Each item must have a 'type' field. Valid values are: LanczosEigenSolver", discriminator='type')
    user_info: str = Field(..., description="Additional information (if any) provided by the user on what observables/propagators this eigensolver will be used for")
    
    def validate(self, state):
        return self.solver_args.validate(state)

    def setXML(self,xml):
        self.solver_args.setXML(self.name, xml)

class EigenSolversConfig(BaseModel):
    solvers: List[EigenSolverConfig] = Field(...,description="The list of eigensolver instances")

        
def setupEigenSolvers(model, state, user_interactions: list[BaseMessage]) -> EigenSolversConfig:
    """
    Setup (optional) eigensolvers
    """
    
    sys = """
    You are an assistant responsible for identifying all lattice QCD Dirac operator eigenvector solvers (aka, "eigensolvers") required for the calculation, based solely on user input. Eigenvectors can be used to optionally accelerate the calculation of quark propagators (particularly for light quarks) or as an independent target for the calculation.

    Output:
    You must populate the parameters of the EigenSolversConfig data structure
    
    Workflow:
    1. Based upon the message history determine whether the user has requested the use of eigenvectors either for any of the propagators or as an explicit calculation target
    2. If the user did not ask for eigenvectors, you must ask the user whether they wish to compute them to accelerate their calculation. An example output is as follows:
       "I have determined that you did not request the use of eigenvectors to accelerate the propagator calculation. This can provide benefits if either the eigenvectors already exist on disk or if you are computing a large number of propagators for the same action. Do you wish to use eigenvectors for the calculation?"
   
       Note that eigenvectors are commonly used when applying the all-mode averaging (AMA) method as this typically involves solving propagators on every timeslice.    
    3. If the user answers in the negative, leave the "solvers" field empty and perform no further actions.
    4. If eigenvectors are desired, ask the user to describe which propagators/observables to use the eigenvectors for. Ask this question only once and use the answer to infer your following actions. 
    5. Using this information and the message history, determine how many eigensolver instances are required. A separate instance is required if any of the parameters differ; for instance, if eigenvectors are required for two different action instances or if solvers for the same action are required with different residual tolerances, create two separate entries.   
    6. Work with the user to populate the parameters of each eigensolver instance in the order they appear in the data EigenSolversConfig schema.
       
    Parameter rules:
    - If a parameter value is unknown you must ask the user; never guess parameters.
    - If a parameter has a default, you may suggest that value to the user but you must not assume a value without asking.
    - The fileStem parameter is an exception to the previous rule. It is only necessary to specify a fileStem if the user wants the eigenvectors to be saved (storeEvecs == True)
    - You should infer the action name based upon the information the user provided regarding the use of the eigenvectors by correlating that information with the user_info fields of the action instances. However you must ask the user to confirm your result.
    - The polynomial order Npoly must be an odd integer. If the user specifies an even integer, explain the issue to the user and request an odd value.
    - For the 'user_info' field, summarize any information relevant to what observables/propagators this eigensolver will be used for provided by the user. It is important that any positional information about the propagator be included, for example whether it is the first or second propagator of a two-point function, or if it is a 'spectator' quark in a baryon. If the user does now specify any details, use an empty string. For example, if the user specifies that this source will be used for light quark propagators, enter "use for all light quark propagators" in user_info.
    - For the 'name' field, assign a unique tag/name to the instance. Never use the same tag for different instances. The tag should include the solver type and action name and enough of the parameter values to uniquely distinguish it among the other eigensolver instances, prefering shorter tags if possible.
    
    Eigensolver instance rules:    
    - Create a separate entry for each eigensolver instance, even if the same eigensolver type appears multiple times with different parameters.
    - Your list must include every eigensolver instance explicitly mentioned, and only those. Do not invent instances. Do not combine instances unless the user explicitly describes them as the same.
    - Only create separate entries for eigensolvers whose parameters differ, even if those eigensolver will be associated with different propagators. 

    User Query rules:
    - Use the getUserInput tool
    - Be brief and to the point with your questions. Prefer asking multiple consecutive questions rather than one question that requires specifying many choices.
    - Do not output exhaustive lists of parameters associated with options.
    - If the user responds to a query with an invalid response, repeat the query until a valid response is provided. Never accept an invalid response.
    - Instead of answering your question, the user might respond to your query with a question. If this occurs, answer the user's question using provideInformationToUser tool and ensure the user is satisfied with a follow-up call to getUserInput. Once satisfied, repeat the original question.
    - If the user asks for advice or help regarding which solver to use or for what parameters to use, refer to the information provided below regarding each solver.
    
Your output must be in JSON format and adhere to the following schema:    
""" + json.dumps(EigenSolversConfig.model_json_schema()) + """

     Information about the solvers:

     - LanczosEigenSolver
          This is the implicitly restarted Lanczos algorithm applied to the even-odd preconditioned fermion fields.
          For each iteration the solver estimates Nm = Nk + Nextra eigenvectors by running the Lanczos algorithm for a fixed number of steps then applying a transformation that compresses the interesting information into an Nk-dimension subspace, discarding the remaining Nextra spurious vectors. This process is repeated until Nstop (where Nstop <= Nk) eigenvectors have converged to a required tolerance.
          A Chebyshev polynomial filter is used to make the wanted eigenvectors large and well separated which suppressing the contribution of eigenvectors above the desired range. The parameter beta is set to a value above the largest eigenvalue of the (Hermitian) Dirac operator and alpha to a value just larger than the largest desired eigenvector. This suppresses eigenvectors outside of the desired range. The polynomial order is typically chosen to be sufficiently large that the in-range eigenvectors become large and well separated but not so large that the added computational cost of applying the Dirac operator more times outweights the benefit. Typical values are in the range 100-300.
"""
    
    agent = create_agent(model=model, tools=[getUserInput,provideInformationToUser], system_prompt=sys, response_format=EigenSolversConfig)

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
            if r.name in names:
                invalid_why += f"\n-Eigensolver name '{r.name}' is not unique"
                valid = False
            names.append(r.name)
                
            v, rs = r.validate(state)
            if not v:
                valid = False
                invalid_why += f"\n-Eigensolver '{r.name}' failed validation for reason: {rs}"
                                
        if not valid:
            user_interactions.append(HumanMessage(invalid_why))
            continue       
        
        #Human validation
        output = f"Obtained {len(obj.solvers)} eigensolvers\n" + prettyPrintPydantic(obj.solvers)
        Print(output)

        accepted = queryYesNo("Is this correct?")
        if(accepted == False):
            reason = Input("Explain what is wrong: ")
            user_interactions.append(HumanMessage(f"Your previous response was not accepted for the following reason: {reason}"))
            
    return obj
