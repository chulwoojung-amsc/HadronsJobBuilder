from femtomeas.meas_config_agent.agent_workflows import *
from femtomeas.meas_config_agent.agent import *
from femtomeas.meas_config_agent.action_config_models import DWFaction, ActionConfig
from femtomeas.agent_common.python_update_agent import InstanceInfo
from femtomeas.meas_config_agent.source_config_models import SourceConfig, PointSource, WallSource
from femtomeas.meas_config_agent.solver_config_models import RBPrecCGsolver, SolverConfig
from femtomeas.meas_config_agent.propagator_config_models import PropagatorConfig
from femtomeas.meas_config_agent.smeared_prop_config_models import SmearedPropagatorConfig, WallSmear
from femtomeas.meas_config_agent.eigenvectors_models import EigenSolverConfig, LanczosEigenSolver, ChebyParams

from femtomeas.meas_config_agent.action_config import createActionGroup, ActionGroup
from femtomeas.meas_config_agent.solver_config import createSolverGroup, SolverGroup
from femtomeas.meas_config_agent.source_config import createSourceGroup, SourceGroup
from femtomeas.meas_config_agent.propagator_config import createPropagatorGroup, PropagatorGroup
from femtomeas.meas_config_agent.observable_config import createMeson2ptGroup
from femtomeas.meas_config_agent.smeared_prop_config import createSmearedPropagatorGroup, SmearedPropagatorGroup
from femtomeas.meas_config_agent.eigenvectors import createEigenSolverGroup, EigenSolverGroup

from femtomeas.meas_config_agent.state import State, reloadStateCheckpoint, checkpointState
import os
from femtomeas.meas_config_agent.gauge import GaugeFieldConfig, UnitGauge
import io
from langchain_openai import ChatOpenAI
import sys

def subWorkflowAgent(llm_model):
    role = f"""creating a code snippet that performs the instructions provided by the user during your conversation.

When you begin your workflow, ask the user for instructions.

Your code snippet must employ the functions in the "Registry" below to construct these module instances for this particular observable. These functions act upon a hidden internal state and query the user internally. Your focus should only be on calling these tools in the appropriate order. Follow the "Code rules" below. 

Do not ask the user to confirm or accept your code.
Never ask the user to provide the code.

If the user asks you questions you must answer them.

-------------
Registry
-------------
{function_manifest()}

-------------
Code rules
-------------
- Do not import any modules or functions; assume that the functions in the registry have already been imported.
- Functions in the registry act on a hidden internal state. The inputs and outputs are merely handles for chaining the logic. Handles must all be consumed within the code snippet; do not store them in any output structures (lists, dictionaries, etc)
- Your snippet should only perform the user's instructions and nothing else. Do not add code to write outputs.
- The output handles from your code snippet should be stored in an array named 'results'. Only store the handles for the outputs (i.e. the last calls in any given chain), not the intermediaries.

-----------
Tool usage
-----------
- The workflow you define may be one of many. You have been provided tools (listGroupHandles, listWorkflows, retrieveWorkflowCode) that you can use to obtain information about previously-created groups and workflows.
- If a tool for obtaining a list returns an empty list, interpret this as meaning no elements currently exist. NEVER repeatedly call the same tool over and over if it returns an empty list.
"""

    user_query_rules = [
    "You can only ask the user questions about the sequence of registry function calls", 
    "NEVER ask the user to provide details on groups or their parameters.",
    "If there are optional steps, you MUST ask the user if they want to perform those steps; NEVER make assumptions.",
    "You are allowed to choose group names if they have not been specified by the user, unless otherwise directed. Never tell the user that you cannot choose a group name for them."
    ]

    def printCode(obj):
        return prettyPrintPydantic(obj.code)

    graphs = None
    
    def validator(obj):
        symtable_in = asteval.make_symbol_table(use_numpy=False, **registry)
        aeval = asteval.Interpreter(symtable=symtable_in)
        aeval(obj.code)

        errors = ""
        if len(aeval.error)>0:
            for err in aeval.error:
                e = err.get_error()
                errors = errors + f"{e[0]}:{e[1]}\n"
        if len(errors) > 0:
            print("USED INSTANCE CODE ERRORS", errors)
            return False, HumanMessage(f"Running your use_instance_code code produced error(s): {errors}")    

        if "results" not in aeval.symtable:
            print("RESULTS NOT IN CODE")
            return False, HumanMessage("Your code must produce an array of handles named 'results'")
        if not isinstance(aeval.symtable['results'], list):
            print("RESULTS NOT LIST")
            return False, HumanMessage("Your 'results' output must be an array")
        for h in aeval.symtable['results']:
            if not isinstance(h, BaseGroupHandle):
                print("RESULTS ELEMENT NOT GROUPHANDLE")
                return False, HumanMessage("Your 'results' output array must contain only handles")

        nonlocal graphs
        graphs = [ h.parent_node for h in aeval.symtable['results'] ]
        return True, ""

    obj = parameterAgent(llm_model, AgentOutput, role, tools=[listGroupHandles,listWorkflows, retrieveWorkflowCode], additional_user_query_rules=user_query_rules, human_validation_output_formatter=printCode, validator=validator)

    assert graphs is not None
    cache={}
    #Evaluate all output handles with caching in case they are branches from the same chain
    for g in graphs:
        g.evalWithCache(cache)

    workflow_name = AgentInput("Provide a name for this workflow")
    state.workflows[workflow_name] = (graphs, obj.code)

def encodeInstances(list_name, inst):
    if not isinstance(inst, list):
        return encodeInstances(list_name, [inst])
    return f"{list_name} = [" + ", ".join([ str(a.model_dump()) for a in inst ] ) + "]"    

def encodeGroup(list_name, names):
    if not isinstance(names, list):
        return encodeGroup(list_name, [names])
    return encodeInstances(list_name, [ InstanceInfo(instance_tag=n, user_info="") for n in names ])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Need param file")

    config = readManagerConfigFile(sys.argv[1])

    amsc_llm_0t = ChatOpenAI(
        model="gpt-oss-120b",
        base_url="https://api.i2-core.american-science-cloud.org/",
        temperature=0,
        api_key = readI2APIkey(config.agent.i2api_key_path)
    )

    if 0:
        #Test the function manifest includes all the agent actions
        print(function_manifest())

    if 0:
        #Test state serialization
        state = State()
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ) )
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", "action_inst"))

        state.instances["sources"] = encodeInstances("sources", [ SourceConfig(name="wall_0", source=WallSource(timeslice=0, momentum=(0,0,0,0)) ),  SourceConfig(name="point_0", source=PointSource(location=(0,0,0,0)) )   ])            
        state.groups["source_group"] = SourceGroup(code = encodeGroup("source_group", ["wall_0", "point_0"]) )

        fn = "test_state.json"
        if os.path.exists(fn):
            os.remove(fn)
        checkpointState(state, fn)

        rstate = reloadStateCheckpoint(fn)

        assert "actions" in rstate.instances
        assert rstate.instances["actions"] == state.instances["actions"]
        assert "agroup1" in rstate.groups and isinstance(rstate.groups["agroup1"], ActionGroup)
        assert rstate.groups["agroup1"].code == state.groups["agroup1"].code

        assert "sources" in rstate.instances
        assert rstate.instances["sources"] == state.instances["sources"]
        assert "source_group" in rstate.groups and isinstance(rstate.groups["source_group"], SourceGroup)
        assert rstate.groups["source_group"].code == state.groups["source_group"].code



    def testEnactor(func, args):
        print("ENACTING ",func.__name__, args)
        return func(*args)

    if 0:
        #Test solver agent
        state = State()
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ) )
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", "action_inst"))

        initializeState(amsc_llm_0t, state)

        ag1_h = retrieveGroupHandle("agroup1")
        sg1_h = createSolverGroup("sgroup1", ag1_h)
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)


    if 0:
        #Test source agent
        state = State()
        initializeState(amsc_llm_0t, state)

        sg1_h = createSourceGroup("sgroup1")
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)


    if 0:
        #Test propagator agent
        state = State()
        state.instances["sources"] = encodeInstances("sources", [ SourceConfig(name="wall_0", source=WallSource(timeslice=0, momentum=(0,0,0,0)) ),  SourceConfig(name="point_0", source=PointSource(location=(0,0,0,0)) )   ])            
        state.groups["source_group"] = SourceGroup(code = encodeGroup("source_group", ["wall_0", "point_0"]) )
        state.instances["solvers"] = encodeInstances("solvers", [ SolverConfig(name="solv", solver_args=RBPrecCGsolver(residual=1e-8,maxIteration=10000,guesser=""), action="action_1")] )    
        state.groups["solver_group"] = SolverGroup(code = encodeGroup("solver_group", "solv") )

        initializeState(amsc_llm_0t, state)

        src_h = retrieveGroupHandle("source_group")
        slv_h = retrieveGroupHandle("solver_group")
        prop_h = createPropagatorGroup("prop1", src_h, slv_h)
            
        graph = prop_h.parent_node
        graph.eval(enactor=testEnactor)

    if 0:
        #Test meson2pt agent
        state = State()
        state.instances["propagators"] = encodeInstances("propagators",  [ PropagatorConfig(name="prop_wall_t32", source="wall_src_t32", solver="solver"),   PropagatorConfig(name="prop_wall_t0", source="wall_src_t0", solver="solver")     ]  )
        state.groups["prop_group"] = PropagatorGroup(code = encodeGroup("prop_group", ["prop_wall_t32", "prop_wall_t0"]) )

        initializeState(amsc_llm_0t, state)

        prop_h = retrieveGroupHandle("prop_group")
        meson_h = createMeson2ptGroup("meson2pt_group",prop_h)

        graph = meson_h.parent_node
        graph.eval(enactor=testEnactor)


    if 0:
        #Test smeared propagator agent
        state = State()
        
        state.instances["propagators"] = encodeInstances("propagators", [ PropagatorConfig(name="prop_wall_t32", source="wall_src_t32", solver="solver"),   PropagatorConfig(name="prop_wall_t0", source="wall_src_t0", solver="solver")     ])    
        state.groups["prop_group"] = PropagatorGroup(code = encodeGroup("prop_group", ["prop_wall_t32", "prop_wall_t0" ]) )

        initializeState(amsc_llm_0t, state)

        prop_h = retrieveGroupHandle("prop_group")
        sprop_h = createSmearedPropagatorGroup("sprop_group",prop_h)

        graph = sprop_h.parent_node
        graph.eval(enactor=testEnactor)


    if 0:
        #Test meson2pt agent with smeared props
        state = State()        
        state.instances["propagators"]  = encodeInstances("propagators", [ PropagatorConfig(name="prop_wall_t32", source="wall_src_t32", solver="solver"),   PropagatorConfig(name="prop_wall_t0", source="wall_src_t0", solver="solver")     ])        
        state.instances["smeared_propagators"] = encodeInstances("smeared_propagators", [ SmearedPropagatorConfig(name="sprop_w32", input_prop="prop_wall_t32", smearing=WallSmear(momentum=(0,0,0,0))),
                       SmearedPropagatorConfig(name="sprop_w0", input_prop="prop_wall_t0", smearing=WallSmear(momentum=(0,0,0,0))),
                         ])
        state.groups["sprop_group"] = SmearedPropagatorGroup(code = encodeGroup("sprop_group", ["sprop_w32", "sprop_w0"]) )
                                                             
        initializeState(amsc_llm_0t, state)

        prop_h = retrieveGroupHandle("sprop_group")
        meson_h = createMeson2ptGroup("meson2pt_group",prop_h)

        graph = meson_h.parent_node
        graph.eval(enactor=testEnactor)        


    if 0:
        #Test eigensolver agent
        state = State()        
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ))
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", "action_inst"))

        initializeState(amsc_llm_0t, state)

        ag1_h = retrieveGroupHandle("agroup1")
        sg1_h = createEigenSolverGroup("sgroup1", ag1_h)
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)        


    if 0:
        #Test solver agent with eigenvectors
        state = State()
        
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ))    
        state.groups["agroup1"] = ActionGroup(code = encodeGroup("agroup1", "action_inst") )
       
        state.instances["eigensolvers"] = encodeInstances("eigensolvers",  EigenSolverConfig(name="esol1", action="action_inst", solver_args=LanczosEigenSolver(cheby=ChebyParams(alpha=0.01,beta=3.2,Npoly=101), Nstop=100, Nk=100, Nextra=10, resid=1e-7, MaxIt=20, storeEvecs=False, fileStem="" ))  )
        state.groups["egroup1"] = EigenSolverGroup(code = encodeGroup("egroup1", "esol1"))

        initializeState(amsc_llm_0t, state)

        ag1_h = retrieveGroupHandle("agroup1")
        eg1_h = retrieveGroupHandle("egroup1")
        sg1_h = createSolverGroup("sgroup1", ag1_h, eg1_h)
            
        graph = sg1_h.parent_node
        graph.eval(enactor=testEnactor)        


    if 0:
        #Test XML output
        state = State()
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ))    
        state.gauge = GaugeFieldConfig(Lx=4,Ly=4,Lz=4,Lt=4,config=UnitGauge())
        xml = state.toHadronsXML()
        print(xml.toString())

    if 0:
        #Test code generator output
        state = State()
        state.instances["actions"] = encodeInstances("actions", ActionConfig(name="action_inst", action=DWFaction(Ls=12, mass=0.01, M5=1.8) ))    
        state.gauge = GaugeFieldConfig(Lx=4,Ly=4,Lz=4,Lt=4,config=UnitGauge())
        strm = io.StringIO()
        state.toXMLgeneratorCodeStream(strm)
        print(strm.getvalue())
        state.toXMLgeneratorCode("test.py")