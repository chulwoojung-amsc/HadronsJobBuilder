import json
from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple
from pathlib import Path
from textwrap import indent
from .observable_info_models import ObservablesInfo
from femtomeas.meas_config_agent.gauge import GaugeFieldConfig
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.agent_common.common import Print
from femtomeas.agent_common.python_output_agent import executeCode, executeCodeAndParse

# from .action_config import ActionConfig
# from .source_config import SourceConfig
# from .solver_config import SolverConfig
# from .propagator_config import PropagatorConfig
# from .observable_config import ObservableConfig
# from .eigenvectors import EigenSolverConfig
# from .smeared_prop_config import SmearedPropagatorConfig

def checkpointState(state, filename):
    #j = json.dumps({k: v.model_dump() for k, v in state.items()}, indent=2)
    j=json.loads(state.model_dump_json())
    with open(filename, 'w') as wr:
        wr.write(json.dumps(j,indent=2))

def reloadStateCheckpoint(filename):
    Print("Reloading state checkpoint from",filename)
    with open(filename, 'r') as rd:
        j = rd.read()
    return State.model_validate_json(j)

class State(BaseModel):
    query: str | None = Field(None,description="The original query")
    observables: ObservablesInfo | None = Field(None,description="The list of observables and associated relevant information")
    actions: str | None = Field(None,description="Code for generating action instances")
    observable_actions : dict[str, str] | None = Field(None, description="Code for the action instances associated with each observable")

    sources: str | None = Field(None,description="Code for generating source instances")
    observable_sources : dict[str, str] | None = Field(None, description="Code for the source instances associated with each observable")

    eigensolvers: str | None = Field(None,description="Code for generating eigensolver instances")
    observable_eigensolvers : dict[str, str] | None = Field(None, description="Code for the eigensolver instances associated with each observable")

    solvers: str | None = Field(None,description="Code for generating the solver instances")
    observable_solvers : dict[str, str] | None = Field(None, description="Code for the solver instances associated with each observable")

    propagators: str | None = Field(None,description="Code for generating the propagator instances")
    observable_propagators : dict[str, str] | None = Field(None, description="Code for the propagator instances associated with each observable")

    smeared_propagators: str | None = Field(None,description="Code for generating smeared propagator instances")
    observable_smeared_propagators : dict[str, str] | None = Field(None, description="Code for the smeared propagator instances associated with each observable")

    observable_configs : str | None = Field(None,description="Code for generating observable instances")
    observable_observable_configs: dict[str, str] | None = Field(None, description="Code for the observable instances associated with each observable tag")

    gauge: GaugeFieldConfig | None = Field(None,description="The gauge configuration parameters")
    

    def getObservableType(self, observable_tag):
        if self.observables == None:
            raise Exception("state.observables is None!")

        obs_type = None
        for o in self.observables.observables:
            if o.obs_tag == observable_tag:
                obs_type = o.obs_type
                break
        if obs_type == None:
            raise Exception(f"Observable with tag {observable_tag} does not exist!")
        return obs_type
    

    def isValidObservable(self, obs_name):
        for p in self.observables:
            if p.name == obs_name:
                return True
        return False

    # def locateObservable(self, obs_name) -> ObservableInfo | None:
    #     for p in self.observables:
    #         if p.name == obs_name:
    #             return p
    #     return None
    
    def isValidAction(self, action_name):
        r, e = executeCode(self.actions)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "actions" in r.keys()
        actions = r["actions"]

        for p in actions:
            if p["name"] == action_name:
                return True
        return False
    
    def isValidSource(self, source_name):
        r, e = executeCode(self.sources)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "sources" in r.keys()
        sources = r["sources"]

        for p in sources:
            if p["name"] == source_name:
                return True
        return False

    def isValidSolver(self, solver_name):
        r, e = executeCode(self.solvers)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "solvers" in r.keys()
        solvers = r["solvers"]

        for p in solvers:
            if p["name"] == solver_name:
                return True
        return False
    
    def isValidPropagator(self, propagator_name):
        if self.propagators is None:
            return False

        r, e = executeCode(self.propagators)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "propagators" in r.keys()
        propagators = r["propagators"]

        for p in propagators:
            if p["name"] == propagator_name:
                return True
        return False

    def isValidSmearedPropagator(self, sprop_name):
        if self.smeared_propagators is None:
            return False

        r, e = executeCode(self.smeared_propagators)
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert "smeared_propagators" in r.keys()
        smeared_props = r["smeared_propagators"]

        for p in smeared_props:
            if p["name"] == sprop_name:
                return True
        return False

    def _toHadronsXMLbase(self)->HadronsXML:
        """
        Set all elements bar the gauge module, which needs special treatment
        """
        xml = HadronsXML()
        xml.setRunID(1234) #What does this do?

        for c in [(self.actions, ActionConfig, "actions"), (self.sources, SourceConfig, "sources"), (self.eigensolvers, EigenSolverConfig, "eigensolvers"), (self.solvers, SolverConfig, "solvers"), (self.propagators, PropagatorConfig, "propagators"), (self.smeared_propagators, SmearedPropagatorConfig, "smeared_propagators"), (self.observable_configs, ObservableConfig, "observable_configs")]:            
            r, e = executeCodeAndParse(*c)
            if len(e) > 0:
                raise Exception(f"Executing code for type {c[1]} gave the following exceptions: {e}")
            for a in r:
                a.setXML(xml)
        
        return xml
    
    def toHadronsXML(self)->HadronsXML:
        xml=self._toHadronsXMLbase()
        self.gauge.setXML(xml)
        return xml

    def toHadronsXMLsingleConf(self,job_index,override_path = None ):
        """
        Output the XML just for a single configuration.
        job_index : The index of the entry in the range, i.e. 0 -> start, 1 -> start+step,  etc
        override_path : Replace the path in which the file resides, e.g. if it was moved prior to execution
        """
        xml=self._toHadronsXMLbase()
        self.gauge.setXMLsingle(xml,job_index,override_path)
        return xml

    def toXMLgeneratorCodeStream(self, stream):        
        gauge_json = "gauge_json = " + self.gauge.model_dump_json(indent=2)
        
        stream.write(f"""
from femtomeas.meas_config_agent.action_config import ActionConfig
from femtomeas.meas_config_agent.source_config import SourceConfig
from femtomeas.meas_config_agent.solver_config import SolverConfig
from femtomeas.meas_config_agent.propagator_config import PropagatorConfig
from femtomeas.meas_config_agent.smeared_prop_config import SmearedPropagatorConfig
from femtomeas.meas_config_agent.observable_config import ObservableConfig
from femtomeas.meas_config_agent.gauge import GaugeFieldConfig
from femtomeas.meas_config_agent.eigenvectors import EigenSolverConfig                    
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
import sys

def actions(xml):
{indent(self.actions, '    ')}
    for r in actions:
        rm = ActionConfig.model_validate(r)
        rm.setXML(xml)                    

def sources(xml):
{indent(self.sources, '    ')}
    for r in sources:
        rm = SourceConfig.model_validate(r)
        rm.setXML(xml)

def eigensolvers(xml):
{indent(self.eigensolvers, '    ')}
    for r in eigensolvers:
        rm = EigenSolverConfig.model_validate(r)
        rm.setXML(xml)              

def solvers(xml):
{indent(self.solvers, '    ')}
    for r in solvers:
        rm = SolverConfig.model_validate(r)
        rm.setXML(xml)            

def propagators(xml):
{indent(self.propagators, '    ')}
    for r in propagators:
        rm = PropagatorConfig.model_validate(r)
        rm.setXML(xml)          

def smeared_propagators(xml):
{indent(self.smeared_propagators, '    ')}
    for r in smeared_propagators:
        rm = SmearedPropagatorConfig.model_validate(r)
        rm.setXML(xml)              

def observables(xml):
{indent(self.observable_configs, '    ')}
    for r in observable_configs:
        rm = ObservableConfig.model_validate(r)
        rm.setXML(xml)            

def gauge(xml):
{indent(gauge_json,  '    ')}
    rm = GaugeFieldConfig.model_validate(gauge_json)
    rm.setXML(xml)


xml = HadronsXML()
xml.setRunID(1234) #What does this do?        

actions(xml)
sources(xml)
eigensolvers(xml)
solvers(xml)
propagators(xml)
smeared_propagators(xml)
observables(xml)
gauge(xml)
""")

    def toXMLgeneratorCode(self, output_file):
        assert Path(output_file).suffix.lower() == ".py"        
        with open(output_file, 'w') as f:
            self.toXMLgeneratorCodeStream(f)
            f.write("""
if len(sys.argv) == 0:
    raise Exception("Require the XML filename")
xml.write(sys.argv[1])
""")                                        
    