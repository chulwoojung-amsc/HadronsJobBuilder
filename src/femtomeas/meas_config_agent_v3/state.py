from femtomeas.agent_common.python_output_agent import executeCode, executeCodeAndParse
from .agent_workflow_base import BaseGroup
from typing import Dict
from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
import json
from femtomeas.agent_common.common import Print

class DictRef:
    """A 'reference' to an element in a dict, allowing it to be obtained and set like a member variable"""
    def __init__(self, d : dict, key: str):
        self._dict = d
        self._key = key

    @property
    def value(self):
        return self._dict[self._key]

    @value.setter
    def value(self, v):
        self._dict[self._key] = v

class State(BaseModel):
    instances: Dict[str,str] = Field({}, description="map of instance class (actions, solvers, etc) to the code for generating those instances")
    groups: Dict[str, BaseGroup] = Field({}, description="map of group name to a BaseGroup-derived objects containing code for generating the list of instances in the group (by name)")

    def isValidInstance(self, name: str, instance_class : str):
        assert instance_class in self.instance
        r, e = executeCode(self.instances[instance_class])
        if len(e) > 0:
            raise Exception(f"Executing code gave the following exceptions: {e}")
        assert instance_class in r.keys()
        instances = r[instance_class]

        for p in instances:
            if p["name"] == name:
                return True
        return False

    def getInstanceCode(self, instance_class: str)->DictRef:
        """Return a 'reference' to the entry in the instance-code dictionary for this instance class"""
        if instance_class not in self.instances:
            self.instances[instance_class] = None
        return DictRef(self.instances, instance_class)


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