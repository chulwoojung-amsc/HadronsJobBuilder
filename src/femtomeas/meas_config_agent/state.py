from femtomeas.agent_common.python_output_agent import executeCode, executeCodeAndParse
from femtomeas.agent_common.python_update_agent import InstanceInfo
from .agent_workflow_base import BaseGroup
from typing import Dict, ClassVar, List
from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
import json
from femtomeas.agent_common.common import Print
from .gauge import GaugeFieldConfig
from .hadrons_xml import HadronsXML
from pathlib import Path
from .agent_workflow_globals import instance_class_registry

def get_import_line(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    return f"from {cls.__module__} import {cls.__qualname__}"

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

    gauge: GaugeFieldConfig | None = Field(None,description="The gauge configuration parameters")
    
    def isValidInstance(self, name: str, instance_class : str):
        assert instance_class in self.instances
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

    _instance_cache: ClassVar[dict[str, dict[str, dict[str, BaseModel]]]] = {}  #[instance_class][instance_code][instance_name]
    def getInstances(self, instance_class: str)->Dict[str,BaseModel]:
        """Return a dictionary of instances generated from the stored code string"""   
        assert instance_class in instance_class_registry
        if instance_class not in self.instances:
            return {}
        instance_code = self.instances[instance_class]

        #Caching
        if instance_class not in self._instance_cache:
            self._instance_cache[instance_class] = {}       

        #Use instance code as a singleton key, allowing us to regenerate the cached instances if the code has changed
        if instance_code not in self._instance_cache[instance_class]:
            self._instance_cache[instance_class].clear() #clear old code and instances if exist
            self._instance_cache[instance_class][instance_code] = {}

            r, e = executeCodeAndParse(self.instances[instance_class], instance_class_registry[instance_class], instance_class)
            if len(e) > 0:
                raise Exception(f"Executing code gave the following exceptions: {e}")            

            for p in r:
                self._instance_cache[instance_class][instance_code][p.name] = p

        return self._instance_cache[instance_class][instance_code]

    def getInstance(self, instance_class: str, instance_name: str)->BaseModel | None:
        """Return a particular instance by name as generated from the stored code string"""   
        instances = self.getInstances(instance_class)        
        if instance_name not in instances:
            return None
        else:
            return instances[instance_name]

    _group_cache: ClassVar[dict[str, List[str]]] = {}
    def getGroup(self, group_name)->List[str] | None:
        """Get the list of instance names in the group, based on the stored code"""
        if group_name not in self.groups:            
            return None #even if in cache
        if group_name not in self._group_cache:            
            r, e = executeCodeAndParse(self.groups[group_name].code, InstanceInfo, group_name) 
            if len(e) > 0:
                raise Exception(f"Executing code gave the following exceptions: {e}")      
            assert isinstance(r, list)
            self._group_cache[group_name] = r            

        return self._group_cache[group_name]


    def _toHadronsXMLbase(self)->HadronsXML:
        """
        Set all elements bar the gauge module, which needs special treatment
        """
        xml = HadronsXML()
        xml.setRunID(1234) #What does this do?

        for instance_class in self.instances.keys():
            r = self.getInstances(instance_class)
            for _, a in r.items():
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
        #Import modules containing models
        for instance_class in self.instances.keys():
            assert instance_class in instance_class_registry
            cls = instance_class_registry[instance_class]
            stream.write(get_import_line(cls)+ '\n')

        #Setup and start
        stream.write(f"""import sys
from femtomeas.meas_config_agent.hadrons_xml import HadronsXML
from femtomeas.meas_config_agent.gauge import GaugeFieldConfig

xml = HadronsXML()
xml.setRunID(1234)        
        
""")

        #Output instance code
        for instance_class, instance_code in self.instances.items():
            cls = instance_class_registry[instance_class]
            stream.write(instance_code)
            stream.write(f"""
for item in {instance_class}:
    rm = {cls.__name__}.model_validate(item)
    rm.setXML(xml)
""")

        #GaugeConfig
        stream.write(f"""
{"gauge = " + str(self.gauge.model_dump())}
rm = GaugeFieldConfig.model_validate(gauge)
rm.setXML(xml)        
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
