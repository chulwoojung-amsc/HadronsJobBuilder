from femtomeas.agent_common.python_update_agent import checkUsedInstances, InstanceInfo
from femtomeas.agent_common.python_output_agent import executeCodeAndParse

from pydantic import BaseModel, Field, ConfigDict, NonNegativeInt, TypeAdapter
from typing import Literal, Union, List, Optional, Tuple

class TestModel(BaseModel):
   """A test model"""
   value: int = Field(..., description="A value")
   name: str = Field(..., description="Instance name")

#Check no rewrite for valid instances
inst_list_nm = "inst_list"
inst_code = """inst_list = [ {"value": 123, "name" : "inst1"}, {"value": 234, "name" : "inst2"} ]"""   

_, e = executeCodeAndParse(inst_code, TestModel, inst_list_nm)
assert len(e) == 0 

use_inst_list_nm = "use_inst"
use_inst_code = """use_inst = [ {"instance_tag" : "inst1" , "user_info" : "" }, {"instance_tag" : "inst2" , "user_info" : "" } ]"""
_, e = executeCodeAndParse(use_inst_code, InstanceInfo, use_inst_list_nm)
assert len(e) == 0 


checker = lambda m: True
v, u = checkUsedInstances(use_inst_list_nm, use_inst_code, TestModel, inst_list_nm, inst_code, checker)
assert v

#Check it removes entries from the list that point to no-longer-existing instances
inst_code = """inst_list = [ {"value": 123, "name" : "inst1"} ]"""   
checker = lambda m: True
v, u = checkUsedInstances(use_inst_list_nm, use_inst_code, TestModel, inst_list_nm, inst_code, checker)
assert not v
new_used_list, e = executeCodeAndParse(u, InstanceInfo, use_inst_list_nm)
assert len(e) == 0
assert len(new_used_list) == 1 and new_used_list[0].instance_tag == "inst1"

#Check it removes entries from the list that do not pass the checker
checker = lambda m: False if m.name == "inst2" else True
use_inst_code = """use_inst = [ {"instance_tag" : "inst1" , "user_info" : "" }, {"instance_tag" : "inst2" , "user_info" : "" } ]"""
v, u = checkUsedInstances(use_inst_list_nm, use_inst_code, TestModel, inst_list_nm, inst_code, checker)
assert not v
new_used_list, e = executeCodeAndParse(u, InstanceInfo, use_inst_list_nm)
assert len(e) == 0
assert len(new_used_list) == 1 and new_used_list[0].instance_tag == "inst1"