import time
import stat
import os
import json
from femtomeas.workflow_manager.manager_config import readManagerConfigFile
from femtomeas.workflow_manager.hadrons_workflow import hadronsSubmissionAgent
from femtomeas.meas_config_agent.state import State, GaugeFieldConfig, UnitGauge
import sys

import os
from langchain_openai import ChatOpenAI

amsc_llm_0t = ChatOpenAI(
    model="gpt-oss-120b",
    base_url="https://api.i2-core.american-science-cloud.org/",
    temperature=0
)

llm = amsc_llm_0t

if len(sys.argv) == 1:
    raise Exception("Must provide the manager configuration JSON")

readManagerConfigFile(sys.argv[1])

state = State()
state.gauge = GaugeFieldConfig(Lx=32, Ly=32, Lz=32, Lt=64, config=UnitGauge())

hadronsSubmissionAgent(state, None, llm)
