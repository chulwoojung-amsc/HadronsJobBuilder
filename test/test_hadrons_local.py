from femtomeas.workflow_manager.manager_config import readManagerConfigFile, setupManager
from femtomeas.workflow_manager.hadrons import submitHadronsJob
from femtomeas.workflow_manager.local_api import getJobState
import sys

def hadronsXMLexample(output_file):
    with open(output_file, 'w') as f:
        f.write("""<?xml version='1.0' encoding='us-ascii'?>
<grid>
  <parameters>
    <trajCounter>
      <start>0</start>
      <end>1</end>
      <step>1</step>
    </trajCounter>
    <database>
      <applicationDb>app.db</applicationDb>
      <resultDb>results.db</resultDb>
      <restoreModules>false</restoreModules>
      <restoreMemoryProfile>false</restoreMemoryProfile>
      <restoreSchedule>false</restoreSchedule>
      <statDbBase>stats.db</statDbBase>
      <statDbPeriodMs>1000</statDbPeriodMs>
      <statDbAllRanks>false</statDbAllRanks>
    </database>
    <genetic>
      <popSize>20</popSize>
      <maxGen>100</maxGen>
      <maxCstGen>100</maxCstGen>
      <mutationRate>0.1</mutationRate>
    </genetic>
    <graphFile />
    <scheduleFile />
    <saveSchedule>false</saveSchedule>
    <parallelWriteMaxRetry>-1</parallelWriteMaxRetry>
    <runId>1234</runId>
  </parameters>
  <modules>
    <module>
      <id>
        <name>gauge</name>
        <type>MGauge::Unit</type>
      </id>
      <options />
    </module>
    <module>
      <id>
        <name>DWF_Ls12_M51.8_m0.01</name>
        <type>MAction::DWF</type>
      </id>
      <options>
        <gauge>gauge</gauge>
        <Ls>12</Ls>
        <mass>0.01</mass>
        <M5>1.8</M5>
        <boundary>1 1 1 -1</boundary>
        <twist>0. 0. 0. 0.</twist>
      </options>
    </module>
    <module>
      <id>
        <name>DWF_Ls12_M51.8_m0.01_wall_t0</name>
        <type>MSource::Wall</type>
      </id>
      <options>
        <tW>0</tW>
        <mom>0. 0. 0. 0.</mom>
      </options>
    </module>
    <module>
      <id>
        <name>solver_DWF_Ls12_M51.8_m0.01_1e-8</name>
        <type>MSolver::RBPrecCG</type>
      </id>
      <options>
        <action>DWF_Ls12_M51.8_m0.01</action>
        <maxIteration>10000</maxIteration>
        <residual>1e-08</residual>
        <guesser />
      </options>
    </module>
    <module>
      <id>
        <name>prop_solver_DWF_Ls12_M51.8_m0.01_1e-8_DWF_Ls12_M51.8_m0.01_wall_t0</name>
        <type>MFermion::GaugeProp</type>
      </id>
      <options>
        <source>DWF_Ls12_M51.8_m0.01_wall_t0</source>
        <solver>solver_DWF_Ls12_M51.8_m0.01_1e-8</solver>
      </options>
    </module>
    <module>
      <id>
        <name>point_sink_zerop</name>
        <type>MSink::ScalarPoint</type>
      </id>
      <options>
        <mom>0. 0. 0.</mom>
      </options>
    </module>
    <module>
      <id>
        <name>pion2pt_1</name>
        <type>MContraction::Meson</type>
      </id>
      <options>
        <q1>prop_solver_DWF_Ls12_M51.8_m0.01_1e-8_DWF_Ls12_M51.8_m0.01_wall_t0</q1>
        <q2>prop_solver_DWF_Ls12_M51.8_m0.01_1e-8_DWF_Ls12_M51.8_m0.01_wall_t0</q2>
        <gammas>(Gamma5 Gamma5)</gammas>
        <sink>point_sink_zerop</sink>
        <output>pion2pt_1.out</output>
      </options>
    </module>
  </modules>
</grid>""")






if len(sys.argv) == 1:
    raise Exception("Must provide the manager configuration JSON")

config = readManagerConfigFile(sys.argv[1])
print(type(config))
setupManager(config)

hadronsXMLexample("test.xml")



jobid = submitHadronsJob(machine="local", job_run_dir=config.workflow.sandbox_directories["local"] + "/test",
                 hadrons_xml_file="test.xml",
                 account="", queue="", time="",
                 grid=(4,4,4,4), mpi=(1,1,1,1) )

print(getJobState(jobid))
