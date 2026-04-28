from femtomeas.workflow_manager.api_general import *
from femtomeas.workflow_manager.api_tools import *
from femtomeas.workflow_manager.hadrons import *
import time
import stat
import os
import json
from femtomeas.workflow_manager.manager_config import readManagerConfigFile
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

readManagerConfigFile(sys.argv[1])

machine = "Perlmutter"
safe_dir = globals.remote_workdir[machine]
print("Machine",machine, "is up?:", queryMachineStatus(machine))


if 0:
    import femtomeas.workflow_manager.iri_api
    print(femtomeas.workflow_manager.iri_api.getUserAccountProjects(machine))
    


if 0:
    print("TESTING BATCH JOB SUBMISSION")
    jobid = executeBatchJobCompat(machine, '''echo -e '#!/bin/bash\necho "Hello from ${SLURM_PROCID}"' > script.sh
    chmod u+x script.sh
    srun -n 4 ./script.sh
    ''',
    nodes=1, ranks_per_node=4, gpus_per_rank=1, time="600", queue="debug", account="amsc013_g", job_run_dir=safe_dir, exclusive=False, allow_unsafe=True)

    watchJobStatus(machine, jobid)

    
if 0:
    jobid = executeBatchJobCompat(machine, '''echo -e '#!/bin/bash\necho "Hello from ${SLURM_PROCID}"' > script.sh
    chmod u+x script.sh
    srun -n 4 ./script.sh
    ''',
    nodes=1, ranks_per_node=4, gpus_per_rank=1, time="1800", queue="debug", account="amsc013_g", job_run_dir=safe_dir, exclusive=False, allow_unsafe=True)
    getJobState(machine,jobid)
    cancelJob(machine, jobid)

    
if 0:
    test_file = "test_upload.txt"
    with open(test_file,"w") as f:
       f.write("HELLO")

    assert uploadSmallFile(machine, safe_dir + "/test1.txt", test_file ) == True

if 0: #test if it preserves executable privilege  NO
    test_script = "test_upload_script.sh"
    with open(test_script,"w") as f:
       f.write('''#!/bin/bash
       echo HELLO''')
    os.chmod(test_script, os.stat(test_script).st_mode | stat.S_IXUSR)

    assert uploadSmallFile(machine, safe_dir + "/test_upload_script.sh", test_script ) == True

if 0:
    test_script = "test_upload_script.sh"
    with open(test_script,"w") as f:
       f.write('''#!/bin/bash
       echo HELLO''')
    assert uploadSmallFile(machine, safe_dir + "/test_upload_script.sh", test_script ) == True
    remoteChmod(machine, safe_dir + "/test_upload_script.sh", "744")
    
    
    
if 0:
    caught_exception=False
    try:
       remoteMkdir(machine, "pooh", allow_unsafe=True)
    except Exception as e:
       print("Caught expected exception '",e,"'")
       caught_exception = True
    assert caught_exception

if 0:    
    ret = remoteMkdirUnsafe(machine, safe_dir + "/1/2")
    assert ret == 1 or ret == 2

if 0:
    hadronsXMLexample("hadrons_run.xml")
    print("TESTING HADRONS JOB SUBMISSION 1 RANKS, 1 NODE" )
    jobid = submitHadronsJob(machine, "hadrons_run.xml", f"{safe_dir}/test_job", "mp13_g", "debug", "5", (8,8,8,8), (1,1,1,1))
    while(1):
        state = getJobState(machine, jobid)
        print(state)
        if state not in ("new", "queued", "active"):
            print("Detected job completion")
            break    
        time.sleep(10)

    
if 0:
    hadronsXMLexample("hadrons_run.xml")
    print("TESTING HADRONS JOB SUBMISSION 4 RANKS, 1 NODE" )
    jobid = submitHadronsJob(machine, "hadrons_run.xml", f"{safe_dir}/test_job", "mp13_g", "debug", "5", (16,16,8,8), (2,2,1,1))
    while(1):
        state = getJobState(machine, jobid)
        print(state)
        if state not in ("new", "queued", "active"):
            print("Detected job completion")
            break    
        time.sleep(10)

if 0:
    print(globusTransferStatus(machine, "1234"))
        
if 0:
    tid = globusCopyToMachine(machine, safe_dir, "dtn", "/global/cfs/cdirs/mp13/ckelly/globus_source_test_dir", block_until_complete=True)
    for i in range(10):
        print(globusTransferStatus(machine, tid))
        time.sleep(2)

if 0:
    globusCopyFromMachine("dtn", "/global/cfs/cdirs/mp13/ckelly/globus_source_test_dir/copyback",  machine, safe_dir + "/test.dat", block_until_complete=True)

if 0:
    result = downloadFile(machine, "/global/u2/c/ckelly/tocopy")
    print(result)
    
if 1:
    remoteRun(machine, ["rm -f /global/u2/c/ckelly/test_remote_run", "echo hello >  /global/u2/c/ckelly/test_remote_run"])
    result = downloadFile(machine, "/global/u2/c/ckelly/test_remote_run")
    assert "hello" in result
