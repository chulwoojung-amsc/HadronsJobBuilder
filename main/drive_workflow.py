"""Drive main/workflow.py through the job-submission conversation.

Reloads the complete pion checkpoint (skips the measurement agent), then
answers the hadronsSubmissionAgent's questions by keyword match on the
question line only (input() re-prints the question as its prompt, so the
text immediately before " : " is the question itself).
"""
import sys
import pexpect

CMD = ("/home/chulwoo/Claude/.venv/bin/python3 -u main/workflow.py "
       "main/workflow_local.json --reload-checkpoint ckpoint_state.json "
       "--execute-workflow")

FULL_INFO = ("Run on machine Local with account chulwoo on queue local. "
             "Duration 600 seconds. Use MPI rank decomposition 1 1 1 1 "
             "(a single rank). Job group name: pion_bg_test. "
             "Do not copy results to a remote machine.")

YESNO_CUES = ["[y/n]", "either 'y' or 'n'", "is the following correct"]

# ordered: first rule whose any-keyword hits wins; tried on the last line
# first, then on a 3-line window. account/queue precede machine because
# their questions mention "machine" too.
RULES = [
    (["account"], "chulwoo"),
    (["queue"], "local"),
    (["duration", "how long", "job time"], "600 seconds"),
    (["decomposition", "rank_geom", "rank geometry", "mpi rank"], "1 1 1 1"),
    (["how many"], "1"),
    (["job group", "job_group", "collection", "group name", "name to assign"],
     "pion_bg_test"),
    (["copy", "remote machine", "globus", "postprocess"], "no"),
    (["machine"], "Local"),
]

def answer_for(question_lines, n_asked: int) -> str:
    window = " ".join(question_lines[-3:]).lower()
    if any(c in window for c in YESNO_CUES):
        return "y"
    for scope in ([question_lines[-1].lower()] if question_lines else [], [window]):
        for text in scope:
            for keys, ans in RULES:
                if any(k in text for k in keys):
                    return ans
    return FULL_INFO if n_asked == 0 else "y"

child = pexpect.spawn("/bin/bash", ["-lc", CMD],
                      cwd="/home/chulwoo/Claude/HadronsJobBuilder_kelly",
                      encoding="utf-8", timeout=1800)
child.logfile_read = sys.stdout

n_asked = 0
last = ("", "")   # (question, answer) repeat guard
repeats = 0
while True:
    i = child.expect([r" : $", pexpect.EOF, pexpect.TIMEOUT])
    if i == 1:
        break
    if i == 2:
        print("\n[driver] TIMEOUT waiting for prompt or EOF", flush=True)
        child.terminate(force=True)
        sys.exit(2)
    # question = last non-empty lines of the prompt text
    lines = [l.strip() for l in (child.before or "").splitlines() if l.strip()]
    question = " | ".join(lines[-3:])
    ans = answer_for(lines, n_asked)
    if (question, ans) == last:
        repeats += 1
        if repeats >= 3:
            print(f"\n[driver] STUCK repeating {ans!r} to {question!r}; giving full info", flush=True)
            ans = FULL_INFO
            repeats = 0
    else:
        repeats = 0
    last = (question, ans)
    n_asked += 1
    print(f"\n[driver] Q: {question!r} -> A: {ans!r}", flush=True)
    child.sendline(ans)

child.close()
print(f"\n[driver] child exited with status {child.exitstatus}", flush=True)
sys.exit(child.exitstatus or 0)
