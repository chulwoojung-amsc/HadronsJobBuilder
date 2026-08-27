"""Smoke-test the fit_workflow.py conversational agent non-interactively.

Drives the four-stage parameterAgent conversation (DATASET / FIT MODEL /
COVARIANCE & RESAMPLING / RUN & OUTPUT) with pexpect, answering the LLM's
questions by keyword and falling back to "use the default" for anything
unmatched. Target: a single-channel pion fit over the six-config SDCC XML run.

    source setup.sh   # (the driver sources it inside the spawn too)
    python3 main/drive_fit_workflow.py

The conversation is LLM-driven and therefore nondeterministic; this is a smoke
test that the plumbing works end to end, not a fixed-output regression.
"""
import os
import sys
import pexpect

CKPT = "/tmp/smoke_fit_ckpt.json"

#A fresh (nonexistent) checkpoint path -> fitAgent starts clean and writes its
#checkpoint there instead of polluting the repo cwd.
CMD = ("source setup.sh && /home/chulwoo/Claude/.venv/bin/python3 -u "
       "main/fit_workflow.py main/fit_workflow_config.json "
       f"--reload-checkpoint {CKPT} --execute-fit")

#Full description handed over on the first prompt so the agent can fill many
#fields at once; per-field follow-ups are then keyword-matched below.
FULL_INFO = (
    "Fit the pion two-point correlator from the Hadrons XML output in "
    "/home/chulwoo/Claude/hadrons_jobs/pion_16c_sdcc. Auto-discover the single "
    "observable, so leave the data file name empty. Nt is 16 with no binning "
    "(Nbin=1). Fit operator column index 0 only, and label that operator 'pion'. "
    "Use the time range tmin=2 to "
    "tmax=7. Use 2 normal exponential states and 0 alternating-sign states with "
    "the analytic backend. Put the covariance on the raw correlator, jackknife "
    "inner and bootstrap outer resampling, 20 outer bootstrap samples, "
    "Ledoit-Wolf shrinkage 1.0. Do not use differential-evolution refinement, no "
    "tmin tuning, and do not make plots. Use defaults for everything else.")

YESNO_CUES = ["[y/n]", "either 'y' or 'n'", "is the following correct"]

#First rule whose any keyword hits (tried on the last line, then a 3-line window)
#wins. Order matters: more specific questions first.
RULES = [
    (["directory", "data_path", "where the data", "where is the data", "where the correlator"],
     "/home/chulwoo/Claude/hadrons_jobs/pion_16c_sdcc"),
    (["file name", "name of the data", "observable name", "base name"],
     "leave it empty to auto-discover the single observable"),
    (["binning", "nbin"], "1"),
    (["temporal extent", "value of nt", "confirm nt", "what nt", "nt="], "16"),
    (["label", "op_names", "operator name", "name the operator"], "pion"),
    (["which operator", "operator column", "operator indices", "i_fit", "columns to include"], "0"),
    (["time-range", "time range", "fit range", "tmin", "tmax", "range of time"], "tmin 2, tmax 7"),
    (["alternating"], "0"),
    (["normal exponential", "number of exponential", "how many exponential", "nmass"], "2"),
    (["backend"], "analytic"),
    (["circular", "wrap-around", "wrap‑around", "wraparound"], "false"),
    (["outer bootstrap", "nboots_outer", "outer resampling samples", "number of outer"], "20"),
    (["outer resample", "error estimation"], "bootstrap"),
    (["inner resample", "covariance estimation"], "jackknife"),
    (["shrinkage", "ledoit", "lw"], "1.0"),
    (["random seed", "rng_seed", "seed"], "use the default"),
    (["block size", "block_size", "block length"], "use the default"),
    (["data vector", "cov_on", "log-ratio", "raw correlator"], "corr"),
    (["differential evolution", "de refinement", "de_refinement", "refine"], "no"),
    (["tmin tuning", "tmin_tuning", "tuning"], "none"),
    (["make plots", "make_plots", "diagnostic plot", "save plot", "plots"], "no"),
    (["output directory", "plot_dir", "suffix"], "use the default"),
]

DEFAULT_ANSWER = "Use the default value."


def question_lines(before):
    """The real question shown to the user, with the agent's debug dump removed.

    parameterAgent prints  `OUTPUT <AgentOutput repr incl. params_struct=...> done=...`
    and only then the actual prompt via input(). Keywords inside the echoed
    params_struct JSON (e.g. `"Nboots_outer": 20`) would otherwise collide with the
    rules, so drop everything up to and including the last debug-dump line and keep
    only what follows - the question itself."""
    lines = (before or "").splitlines()
    cut = 0
    for i, l in enumerate(lines):
        s = l.lstrip()
        if s.startswith("OUTPUT ") or "params_struct=" in l or "done=True" in l or "done=False" in l:
            cut = i + 1
    return [l.strip() for l in lines[cut:] if l.strip()]


def answer_for(qlines):
    window = " ".join(qlines[-3:]).lower()
    if any(c in window for c in YESNO_CUES):
        return "y"
    last = qlines[-1].lower() if qlines else ""
    for scope in (last, window):
        for keys, ans in RULES:
            if any(k in scope for k in keys):
                return ans
    return DEFAULT_ANSWER


def main():
    if os.path.exists(CKPT):
        os.remove(CKPT)

    child = pexpect.spawn("/bin/bash", ["-lc", CMD],
                          cwd="/home/chulwoo/Claude/HadronsJobBuilder_kelly",
                          encoding="utf-8", timeout=600)
    child.logfile_read = sys.stdout

    #First: the free-text description prompt (ends with "run: ", not " : ").
    child.expect(r"Describe the correlator fit[^\n]*: ")
    print(f"\n[driver] describe -> FULL_INFO", flush=True)
    child.sendline(FULL_INFO)

    n_answers = 0
    last = ("", "")
    repeats = 0
    MAX_ANSWERS = 150
    while True:
        i = child.expect([r" : $", pexpect.EOF, pexpect.TIMEOUT])
        if i == 1:
            break
        if i == 2:
            print("\n[driver] TIMEOUT waiting for a prompt", flush=True)
            child.terminate(force=True)
            sys.exit(2)

        qlines = question_lines(child.before)
        question = " | ".join(qlines[-3:])
        ans = answer_for(qlines)

        #Loop guard: if the same (question, answer) recurs, escalate to full info
        #once, then bail rather than spin forever.
        if (question, ans) == last:
            repeats += 1
            if repeats == 3:
                ans = FULL_INFO
            elif repeats >= 5:
                print(f"\n[driver] STUCK on {question!r}; giving up", flush=True)
                child.terminate(force=True)
                sys.exit(3)
        else:
            repeats = 0
        last = (question, ans)

        n_answers += 1
        if n_answers > MAX_ANSWERS:
            print("\n[driver] exceeded MAX_ANSWERS; giving up", flush=True)
            child.terminate(force=True)
            sys.exit(4)

        print(f"\n[driver] Q: {question!r} -> A: {ans!r}", flush=True)
        child.sendline(ans)

    child.close()
    print(f"\n[driver] child exited with status {child.exitstatus}", flush=True)
    sys.exit(child.exitstatus or 0)


if __name__ == "__main__":
    main()
