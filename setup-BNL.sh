#source ~/miniconda3/bin/activate
#conda activate ai
#export NERSC_SFAPI_KEY_PATH=$(readlink -f ~/AmSC_auth/priv_key.pem)
#source /home/ckelly/AmSC_auth/setup_amsc_llm.sh
#https://docs.i2-core.american-science-cloud.org/docs/api-access
export AMSC_I2_API_KEY=$(cat /home/chulwoo/Claude/BNL/BNL-key)
#export OPENAI_API_KEY=${AMSC_I2_API_KEY}
#export OPENAI_BASE_URL="https://inference0-api.sdcc.bnl.gov"

# Agent LLM endpoint, read by workflow.py (env-switchable per environment).
# Set FEMTOMEAS_LLM_MODEL to whatever model the BNL server exposes, and add a
# trailing /v1 to the base URL if that server requires it (test with --skip-agent,
# which prints the resolved endpoint, then a real run).
#  models   : gemma-4-26b nemotron-3-ultra-550b-nvfp4 gpt-oss-120b nemotron-3-super-120b
export FEMTOMEAS_LLM_MODEL="nemotron-3-ultra-550b-nvfp4"
export FEMTOMEAS_LLM_MODEL="nemotron-3-super-120b"
#export FEMTOMEAS_LLM_MODEL="gpt-oss-120b"
export FEMTOMEAS_LLM_BASE_URL="https://inference0-api.sdcc.bnl.gov"
export FEMTOMEAS_LLM_API_KEY=${AMSC_I2_API_KEY}

DIR=/home/chulwoo/Claude/HadronsJobBuilder_master
unset PYTHONPATH
export PYTHONPATH=${DIR}/src:${DIR}/build/lib${PYTHONPATH:+:${PYTHONPATH}}

# API backend: LOCAL = real execution on this machine, SPOOF = fake dry-run,
# IRI/IRI_SF_HYBRID/SF = remote (NERSC). Pre-set FEMTOMEAS_API_IMPL to override.
#export FEMTOMEAS_API_IMPL=${FEMTOMEAS_API_IMPL:-LOCAL}
export FEMTOMEAS_API_IMPL="SPOOF"
#export FEMTOMEAS_API_IMPL="LOCAL"
