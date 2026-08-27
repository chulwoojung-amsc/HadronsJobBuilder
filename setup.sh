#source ~/miniconda3/bin/activate
#conda activate ai
#export NERSC_SFAPI_KEY_PATH=$(readlink -f ~/AmSC_auth/priv_key.pem)
#source /home/ckelly/AmSC_auth/setup_amsc_llm.sh
#https://docs.i2-core.american-science-cloud.org/docs/api-access
export AMSC_I2_API_KEY=$(cat /home/chulwoo/Claude/API_KEY-AmSC)
export OPENAI_API_KEY=${AMSC_I2_API_KEY}
#export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="https://api.i2-core.american-science-cloud.org/v1"

# Agent LLM endpoint, read by workflow.py (env-switchable per environment). The
# base URL here is the langchain ChatOpenAI root (no trailing /v1, unlike
# OPENAI_BASE_URL above). Unset -> workflow.py defaults to these same AmSC values.
export FEMTOMEAS_LLM_MODEL="gpt-oss-120b"
export FEMTOMEAS_LLM_BASE_URL="https://api.i2-core.american-science-cloud.org/"
export FEMTOMEAS_LLM_API_KEY=${AMSC_I2_API_KEY}

DIR=/home/chulwoo/Claude/HadronsJobBuilder_kelly
unset PYTHONPATH
export PYTHONPATH=${DIR}/src:${DIR}/build/lib${PYTHONPATH:+:${PYTHONPATH}}

# API backend: LOCAL = real execution on this machine, SPOOF = fake dry-run,
# IRI/IRI_SF_HYBRID/SF = remote (NERSC). Pre-set FEMTOMEAS_API_IMPL to override.
#export FEMTOMEAS_API_IMPL=${FEMTOMEAS_API_IMPL:-LOCAL}
#export FEMTOMEAS_API_IMPL="SPOOF"
export FEMTOMEAS_API_IMPL="LOCAL"
