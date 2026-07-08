#source ~/miniconda3/bin/activate
#conda activate ai
#export NERSC_SFAPI_KEY_PATH=$(readlink -f ~/AmSC_auth/priv_key.pem)
#source /home/ckelly/AmSC_auth/setup_amsc_llm.sh
#https://docs.i2-core.american-science-cloud.org/docs/api-access
export AMSC_I2_API_KEY=$(cat /home/chulwoo/Claude/API_KEY-AmSC)
export OPENAI_API_KEY=${AMSC_I2_API_KEY}
#export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="https://api.i2-core.american-science-cloud.org/v1"
DIR=/home/chulwoo/Claude/HadronsJobBuilder_kelly
unset PYTHONPATH
export PYTHONPATH=${DIR}/src:${DIR}/build/lib${PYTHONPATH:+:${PYTHONPATH}}

# API backend: LOCAL = real execution on this machine, SPOOF = fake dry-run,
# IRI/IRI_SF_HYBRID/SF = remote (NERSC). Pre-set FEMTOMEAS_API_IMPL to override.
#export FEMTOMEAS_API_IMPL=${FEMTOMEAS_API_IMPL:-LOCAL}
export FEMTOMEAS_API_IMPL="SPOOF"
