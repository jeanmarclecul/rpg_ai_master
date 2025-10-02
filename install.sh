#!/usr/bin/env bash
set -e
VENV_DIR=".venv_rpg_ai"

python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/Scripts/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Environnement virtuel créé et dépendances installées."
echo "Pour lancer le jeu :"
echo "  source ${VENV_DIR}/bin/activate"
echo "  export HF_HOME=$(pwd)/hf_cache"
echo "  python rpg_ai_master.py --config config.json"
