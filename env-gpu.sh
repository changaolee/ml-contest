ROOT_PATH="$(pwd)"
UTILS_PATH="${ROOT_PATH}/utils"
export PYTHONPATH="${ROOT_PATH}:${UTILS_PATH}:${PYTHONPATH}"

pip install -r requirements-gpu.txt