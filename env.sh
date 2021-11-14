ROOT_PATH="$(pwd)"
UTILS_PATH="${ROOT_PATH}/utils"
PACKAGES_PATH="${ROOT_PATH}/packages"
export PYTHONPATH="${ROOT_PATH}:${UTILS_PATH}:${PACKAGES_PATH}:${PYTHONPATH}"

pip install -r requirements.txt