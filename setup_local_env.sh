#!/bin/bash
set -e

echo "This script replicates the Dockerfile environment on your host Linux system."
echo "It will ask for sudo privileges to install system packages via apt-get."

# 1. Install dependencies
echo ""
echo "[1/7] Installing system dependencies (requires sudo)..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    make build-essential ca-certificates libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev git gcc g++ apt-utils ninja-build

# 2. Install CMake 3.27.7 locally if needed
echo ""
echo "[2/7] Checking/Installing CMake..."
if ! command -v cmake &> /dev/null || [ "$(cmake --version | head -n1 | awk '{print $3}')" \< "3.21" ]; then
    echo "CMake < 3.21 or not installed. Installing 3.27.7 locally..."
    wget -q https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-x86_64.sh
    mkdir -p "$HOME/.local/cmake"
    sh cmake-3.27.7-linux-x86_64.sh --skip-license --prefix="$HOME/.local/cmake"
    rm cmake-3.27.7-linux-x86_64.sh
    export PATH="$HOME/.local/cmake/bin:$PATH"
    grep -q '.local/cmake/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/cmake/bin:$PATH"' >> ~/.bashrc
else
    echo "Sufficient CMake version is already installed."
fi

# 3. Install Pyenv and Python 3.11.6
echo ""
echo "[3/7] Setting up pyenv and Python 3.11.6 for proper dependency isolation..."
if [ ! -d "$HOME/.pyenv" ]; then
    curl -L https://pyenv.run | /bin/bash
    grep -q 'PYENV_ROOT' ~/.bashrc || {
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
        echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
        echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    }
fi
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

if ! pyenv versions | grep -q "3.11.6"; then
    pyenv install 3.11.6
fi
pyenv global 3.11.6
pyenv rehash

# 4. Install Poetry
echo ""
echo "[4/7] Installing Poetry 1.8.2..."
export POETRY_VERSION=1.8.2
export POETRY_HOME="$HOME/.local/poetry"
if [ ! -d "$POETRY_HOME" ]; then
    curl -sSL https://install.python-poetry.org | python3 -
    grep -q '.local/poetry/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/poetry/bin:$PATH"' >> ~/.bashrc
fi
export PATH="$POETRY_HOME/bin:$PATH"

# Move to repository root
ROOT_DIR="/users/cboumalh/workplace/flatnav"
cd "$ROOT_DIR"

# 5. Build flatnav
echo ""
echo "[5/7] Building flatnav wheel..."
cd "$ROOT_DIR/python-bindings"
pip install --upgrade pip wheel setuptools scikit-build
rm -rf build dist *.egg-info

echo "Installing flatnav in editable mode with optimized flags..."
CMAKE_ARGS='-DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_DO_STRIP=OFF -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=-O3\ -g\ -fno-omit-frame-pointer\ -fno-optimize-sibling-calls' \
    python3 -m pip install -v -e . --force-reinstall --no-build-isolation

echo "Building flatnav wheel..."
python setup.py bdist_wheel
FLATNAV_WHEEL_PATH=$(ls -1 dist/*.whl | head -n 1)
FLATNAV_WHEEL=$(realpath "$FLATNAV_WHEEL_PATH")

# 6. Build Hnswlib (fork)
echo ""
echo "[6/7] Building extended hnswlib..."
cd "$ROOT_DIR"
if [ ! -d "hnswlib-original" ]; then
    git clone https://github.com/BlaiseMuhirwa/hnswlib-original.git
fi
cd hnswlib-original/python_bindings
poetry install --no-root
poetry run python setup.py bdist_wheel
HNSWLIB_WHEEL_PATH=$(ls -1 dist/*.whl | head -n 1)
HNSWLIB_WHEEL=$(realpath "$HNSWLIB_WHEEL_PATH")

# 7. Setup experiments dependency
echo ""
echo "[7/7] Setting up Poetry dependencies in experiments..."
cd "$ROOT_DIR/experiments"

# Remove previously added references if they exist
poetry remove flatnav faiss-cpu hnswlib 2>/dev/null || true

# Add newly built wheels and faiss-cpu
poetry add "$FLATNAV_WHEEL" faiss-cpu "$HNSWLIB_WHEEL"
poetry install --no-root

echo ""
echo "========================================================"
echo "Setup successfully complete!"
echo "Please run:  source ~/.bashrc"
echo "Then you can 'cd /mydata/flatnav/experiments' and run your make commands:"
echo "Current directory: \$(pwd)"
echo "Example: make sift-big-bench-flatnav"
echo "========================================================"
