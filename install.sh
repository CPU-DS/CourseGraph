conda create -n cg python=3.10 -y
conda activate cg
pip install poetry
poetry config virtualenvs.create false
poetry install
cd rust
maturin develop
cd ..