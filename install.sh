conda create -n cg python=3.10 -y
conda activate cg
pip install poetry
poetry config virtualenvs.create false
poetry install
cd src/course_graph_ext
maturin develop