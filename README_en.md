<div align="center">
<img src="docs/public/logo.png" width="200"  alt="" />
<h2>CourseGraph: Automatic Construction of Course Knowledge Graphs Using Large Models</h2>

<p>
    <a href="README.md">Chinese</a> | <b>English</b>
</p>
</div>

CourseGraph utilizes large language models and various prompt optimization techniques to automatically extract knowledge points from textbooks and books, forming a knowledge graph centered around courses, chapters, and knowledge points. To enrich each knowledge point, CourseGraph can link relevant exercises, extended reading materials, and other resources. Additionally, it can leverage multimodal large models to extract information from pptx files, images, and videos, establishing connections with the knowledge points.

## 🤔 Limitations

- Currently, only basic knowledge graph extraction and pptx parsing have been implemented, with room for optimization
- Video parsing is still in the planning phase

## 📈 Future Development

- Improve prompt engineering and explore using Agents for related tasks
- Knowledge graph-based question answering (KBQA or Graph-RAG)

## 🚀 Quick Start

First, obtain an Alibaba Cloud Tongyi Qianwen [API Key](https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key?spm=a2c4g.11186623.0.0.1be847bbvv6p4o), then choose local installation or docker installation:

### Option 1: Local Installation

#### Install Dependencies

Ensure [Anaconda](https://www.anaconda.com/) (or [Miniconda](https://docs.conda.io/en/miniconda.html)), [Neo4j](https://neo4j.com/) and [Rust](https://www.rust-lang.org/) are installed, then execute:

```bash
git clone git@github.com:CPU-DS/CourseGraph.git
cd CourseGraph
conda create -n cg python=3.10 -y
conda activate cg
pip install poetry
poetry config virtualenvs.create false
poetry install
cd rust
maturin develop
cd ..
```

On Linux, libreoffice is required for document conversion. For Debian-based systems: 

```bash
sudo apt install libreoffice
```

#### Execute

Provide the Neo4j connection address, username, and password, then execute:
```bash
python examples/get_knowledge_graph.py -u http://localhost:7474 -n neo4j -p neo4j
```

### Option 2: Docker Installation

```bash
git clone git@github.com:wangtao2001/CourseGraph.git
cd CourseGraph
export DASHSCOPE_API_KEY=
docker-compose -f docker/docker-compose.yml up -d
python examples/get_knowledge_graph.py
```

## 📚 Documentation

Documentation can be found in the `docs` directory, or you can visit the [online documentation](https://CPU-DS.github.io/CourseGraph/) (As the project features are still under rapid development, the online documentation is not yet ready). If you wish to customize the online documentation, follow these steps:

#### Install Dependencies and Preview

The documentation is built with [VitePress](https://vitepress.dev/), requiring [Node.js](https://nodejs.org/) 18 or above. Execute:

```bash
cd docs
npm i
npm run docs:dev
```

Open [http://localhost:5173/](http://localhost:5173/) in your browser to preview.

## 🛠️ Contributing, Protocol and Citation

[PR](https://github.com/CPU-DS/CourseGraph/pulls) and [Issues](https://github.com/CPU-DS/CourseGraph/issues) are welcome, as well as any form of contribution.

This project is open-sourced under the [MIT license](LICENSE). If you find CourseGraph helpful for your work, please refer to [CITATION.cff](CITATION.cff) (or click the `Cite this repository` button on the right side of the Repository) to cite.