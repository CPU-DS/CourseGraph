<div align="center">
<img src="docs/public/logo.png" width="200"  alt="" />
<h2>CourseGraph: Automatic Construction of Course Knowledge Graphs via LLMs</h2>

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

First, obtain an Alibaba Cloud Tongyi Qianwen [API Key](https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key), then choose local installation or Docker installation:

### Option 1: Local Installation

#### Install Dependencies

Ensure [uv](https://docs.astral.sh/uv/), [Neo4j](https://neo4j.com/) and [Rust](https://www.rust-lang.org/) are installed, then execute:

```bash
git clone git@github.com:CPU-DS/CourseGraph.git
cd CourseGraph
uv sync
```

On Linux, libreoffice is required for document conversion. For Debian-based systems: 

```bash
sudo apt install libreoffice
```

#### Execute

Provide the Neo4j connection password and the path to the file to be extracted, then execute:
```bash
uv examples/get_knowledge_graph_pdf.py -p neo4j -f assets/deep-learning-from-scratch.pdf
```

### Option 2: Docker Installation

```bash
git clone git@github.com:wangtao2001/CourseGraph.git
cd CourseGraph
docker-compose -f docker/docker-compose.yml up -d
uv examples/get_knowledge_graph_pdf.py -f assets/deep-learning-from-scratch.pdf
```

## 📚 Documentation

Documentation can be found in the `docs` directory, or you can visit the [online documentation](https://CPU-DS.github.io/CourseGraph/) (As the project features are still under rapid development, the online documentation is not yet ready). If you wish to customize the online documentation, follow these steps:

#### Install Dependencies and Preview

The documentation is built with [VitePress](https://vitepress.dev/), requiring [Node.js](https://nodejs.org/) 18 or above. Execute:

```bash
cd docs
npm i
npm run dev
```

Open [http://localhost:5173/](http://localhost:5173/) in your browser to preview.

## 🛠️ Contributing, Protocol and Citation

[PR](https://github.com/CPU-DS/CourseGraph/pulls) and [Issues](https://github.com/CPU-DS/CourseGraph/issues) are welcome, as well as any form of contribution.

This project is open-sourced under the [MIT license](LICENSE). 

If you find CourseGraph helpful for your work, please click the `Cite this repository` button on the right side of the Repository to cite.