# RAG 2.0 - Advanced Retrieval-Augmented Generation

A production-ready RAG (Retrieval-Augmented Generation) system built with LangGraph and LangChain, featuring document ingestion, vector storage with ChromaDB, and intelligent query processing.

## Features

- **Document Ingestion**: Support for PDF and text files with automatic chunking
- **Vector Storage**: ChromaDB integration for efficient similarity search
- **LangGraph Workflow**: Modular RAG pipeline with clear state management
- **OpenAI Integration**: GPT models for embeddings and text generation
- **Production Ready**: Environment configuration, error handling, and logging
- **LangGraph Studio**: Compatible with LangGraph Studio for visual workflow editing

## Project Structure

```
RAG-2.0/
├── src/
│   └── rag_2_0/                 # Main package
│       ├── __init__.py
│       ├── agents/              # RAG agents and workflows
│       │   ├── __init__.py
│       │   └── rag_agent.py
│       ├── ingestion/           # Document processing
│       │   ├── __init__.py
│       │   └── document_ingester.py
│       └── utils/               # Utility functions
│           └── __init__.py
├── scripts/                     # Command-line scripts
│   ├── __init__.py
│   └── main.py                  # Application entry point
├── tests/                       # Test suite
│   ├── __init__.py
│   └── test_rag_agent.py
├── configs/                     # Configuration files
│   └── .env.example             # Environment variables template
├── data/                        # Document storage directory
├── chroma_db/                   # Vector database (auto-generated)
├── pyproject.toml               # Project metadata and dependencies
├── langgraph.json               # LangGraph Studio configuration
├── .gitignore                   # Git ignore patterns
└── README.md                    # This file
```

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key
- UV package manager (recommended) or pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/LoopFireAI/RAG-2.0.git
cd RAG-2.0
```

2. Install dependencies:
```bash
uv sync
# or with pip:
# pip install -e .
```

3. Set up environment variables:
```bash
cp configs/.env.example configs/.env
# Edit configs/.env with your OpenAI API key and other settings
```

### Configuration

Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.1
MAX_TOKENS=1000

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=3
```

### Usage

1. **Ingest Documents**: Place PDF or text files in the `data/` directory and run:
```bash
python -m rag_2_0.ingestion.document_ingester
# or
uv run ingest
```

2. **Run the RAG Agent**:
```bash
python scripts/main.py
# or
uv run rag
```

3. **Test the System**:
```bash
python -m pytest tests/
# or
python tests/test_rag_agent.py
```

4. **LangGraph Studio**: Open the project in LangGraph Studio for visual workflow editing:
```bash
langgraph dev
```

## API Reference

### RAG Agent

The main RAG workflow consists of three nodes:

1. **Extract Query**: Processes input messages to extract the user query
2. **Retrieve Documents**: Performs similarity search against the vector database
3. **Generate Response**: Uses retrieved context to generate relevant answers

### Document Ingester

The `DocumentIngester` class handles:
- Loading documents from the data directory
- Splitting documents into optimally-sized chunks
- Creating embeddings and storing in ChromaDB
- Managing vector store collections

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `MODEL_NAME` | OpenAI model to use | `gpt-3.5-turbo` |
| `TEMPERATURE` | Model temperature | `0.1` |
| `MAX_TOKENS` | Maximum response tokens | `1000` |
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` |
| `TOP_K` | Number of documents to retrieve | `3` |

## Development

### Project Setup

1. Install development dependencies:
```bash
uv sync --dev
```

2. Run tests:
```bash
python -m pytest tests/
```

3. Code formatting and linting:
```bash
uv run black src/ scripts/ tests/
uv run ruff check src/ scripts/ tests/
```

### Package Development

The project uses a modern Python packaging structure:

- **src/rag_2_0/**: Main package code
- **scripts/**: Executable scripts
- **tests/**: Test suite
- **configs/**: Configuration files

Import the package components:
```python
from rag_2_0.agents import create_rag_graph
from rag_2_0.ingestion import DocumentIngester
```

### Adding New Document Types

To support additional document formats, extend the `load_documents` method in `src/rag_2_0/ingestion/document_ingester.py`:

```python
elif file_path.suffix.lower() == '.docx':
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(str(file_path))
    docs = loader.load()
```

## Deployment

This project is designed for easy deployment to various platforms:

- **Local**: Run directly with Python
- **Docker**: Create a Dockerfile for containerized deployment
- **Cloud**: Deploy to AWS, GCP, or Azure with minimal configuration
- **LangGraph Cloud**: Deploy workflows to LangGraph Cloud

### Docker Deployment

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
CMD ["uv", "run", "rag"]
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Troubleshooting

### Common Issues

1. **ChromaDB Persistence**: If you encounter database issues, delete the `chroma_db/` directory and re-run ingestion
2. **OpenAI API Limits**: Monitor your API usage and adjust `MAX_TOKENS` if needed
3. **Memory Issues**: Reduce `CHUNK_SIZE` for large documents or limited memory environments
4. **Import Errors**: Ensure you're running commands from the project root directory

### Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review the test files for usage examples

---

Built with ❤️ by LoopFire AI