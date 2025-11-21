# Store Policy & SOP Agent

A production-ready RAG (Retrieval-Augmented Generation) agent built with LangGraph, GCP Vertex AI RAG Corpus, and the A2A (Agent-to-Agent) protocol. This agent provides intelligent access to store policies and standard operating procedures using semantic retrieval from a GCP-managed knowledge base.

## 🌟 Features

- ✅ **GCP Vertex AI RAG Integration** - Official `vertexai.preview.rag` API for semantic search
- ✅ **A2A Protocol Compliant** - Full support for agent-to-agent communication
- ✅ **Configurable RAG Pipeline** - JSON-based configuration for prompts, models, and retrieval settings
- ✅ **Multi-Session Support** - PostgreSQL-backed conversation persistence
- ✅ **Production Ready** - Type hints, error handling, logging, and comprehensive tests
- ✅ **Flexible Deployment** - Standalone or as part of multi-agent orchestration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│          User Query (A2A Protocol)          │
└────────────────┬────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  RAG Agent    │
         │  Executor     │
         └───────┬───────┘
                 │
      ┌──────────┼──────────┐
      │          │          │
      ▼          ▼          ▼
┌──────────┐ ┌──────┐ ┌─────────┐
│ Generate │ │Vertex│ │Generate │
│  Search  │ │  AI  │ │Response │
│  Query   │ │ RAG  │ │  (LLM)  │
│  (LLM)   │ │Corpus│ │         │
└──────────┘ └──────┘ └─────────┘
                 │
                 ▼
        ┌────────────────┐
        │   PostgreSQL   │
        │  Checkpointer  │
        └────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- GCP Project with Vertex AI enabled
- OpenAI API key
- PostgreSQL (optional, for conversation persistence)

### Installation

```bash
# Clone the repository
cd sop-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Create `.env` file**:
```bash
cp .env.example .env
```

2. **Configure environment variables**:
```bash
# OpenAI
OPENAI_API_KEY=sk-your-openai-api-key

# GCP Discovery Engine
GCP_PROJECT_ID=your-gcp-project-id
GCP_LOCATION=us
GCP_DATA_STORE_ID=your-sop-datastore-id
GCP_SERVICE_ACCOUNT_JSON=path/to/service-account.json  # Optional

# Agent Settings
AGENT_NAME=Store Policy Assistant
AGENT_URL=http://localhost:9998

# PostgreSQL (optional)
POSTGRES_CONNECTION_STRING=postgresql://user:pass@localhost:5432/db
```

3. **Configure Discovery Engine settings** in `config/sop_config.json`:
```json
{
  "system_prompt": "Your system prompt...",
  "retrieval_prompt": "Extract search terms: {conversation}",
  "rag_prompt": "Answer using: {conversation}\n{context}",
  "vector_store": {
    "data_store_id": "your-datastore-id",
    "corpus_name": null,
    "top_k": 7,
    "score_threshold": 0.6
  },
  "llm_config": {
    "retrieval_model": "gpt-4o-mini",
    "response_model": "gpt-4o",
    "retrieval_temperature": 0.2,
    "response_temperature": 0.7
  }
}
```

### Running the Agent

```bash
# Start the agent
python -m src.agent

# In another terminal, test it
python test_a2a_client.py

# With debug mode
python test_a2a_client.py --debug
```

## 📋 Usage

### Standalone Agent

```python
from src.agent.executor.sop_agent_executor import SOPAgentExecutor

# Use default config
agent = SOPAgentExecutor()

# Use custom config file
agent = SOPAgentExecutor(config_path="custom_config.json")

# Use config dictionary
config = {
    "vector_store": {"corpus_name": "my-corpus"},
    "llm_config": {"response_model": "gpt-4o"}
}
agent = SOPAgentExecutor(config_dict=config)

# Update config at runtime
agent.update_config({"vector_store": {"top_k": 10}})
```

### A2A Protocol Integration

The agent exposes an A2A-compliant endpoint:

**Agent Card**: `http://localhost:9998/.well-known/agent-card.json`

**Message Endpoint**: `POST http://localhost:9998/`

```json
{
  "jsonrpc": "2.0",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"type": "text", "text": "What is the return policy?"}],
      "messageId": "unique-id"
    },
    "metadata": {"tenant_id": "default", "thread_id": "employee_001"}
  }
}
```

## 🧪 Testing

### Run All Tests

```bash
python run_tests.py
```

### Run Specific Tests

```bash
# Unit tests
pytest src/tests/test_gcp_vector_search.py -v

# Integration tests
pytest src/tests/test_rag_agent_integration.py -v

# With coverage
pip install pytest-cov
pytest src/tests/ --cov=src/agent --cov-report=html
```

### Test Agent Responses

```bash
# Basic test
python test_a2a_client.py

# Debug mode (shows raw responses)
python test_a2a_client.py --debug
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t sop-agent .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f agent

# Stop services
docker-compose down
```

### Environment Variables

Set these in `.env` file or pass to docker-compose:

- `OPENAI_API_KEY` - Your OpenAI API key
- `GCP_PROJECT_ID` - GCP project ID
- `GCP_DATA_STORE_ID` - Discovery Engine datastore ID
- `GCP_LOCATION` - GCP location (default: "us")
- `GCP_SERVICE_ACCOUNT_JSON` - Path to service account JSON (optional)

## 📁 Project Structure

```
sop-agent/
├── src/
│   ├── agent/
│   │   ├── __main__.py              # Entry point
│   │   ├── config/
│   │   │   └── settings.py          # Settings management
│   │   ├── executor/
│   │   │   └── sop_agent_executor.py # RAG logic
│   │   ├── integrations/
│   │   │   └── gcp_vector_search.py  # GCP integration
│   │   └── utils/
│   │       └── llm_provider.py       # LLM utilities
│   └── tests/
│       ├── test_gcp_vector_search.py      # Unit tests
│       └── test_rag_agent_integration.py  # Integration tests
├── config/
│   └── sop_config.json                    # Default config
├── test_a2a_client.py                     # Test client
├── run_tests.py                           # Test runner
├── requirements.txt                       # Dependencies
├── Dockerfile                             # Docker build
├── docker-compose.yml                     # Docker orchestration
├── .env.example                           # Environment template
└── README.md                              # This file
```

## 🔧 Configuration

### Agent Configuration

The agent supports multiple configuration methods:

1. **JSON File** (default: `config/sop_config.json`)
2. **Dictionary** (pass directly to constructor)
3. **API-based** (fetch from API and pass as dict)
4. **Runtime updates** (via `update_config()` method)

### Configuration Schema

```json
{
  "system_prompt": "System-level instructions",
  "retrieval_prompt": "Template for search query generation: {conversation}",
  "rag_prompt": "Template for response: {conversation} {context}",
  "vector_store": {
    "corpus_name": "gcp-corpus-id",
    "top_k": 5,
    "score_threshold": 0.7
  },
  "llm_config": {
    "retrieval_model": "gpt-4o-mini",
    "retrieval_temperature": 0.2,
    "response_model": "gpt-4o",
    "response_temperature": 0.7,
    "max_tokens": 5000
  }
}
```

## 🔒 Security

- ✅ Service account authentication for GCP
- ✅ Environment variables for secrets
- ✅ No hardcoded credentials
- ✅ Secure PostgreSQL connections
- ✅ Input validation and sanitization

## 🐛 Troubleshooting

### Common Issues

**Issue**: `Invalid datastore ID`
**Solution**: Verify `GCP_DATA_STORE_ID` is correct

**Issue**: Empty responses
**Solution**: Check datastore has documents, verify datastore ID and location

**Issue**: GCP authentication fails
**Solution**: Set `GCP_SERVICE_ACCOUNT_JSON` or run `gcloud auth application-default login`

**Issue**: No results from corpus
**Solution**: Verify corpus ID, check if documents are ingested, lower `score_threshold`

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python -m src.agent

# Test with debug output
python test_a2a_client.py --debug
```

## 📚 Documentation

- **GCP Vertex AI RAG**: https://cloud.google.com/vertex-ai/docs/generative-ai/rag-overview
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **A2A Protocol**: https://a2a.to/

## 📄 License

[Your License Here]

## 🙋 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review test cases for examples
