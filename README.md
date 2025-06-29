# Wells RAG 2.0 - Dual Mode System

A sophisticated RAG (Retrieval Augmented Generation) system with persona-driven responses and comprehensive feedback analytics.

## 🚀 Quick Start

### For Development & Testing (LangGraph Studio)
```bash
./development/start-dev.sh
```
Then open: http://localhost:8123

### For Production (Slack Bot)
```bash
./production/start-slack-bot.sh
```
API available at: http://localhost:8000

## 📋 Two Deployment Modes

### 🔧 Development Mode
- **Purpose**: Local development and testing
- **Interface**: LangGraph Studio UI
- **Port**: 8123
- **Features**: 
  - Visual workflow testing
  - Interactive debugging
  - Live code reloading
  - Direct RAG system access

**Start Development:**
```bash
./development/start-dev.sh
# or
docker-compose -f development/docker-compose.dev.yml up --build
```

### 🤖 Production Mode (Slack Bot)
- **Purpose**: 24/7 Slack integration
- **Interface**: FastAPI REST endpoints
- **Port**: 8000
- **Features**:
  - Slack slash commands (`/wells`)
  - App mentions (@bot)
  - Signature verification
  - Production logging

**Start Slack Bot:**
```bash
./production/start-slack-bot.sh
# or  
docker-compose -f production/docker-compose.prod.yml up --build
```

## 🏗️ Architecture

### Core Components
- **RAG Agent**: Multi-persona response generation (Janelle, Doreen, Default)
- **Vector Database**: ChromaDB with 1,000+ documents
- **Feedback System**: User satisfaction tracking & analytics
- **Document Ingestion**: Google Drive integration

### Persona System
- **Janelle**: Strategic business perspective with scaling mindset
- **Doreen**: Relational, equity-focused approach  
- **Default**: Professional, research-backed tone

## 📊 Management Tools

### CLI Commands
```bash
# Query the system directly
python cli/main.py "What is effective leadership?"

# Interactive mode
python cli/main.py

# View feedback analytics
python cli/feedback_admin.py stats

# KPI Dashboard
python cli/kpi_dashboard.py
```

## 🔧 Configuration

### Environment Files
- `.env` - Local development configuration
- `.env.production` - Production configuration  
- `.env.template` - Template for new installations

### Required Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-3.5-turbo

# Vector Database
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=3

# Slack Bot (Production only)
SLACK_SIGNING_SECRET=your_slack_signing_secret
```

## 🚀 Deployment Options

### Local Development
Use the development mode for testing and iteration.

### Production Deployment
Choose from these simple deployment platforms:

#### Railway (Recommended)
1. Connect your GitHub repo
2. Use `production/Dockerfile.slack`
3. Set environment variables
4. Deploy automatically

#### Render
1. Connect GitHub repo  
2. Select `production/Dockerfile.slack`
3. Configure environment
4. Deploy

#### Digital Ocean App Platform
1. Import from GitHub
2. Select `production/Dockerfile.slack`
3. Set environment variables
4. Deploy

## 📈 Monitoring

### Feedback Analytics
- User satisfaction tracking
- Document relevance scoring
- Query pattern analysis
- Performance metrics

### Health Checks
- **Development**: http://localhost:8123/health
- **Production**: http://localhost:8000/health

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Test individual components
python tests/test_rag_system.py

# Test vector store
python tests/test_vector_store.py
```

## 📝 Usage Examples

### Slack Commands
```
/wells What is effective leadership?
/wells How can I improve team communication?
/wells Create a LinkedIn post about innovation
```

### Direct API Testing
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is leadership?", "user_id": "test_user"}'
```

## 🔒 Security Features

- Slack signature verification
- Environment-based configuration
- Secure API key management
- Request validation

## 📚 Documentation

- **Feedback System**: Comprehensive user satisfaction tracking
- **KPI Monitoring**: Performance analytics and alerting
- **Multi-Persona**: Dynamic response tone adaptation
- **Document Ingestion**: Automated Google Drive sync

## 🛠️ Development

### Adding New Features
1. Develop in LangGraph Studio (development mode)
2. Test with CLI tools
3. Deploy Slack bot (production mode)

### Database Management
- ChromaDB: Vector embeddings storage
- SQLite: Feedback and analytics data
- Document deduplication via hashing

---

**Perfect for small teams needing 24/7 knowledge access with sophisticated AI responses!** 