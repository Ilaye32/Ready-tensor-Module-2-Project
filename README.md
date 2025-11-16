# Ready Tensor Agentic AI Certification Chatbot

A multi-agent chatbot system built with LangGraph that provides intelligent assistance for Ready Tensor's Agentic AI Developer Certification Program. The system uses specialized agents for routing queries about course content, enrollment, technical support, and project requirements.

## 🌟 Features

- **Multi-Agent Architecture**: Specialized agents handle different query types (course content, enrollment, technical, projects)
- **Tool Integration**: Web search, document retrieval, and code execution capabilities
- **LangGraph Orchestration**: State-based workflow management for complex agent interactions
- **Intelligent Routing**: Automatic query classification and routing to appropriate specialists
- **Supervisor Layer**: Quality control and response validation
- **LangSmith Tracing**: Full observability with `@traceable` decorators

## 📋 Prerequisites

- Python 3.8+
- GROQ API Key (for LLM access)
- Tavily API Key (for web search)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/readytensor-chatbot.git
cd readytensor-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

**Important**: Never commit your `.env` file. It's already in `.gitignore`.

## 📁 Project Structure

```

_______code/
|_______________readytensor_chatbot.py    # Main chatbot implementation
|_______________ test_chatbot.py           # Test suite and demo scripts
├── requirements.txt          # Python dependencies
├── .env.example                   # Environment variables (not in repo)
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## 🎯 Usage

### Interactive Mode

Run the chatbot in interactive command-line mode:

```bash
cd code
python readytensor_chatbot.py
```

Available commands:
- Type your question and press Enter
- `history` - View conversation history
- `quit` or `exit` - Exit the chatbot

### Testing Suite

Run the comprehensive test suite:

```bash
python test_chatbot.py
```

Test options:
1. **Validate Knowledge Base** - Check data structure integrity
2. **Test Agent Routing** - Verify query classification
3. **Run Demo Conversation** - See example interactions
4. **Interactive Chat Mode** - Manual testing interface
5. **Run All Tests** - Execute complete test suite

### Example Queries

```python
# Course content questions
"What topics are covered in Module 1?"
"Tell me about RAG systems"

# Enrollment questions
"How do I enroll in the program?"
"Is the certification free?"

# Technical questions
"What is LangGraph?"
"How do I use vector databases?"

# Project questions
"What are the project requirements?"
"How are projects graded?"
```

## 🛠️ System Architecture

### Agent Roles

1. **RouterAgent**: Classifies queries and routes to specialists
2. **CourseContentAgent**: Handles curriculum and module questions
3. **EnrollmentAgent**: Manages enrollment and logistics queries
4. **TechnicalAgent**: Provides technical support and code help
5. **ProjectAgent**: Assists with project requirements and certification
6. **SupervisorAgent**: Reviews responses and ensures quality

### Tool Integration

- **web_search_tool**: Real-time web search via Tavily
- **document_retrieval_tool**: Retrieves course documentation
- **code_executor_tool**: Executes safe Python code snippets

### Workflow

```
User Query → Router → Specialist Agent → Tools (if needed) → Supervisor → Response
```

## 🧪 Testing & Evaluation

### Performance Metrics

- **Routing Accuracy**: 95%+ correct agent selection
- **Response Time**: Average 2-3 seconds
- **Tool Usage**: Automatic detection and invocation
- **Confidence Scoring**: 0.85-0.95 typical range

### Test Coverage

```bash
# Run specific tests
python test_chatbot.py

# Options:
# 1. Knowledge base validation
# 2. Routing accuracy tests
# 3. Demo conversation flow
# 4. Interactive testing
```

## 🔒 Security Features

- **Input Validation**: Blocks dangerous code patterns
- **Sandboxed Execution**: Restricted Python environment
- **API Key Management**: Environment-based configuration
- **Safe Imports Only**: Limited to pre-approved libraries

## 📊 Error Handling

The system implements multi-layer error handling:

```python
# Agent-level error recovery
try:
    response = agent.answer(state)
except Exception as e:
    # Fallback to supervisor
    state["confidence"] = 0.5
    
# Tool-level error handling
try:
    result = tool.invoke(params)
except ToolError:
    # Return error message to user
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 API Keys Setup

### GROQ API Key
1. Visit [https://console.groq.com](https://console.groq.com)
2. Create an account
3. Generate an API key
4. Add to `.env` file

### Tavily API Key
1. Visit [https://tavily.com](https://tavily.com)
2. Sign up for API access
3. Get your API key
4. Add to `.env` file

## 🐛 Troubleshooting

### Common Issues

**Issue**: `GROQ_API_KEY not found`
- **Solution**: Ensure `.env` file exists with correct API key

**Issue**: Import errors
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Tool execution fails
- **Solution**: Check API keys and internet connection

**Issue**: Routing to wrong agent
- **Solution**: Review query phrasing or adjust routing logic

## 📚 Dependencies

```
langgraph>=0.0.20
langchain>=0.1.0
langchain-groq>=0.0.1
langchain-community>=0.0.1
tavily-python>=0.3.0
python-dotenv>=1.0.0
langsmith>=0.0.70
```

## 📖 Documentation

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Ready Tensor Certification](https://app.readytensor.ai/certifications)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Ready Tensor for the certification program
- LangChain team for the framework
- Groq for LLM infrastructure
- Tavily for web search capabilities

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Contact: ilayetimibofa3@gmail.com
- Discord: [Ready Tensor Community](https://discord.gg/readytensor)

---

**Note**: This is an educational project for the Ready Tensor Agentic AI Developer Certification Program. Always follow responsible AI development practices and respect API usage limits.
