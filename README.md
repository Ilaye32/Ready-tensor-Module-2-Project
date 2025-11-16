🚀 Installation
Prerequisites
Python 3.8+

Groq API account and API key

Basic knowledge of LangChain and LangGraph

Setup
Clone the repository
----
bash
git clone https://github.com/Ilaye32/Ready-tensor-Module-2-Project.git
cd Ready-tensor-Module-2-Project
Install dependencies
----
bash
pip install -r requirements.txt
Configure environment

bash
cp .env.example .env
# Add your Groq API key to the .env file
🛠️ Usage
Running the Chatbot
bash
python readytensor_chatbot.py
Example Queries
Course Content: "What is RAG and how is it covered in Module 1?"

Technical: "Show me a LangGraph example for multi-agent systems"

Enrollment: "How do I enroll in the certification program?"

Projects: "What are the project requirements for Module 2?"

Testing
Run the test suite to verify all agent nodes:

bash
python test_chatbot.py
🔧 Available Tools
1. Web Search Tool
Real-time information retrieval using DuckDuckGo

Current updates about Ready Tensor courses and AI technologies

2. Document Retrieval Tool
Access to comprehensive course documentation

Topics: RAG, LangGraph, vector databases, security, deployment, testing

3. Code Executor Tool
Safe Python code execution with security restrictions

Educational demonstrations of LangChain/LangGraph concepts

🤖 Agent Specializations
Agent	Purpose	Tools Used
Router	Query classification	None
Course Content	Curriculum questions	Web Search, Document Retrieval
Enrollment	Program logistics	Web Search
Technical	Coding & frameworks	Code Executor, Document Retrieval, Web Search
Projects	Certification requirements	Web Search
Supervisor	Quality control	None
⚙️ Configuration
Create a .env file with your API keys:

env
GROQ_API_KEY=your_groq_api_key_here
📋 Dependencies
Key packages included in requirements.txt:

langchain & langgraph - Agent orchestration

groq - LLM integration with Llama 3.1

duckduckgo-search - Web search capabilities

pytest - Testing framework

🧪 Testing
The test_chatbot.py file contains comprehensive tests for:

Individual agent node functionality

Tool integration and responses

Multi-agent workflow coordination

Error handling and edge cases

Run tests with:

bash
python test_chatbot.py
🎓 Course Integration
This chatbot demonstrates key concepts from the Ready Tensor Agentic AI Certification:

Module 1: RAG systems and vector databases

Module 2: Multi-agent systems with LangGraph

Module 3: Production readiness and testing

🔒 Security Features
Restricted code execution environment

Input validation and sanitization

No file system or network access in code execution

Content filtering for safe responses

🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Built as part of Ready Tensor's Agentic AI Developer Certification

Powered by Groq's Llama 3.1-8b-instant model

Utilizes LangChain and LangGraph for agent orchestration
