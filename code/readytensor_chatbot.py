from langsmith import traceable
import os
from typing import TypedDict, Annotated, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
import operator
import json
from tavily import TavilyClient
import re
from datetime import datetime


# ============================================================================
# TOOLS DEFINITION - FIXED
# ============================================================================

@traceable
@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for real-time information and summarize using an LLM.
    
    Args:
        query: The search query string
        
    Returns:
        Summarized search results
    """
    try:
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Step 1: Perform web search
        search_results = tavily.search(query=query, max_results=5)
        
        # Combine summaries into single context
        context = "\n\n".join(
            item["content"] for item in search_results.get("results", [])
        )
        
        if not context:
            return f"No search results found for: {query}"
        
        # Use Groq to summarize
        client = ChatGroq(
            api_key=os.env("GROQ_API_KEY"),
            model="llama-3.1-8b-instant"
        )
        
        prompt = f"""Based on these web search results, provide a concise and accurate answer.

Search results:
{context}

Question: {query}

Provide a clear, factual answer:"""
        
        response = client.invoke([
            SystemMessage(content="You are a web research assistant. Provide accurate, concise summaries."),
            HumanMessage(content=prompt)
        ])
        
        return response.content
        
    except Exception as e:
        return f"Web search error: {str(e)}"


@traceable
@tool
def document_retrieval_tool(topic: str, module: str = "all") -> str:
    """
    Retrieve specific documentation about course topics, modules, or concepts.
    
    Args:
        topic: The topic to search for (e.g., "RAG", "LangGraph", "vector databases")
        module: Specific module number (1, 2, 3, 4) or "all" for all modules
        
    Returns:
        Relevant documentation snippets
    """
    # Simulated document store
    documents = {
        "rag": {
            "module": "1",
            "content": """
RAG (Retrieval-Augmented Generation) in Module 1:

- Build systems that combine LLM generation with external knowledge retrieval
- Use vector databases (Qdrant, FAISS) to store and retrieve relevant documents
- Implement semantic search using embeddings
- Create context-aware responses by augmenting prompts with retrieved information
- Learn chunking strategies for optimal retrieval

Project: Build a LangGraph-powered assistant that answers questions using 
real documentation with ReAct-based reasoning.
            """
        },
        "langgraph": {
            "module": "2",
            "content": """
LangGraph in Module 2 (Multi-Agent Systems):

- Design complex agent workflows with state management
- Create multi-agent systems with coordination patterns
- Implement human-in-the-loop interactions
- Build conditional routing between agents
- Manage agent memory and conversation state
- Use graph-based orchestration for complex tasks

LangGraph enables you to build stateful, multi-step applications with LLMs.
            """
        },
        "vector_databases": {
            "module": "1",
            "content": """
Vector Databases (Qdrant, FAISS) in Module 1:

- Store embeddings for semantic search
- Qdrant: Production-ready vector database with filtering
- FAISS: Facebook's library for efficient similarity search
- Learn embedding strategies and indexing
- Implement hybrid search (keyword + semantic)
- Optimize for retrieval speed and accuracy
            """
        },
        "security": {
            "module": "3",
            "content": """
Security & Guardrails in Module 3:

OWASP Top 10 for LLM Applications:
1. Prompt Injection
2. Insecure Output Handling
3. Training Data Poisoning
4. Model Denial of Service
5. Supply Chain Vulnerabilities
6. Sensitive Information Disclosure
7. Insecure Plugin Design
8. Excessive Agency
9. Overreliance
10. Model Theft

- Implement input validation and sanitization
- Add content filtering and safety layers
- Monitor for adversarial attacks
            """
        },
        "deployment": {
            "module": "3",
            "content": """
Deployment Strategies in Module 3:

- FastAPI for lightweight, production-ready APIs
- Containerization with Docker
- Cloud deployment (AWS, GCP, Azure)
- Monitoring and observability with LangSmith
- Load testing and performance optimization
- CI/CD pipelines for agentic systems
            """
        },
        "testing": {
            "module": "3",
            "content": """
Testing Agentic AI Systems in Module 3:

- Unit testing with pytest
- Integration testing for multi-agent workflows
- Evaluation frameworks (Giskard)
- Testing for safety and alignment
- Performance benchmarking
- Regression testing for LLM outputs
            """
        }
    }
    
    topic_lower = topic.lower().replace(" ", "_")
    
    # Find matching documents
    matches = []
    for key, doc in documents.items():
        if topic_lower in key or key in topic_lower:
            if module == "all" or doc["module"] == str(module):
                matches.append(doc["content"])
    
    if matches:
        return "\n\n---\n\n".join(matches)
    else:
        return f"No specific documentation found for '{topic}'. Try: RAG, LangGraph, vector_databases, security, deployment, testing."


@traceable
@tool
def code_executor_tool(code: str, language: str = "python") -> str:
    """
    Execute simple Python code snippets safely.
    
    Args:
        code: Python code to execute (safe operations only)
        language: Programming language (currently only "python" supported)
        
    Returns:
        Execution result or explanation
    """
    if language.lower() != "python":
        return f"Currently only Python execution is supported. You requested: {language}"
    
    # Security check
    dangerous_patterns = [
        r'\bimport\s+os\b', r'\bimport\s+sys\b', r'\bimport\s+subprocess\b',
        r'\bopen\s*\(', r'\bexec\s*\(', r'\beval\s*\(',
        r'\b__import__\b', r'\bcompile\s*\(',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return "⚠️ Security Error: Code contains potentially unsafe operations."
    
    # Check for LangChain imports
    if 'langchain' in code.lower() or 'langgraph' in code.lower():
        return "💡 I can't execute LangChain code directly, but I can explain what it does!"
    
    # Execute safe code
    try:
        from io import StringIO
        import sys
        
        # Restricted globals
        safe_globals = {
            '__builtins__': {
                'print': print, 'len': len, 'range': range, 'str': str,
                'int': int, 'float': float, 'list': list, 'dict': dict,
                'set': set, 'tuple': tuple, 'bool': bool, 'sum': sum,
                'max': max, 'min': min, 'abs': abs, 'round': round,
                'sorted': sorted, 'enumerate': enumerate, 'zip': zip,
            }
        }
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        exec(code, safe_globals)
        
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        return f"✅ Code executed:\n\n{output}" if output else "✅ Code executed (no output)."
            
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# KNOWLEDGE BASE
# ============================================================================

COURSE_KNOWLEDGE = {
    "program_overview": {
        "name": "Agentic AI Developer Certification Program",
        "duration": "12 weeks",
        "cost": "Free",
        "provider": "Ready Tensor",
        "structure": "3 modules + 1 optional advanced module",
        "certification": "Complete all 3 projects to earn full certification"
    },
    "modules": {
        "module_1": {
            "name": "Foundations of Agentic AI",
            "weeks": "1-4",
            "topics": ["Core concepts", "LangChain", "Prompt engineering", "RAG systems", "Vector databases"],
            "project": "LangGraph-powered assistant with ReAct reasoning"
        },
        "module_2": {
            "name": "Multi-Agent Systems",
            "weeks": "5-8",
            "topics": ["Agent design", "Tool integration", "Multi-agent coordination", "LangGraph workflows"],
            "project": "Multi-agent research assistant with FastAPI"
        },
        "module_3": {
            "name": "Real-World Readiness",
            "weeks": "9-12",
            "topics": ["Testing", "Security (OWASP Top 10)", "Deployment", "Production best practices"],
            "project": "Production-ready application with testing suite"
        }
    },
    "enrollment": {
        "process": "Visit certifications page → Select program → Click 'Enroll for Free'",
        "flexibility": "Self-paced, start anytime",
        "url": "https://app.readytensor.ai/certifications/agentic-ai-cert-U7HxeL7a"
    },
    "tools_frameworks": {
        "primary": ["LangChain", "LangGraph", "Python"],
        "vector_dbs": ["Qdrant", "FAISS"],
        "deployment": ["FastAPI"],
        "testing": ["pytest", "Giskard"]
    }
}


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    query: str
    route: str
    context: str
    response: str
    confidence: float
    next_agent: str
    tool_results: List[str]


# ============================================================================
# AGENT DEFINITIONS - FIXED TOOL USAGE
# ============================================================================

@traceable
class RouterAgent:
    """Routes user queries to appropriate specialist agents"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def route(self, state: AgentState) -> AgentState:
        query = state["query"]
        
        system_prompt = """You are a routing agent. Analyze the query and respond with ONE word:
- course_content: Questions about lessons, modules, curriculum
- enrollment: Questions about signing up, costs, how to join  
- technical: Questions about tools, coding, LangChain, technical issues
- projects: Questions about project submissions, certification

Respond with ONLY ONE of these words."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Route this: {query}")
        ]
        
        response = self.llm.invoke(messages)
        route = response.content.strip().lower()
        
        valid_routes = ["course_content", "enrollment", "technical", "projects"]
        if route not in valid_routes:
            route = "course_content"
        
        state["route"] = route
        state["next_agent"] = route
        state["messages"].append(AIMessage(content=f"[Routing to {route}]"))
        
        return state


class CourseContentAgent:
    """Handles course content questions - FIXED TOOL CALLS"""
    
    def __init__(self, llm):
        self.llm = llm
        self.knowledge = COURSE_KNOWLEDGE
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        tool_results = []
        
        # Check if we need document retrieval
        doc_keywords = ["rag", "langgraph", "vector", "security", "deployment", "testing"]
        needs_docs = any(kw in query.lower() for kw in doc_keywords)
        
        if needs_docs:
            topic = next((kw for kw in doc_keywords if kw in query.lower()), "general")
            try:
                # FIXED: Call tool directly as a function
                doc_result = document_retrieval_tool(topic=topic)
                tool_results.append(f"📚 Documentation:\n{doc_result}")
            except Exception as e:
                tool_results.append(f"📚 Doc retrieval error: {str(e)}")
        
        # Check if we need web search
        if any(word in query.lower() for word in ["latest", "recent", "current", "2025"]):
            try:
                # FIXED: Call tool directly
                search_result = web_search_tool(query=f"Ready Tensor Agentic AI {query}")
                tool_results.append(f"🔍 Web Search:\n{search_result}")
            except Exception as e:
                tool_results.append(f"🔍 Search error: {str(e)}")
        
        # Build context
        context = self._build_context()
        if tool_results:
            context += "\n\nTOOL RESULTS:\n" + "\n\n".join(tool_results)
        
        system_prompt = f"""You are the Course Content Specialist for Ready Tensor's Agentic AI Certification.

KNOWLEDGE BASE:
{context}

Rules:
1. Only answer questions about AI and this certification program
2. Use the provided knowledge base and tool results
3. Be clear and specific

If asked about unrelated topics, politely redirect to the certification program."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        
        state["context"] = "course_content"
        state["response"] = response.content
        state["confidence"] = 0.9
        state["tool_results"] = tool_results
        state["messages"].append(AIMessage(content=response.content))
        state["next_agent"] = "supervisor"
        
        return state
    
    def _build_context(self) -> str:
        kb = self.knowledge
        context = [
            f"Program: {kb['program_overview']['name']}",
            f"Duration: {kb['program_overview']['duration']}",
            f"Cost: {kb['program_overview']['cost']}",
            "\nMODULES:"
        ]
        
        for mod in kb['modules'].values():
            context.append(f"\n{mod['name']} ({mod['weeks']})")
            context.append(f"Topics: {', '.join(mod['topics'][:3])}")
        
        return "\n".join(context)


class EnrollmentAgent:
    """Handles enrollment questions"""
    
    def __init__(self, llm):
        self.llm = llm
        self.knowledge = COURSE_KNOWLEDGE
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        tool_results = []
        
        # Check for current status questions
        if any(word in query.lower() for word in ["when", "deadline", "available", "open"]):
            try:
                # FIXED: Call tool directly
                search_result = web_search_tool(
                    query="Ready Tensor Agentic AI certification enrollment status 2025"
                )
                tool_results.append(f"🔍 Current Status:\n{search_result}")
            except Exception as e:
                tool_results.append(f"Search error: {str(e)}")
        
        enrollment = self.knowledge["enrollment"]
        program = self.knowledge["program_overview"]
        
        context = f"""
ENROLLMENT INFO:
- Cost: {program['cost']} (completely free!)
- Duration: {program['duration']}
- Enrollment: {enrollment['process']}
- Access: {enrollment['flexibility']}
- URL: {enrollment['url']}
"""
        
        if tool_results:
            context += "\n\n" + "\n".join(tool_results)
        
        system_prompt = f"""You are the Enrollment Specialist.
{context}

Be encouraging and helpful. Emphasize it's free and self-paced."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        
        state["context"] = "enrollment"
        state["response"] = response.content
        state["confidence"] = 0.95
        state["tool_results"] = tool_results
        state["messages"].append(AIMessage(content=response.content))
        state["next_agent"] = "supervisor"
        
        return state


class TechnicalAgent:
    """Handles technical questions - FIXED TOOL CALLS"""
    
    def __init__(self, llm):
        self.llm = llm
        self.knowledge = COURSE_KNOWLEDGE
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        tool_results = []
        
        # Check for code execution request
        code_pattern = r'```python(.*?)```'
        code_match = re.search(code_pattern, query, re.DOTALL)
        
        if code_match:
            code = code_match.group(1).strip()
            try:
                # FIXED: Call tool directly
                exec_result = code_executor_tool(code=code)
                tool_results.append(f"⚙️ Code Execution:\n{exec_result}")
            except Exception as e:
                tool_results.append(f"Execution error: {str(e)}")
        
        # Check for documentation needs
        tech_topics = ["langgraph", "langchain", "rag", "vector", "agent"]
        if any(topic in query.lower() for topic in tech_topics):
            topic = next((t for t in tech_topics if t in query.lower()), "langgraph")
            try:
                # FIXED: Call tool directly
                doc_result = document_retrieval_tool(topic=topic)
                tool_results.append(f"📚 Technical Docs:\n{doc_result}")
            except Exception as e:
                tool_results.append(f"Doc error: {str(e)}")
        
        # Check for latest info
        if any(word in query.lower() for word in ["latest", "new", "version", "update"]):
            try:
                # FIXED: Call tool directly
                search_result = web_search_tool(query=f"LangChain LangGraph {query}")
                tool_results.append(f"🔍 Latest Info:\n{search_result}")
            except Exception as e:
                tool_results.append(f"Search error: {str(e)}")
        
        tools = self.knowledge["tools_frameworks"]
        context = f"""
TECHNICAL STACK:
- Primary: {', '.join(tools['primary'])}
- Vector DBs: {', '.join(tools['vector_dbs'])}
- Deployment: {', '.join(tools['deployment'])}
"""
        
        if tool_results:
            context += "\n\nTOOL RESULTS:\n" + "\n\n".join(tool_results)
        
        system_prompt = f"""You are the Technical Support Specialist.
{context}

Provide practical, code-focused guidance."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        
        state["context"] = "technical"
        state["response"] = response.content
        state["confidence"] = 0.85
        state["tool_results"] = tool_results
        state["messages"].append(AIMessage(content=response.content))
        state["next_agent"] = "supervisor"
        
        return state


class ProjectAgent:
    """Handles project and certification questions"""
    
    def __init__(self, llm):
        self.llm = llm
        self.knowledge = COURSE_KNOWLEDGE
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        tool_results = []
        
        # Search for project examples
        if any(word in query.lower() for word in ["example", "sample", "template"]):
            try:
                # FIXED: Call tool directly
                search_result = web_search_tool(
                    query="Ready Tensor Agentic AI certification project examples"
                )
                tool_results.append(f"🔍 Project Examples:\n{search_result}")
            except Exception as e:
                tool_results.append(f"Search error: {str(e)}")
        
        modules = self.knowledge["modules"]
        context = f"""
PROJECT INFO:
- Score 70%+ required on each project
- Monthly reviews by experts
- Can revise and resubmit

Projects:
1. Module 1: {modules['module_1']['project']}
2. Module 2: {modules['module_2']['project']}
3. Module 3: {modules['module_3']['project']}
"""
        
        if tool_results:
            context += "\n\n" + "\n".join(tool_results)
        
        system_prompt = f"""You are the Project & Certification Specialist.
{context}

Be clear about requirements and encouraging."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        
        state["context"] = "projects"
        state["response"] = response.content
        state["confidence"] = 0.92
        state["tool_results"] = tool_results
        state["messages"].append(AIMessage(content=response.content))
        state["next_agent"] = "supervisor"
        
        return state


class SupervisorAgent:
    """Reviews responses"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def supervise(self, state: AgentState) -> AgentState:
        confidence = state["confidence"]
        tool_results = state.get("tool_results", [])
        
        if confidence >= 0.85:
            state["next_agent"] = "end"
            return state
        
        # Enhance low-confidence responses
        response = state["response"]
        system_prompt = """Review and enhance this response. Keep it concise.
Rules:
1. Only provide info about programming/Ready Tensor
2   Do provide information outside the scope of this program
3.  Be concise and helpful
4.  Do not provide information outside your knowledge base or not related to this program
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Review:\n{response}")
        ]
        
        enhanced = self.llm.invoke(messages)
        state["response"] = enhanced.content
        state["messages"].append(AIMessage(content=enhanced.content))
        state["next_agent"] = "end"
        
        return state


# ============================================================================
# SIMPLIFIED CHATBOT (Working Version)
# ============================================================================

class SimpleChatbot:
    """Simple chatbot with working tool integration"""
    
    def __init__(self, api_key: str):
        self.client = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
        self.history = []
    
    def chat(self, user_input: str) -> str:
        """Process user input and return response"""
        
        # Detect if tools are needed
        tools_used = []
        context_parts = []
        
        # Check for web search trigger
        if any(word in user_input.lower() for word in ["search", "latest", "recent", "current"]):
            try:
                result = web_search_tool(query=user_input)
                context_parts.append(f"Web Search Results:\n{result}")
                tools_used.append("web_search")
            except Exception as e:
                context_parts.append(f"Search failed: {str(e)}")
        
        # Check for documentation trigger
        doc_keywords = ["rag", "langgraph", "vector", "what is", "explain", "how does"]
        if any(kw in user_input.lower() for kw in doc_keywords):
            topic = next((kw for kw in ["rag", "langgraph", "vector", "security"] 
                         if kw in user_input.lower()), "langgraph")
            try:
                result = document_retrieval_tool(topic=topic)
                context_parts.append(f"Documentation:\n{result}")
                tools_used.append("document_retrieval")
            except Exception as e:
                context_parts.append(f"Doc retrieval failed: {str(e)}")
        
        # Check for code execution
        code_match = re.search(r'```python(.*?)```', user_input, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            try:
                result = code_executor_tool(code=code)
                context_parts.append(f"Code Execution:\n{result}")
                tools_used.append("code_executor")
            except Exception as e:
                context_parts.append(f"Execution failed: {str(e)}")
        
        # Build prompt with context
        system_msg = """You are a helpful assistant for Ready Tensor's Agentic AI Certification.

Rules:
1. Only answer questions about AI/programming and this certification
2. Use provided tool results when available
3. Be concise and helpful
4. Do not answer any question outside the scope of this program"""
        
        if context_parts:
            system_msg += f"\n\nTool Results:\n" + "\n\n---\n\n".join(context_parts)
        
        # Build message history
        messages = [SystemMessage(content=system_msg)]
        for item in self.history[-3:]:  # Last 3 exchanges
            messages.append(HumanMessage(content=item["query"]))
            messages.append(AIMessage(content=item["response"]))
        messages.append(HumanMessage(content=user_input))
        
        # Get response
        try:
            response_obj = self.client.invoke(messages)
            response = response_obj.content
        except Exception as e:
            response = f"Error: {str(e)}\nEnsure GROQ_API_KEY is set correctly."
        
        # Store history
        self.history.append({
            "query": user_input,
            "response": response,
            "tools_used": tools_used
        })
        
        return response
    
    def get_history(self):
        return self.history


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Get API key
    API_KEY = os.getenv("GROQ_API_KEY")
    
    if not API_KEY or API_KEY == "your-api-key-here":
        print("ERROR: Please set your GROQ_API_KEY environment variable")
        print("export GROQ_API_KEY='your-key-here'")
        exit(1)
    
    # Initialize chatbot
    chatbot = SimpleChatbot(api_key=API_KEY)
    
    print("=" * 70)
    print("🤖 Ready Tensor Agentic AI Certification Chatbot")
    print("=" * 70)
    print("\n🔧 Available Tools:")
    print("  1. 🔍 Web Search - Real-time information")
    print("  2. 📚 Document Retrieval - Course documentation")
    print("  3. ⚙️ Code Executor - Test Python snippets")
    print("=" * 70)
    print("\n💡 Example queries:")
    print("  • What is RAG?")
    print("  • Search for latest LangGraph updates")
    print("  • Execute: print('Hello!')")
    print("  • How do I enroll?")
    print("\nType 'quit' to exit, 'history' to see conversation")
    print("=" * 70)
    print()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye! Good luck with your certification!")
            break
        
        if user_input.lower() == 'history':
            history = chatbot.get_history()
            print("\n--- Conversation History ---")
            for i, item in enumerate(history, 1):
                tools = item.get('tools_used', [])
                tool_info = f" [Tools: {', '.join(tools)}]" if tools else ""
                print(f"\n{i}. Q: {item['query']}{tool_info}")
                print(f"   A: {item['response'][:200]}...")
            print()
            continue
        
        if not user_input:
            continue
        
        try:
            print("\n🤖 Processing...", end="", flush=True)
            import time
            for _ in range(2):
                print(".", end="", flush=True)
                time.sleep(0.2)
            print("\n")
            
            response = chatbot.chat(user_input)
            print(f"Bot: {response}\n")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Make sure your GROQ_API_KEY is set correctly.\n")
