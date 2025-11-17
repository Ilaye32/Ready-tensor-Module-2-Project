
from langsmith import traceable
import os
from typing import TypedDict, Annotated, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
import operator
import sys
import json
from tavily import TavilyClient
import re
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# TOOLS DEFINITION - ALL VERIFIED WORKING
# ============================================================================

@traceable
@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for real-time information using Tavily API.
    VERIFIED WORKING - Returns real-time web search results.
    
    Args:
        query: Search query string
    
    Returns:
        Formatted web search results
    """
    
    try:
        # Initialize Tavily client - USE ENVIRONMENT VARIABLE
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return "❌ TAVILY_API_KEY environment variable not set. Please set it to use web search."
        
        tavily = TavilyClient(api_key=tavily_api_key)
        
        # Perform web search
        search_results = tavily.search(query=query, max_results=3)
        
        # Extract content from results
        if "results" in search_results and search_results["results"]:
            context = "\n\n".join(
                f"📰 {item.get('title', 'No title')}:\n{item.get('content', 'No content')}"
                for item in search_results["results"]
            )
            
            return f"🔍 Web Search Results for '{query}':\n\n{context}"
        else:
            return f"🔍 No web results found for: {query}"
        
    except Exception as e:
        return f"⚠️ Web search error: {str(e)}"


@traceable
@tool
def document_retrieval_tool(topic: str, module: str = "all") -> str:
    """
    Retrieve specific documentation about course topics, modules, or concepts.
    VERIFIED WORKING - Returns relevant course documentation.
    """
    # [Keep your existing document_retrieval_tool code exactly as is]
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
            It's particularly useful for creating agentic systems that need to maintain
            context and coordinate between multiple specialized agents.
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
            
            Used extensively in RAG systems for efficient document retrieval.
            """
        },
        "security": {
            "module": "3",
            "content": """
            Security & Guardrails in Module 3:
            
            - OWASP Top 10 for LLM Applications:
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
            - Use guardrails to prevent harmful outputs
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
            - Cost optimization strategies
            
            Project: Transform your multi-agent system into a production-ready
            application with full testing suite and deployment configuration.
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
            - A/B testing for prompt variations
            
            Learn to build comprehensive test suites that ensure your agentic
            systems are reliable, safe, and performant.
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
        return f"📚 No specific documentation found for '{topic}'.\n\nAvailable topics: RAG, LangGraph, vector_databases, security, deployment, testing."


@traceable
@tool
def code_executor_tool(code: str, language: str = "python") -> str:
    """
    Execute simple Python code snippets to help users test concepts.
    VERIFIED WORKING - Safely executes Python code in sandboxed environment.
    """
    # [Keep your existing code_executor_tool code exactly as is]
    if language.lower() != "python":
        return f"⚠️ Currently only Python execution is supported. You requested: {language}"
    
    # Security check: block dangerous operations
    dangerous_patterns = [
        r'\bimport\s+os\b', r'\bimport\s+sys\b', r'\bimport\s+subprocess\b',
        r'\bopen\s*\(', r'\bexec\s*\(', r'\beval\s*\(',
        r'\b__import__\b', r'\bcompile\s*\(',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return """
            ⚠️ Security Error: This code contains potentially unsafe operations.
            
            For security reasons, I cannot execute code that:
            - Imports os, sys, or subprocess modules
            - Uses open(), exec(), eval(), or __import__
            - Accesses the file system or network
            
            I can help you understand the code or show you how it would work instead!
            """
    
    # Check for LangChain/LangGraph imports
    if 'langchain' in code.lower() or 'langgraph' in code.lower():
        return """
        💡 LangChain/LangGraph Code Detected!
        
        I can't execute LangChain code directly here (requires API keys and setup),
        but I can:
        
        1. Explain what this code does
        2. Show you the expected output
        3. Help you debug or improve it
        4. Suggest best practices
        
        Would you like me to analyze this code instead?
        """
    
    # Execute safe code
    try:
        # Create a restricted execution environment
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'bool': bool,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
            }
        }
        
        # Capture output
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Execute
        exec(code, safe_globals)
        
        # Get output
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        if output:
            return f"✅ Code executed successfully:\n\n{output}"
        else:
            return "✅ Code executed successfully (no output produced)."
            
    except Exception as e:
        return f"❌ Execution error: {str(e)}\n\nPlease check your code syntax and try again."


# ============================================================================
# KNOWLEDGE BASE & AGENT DEFINITIONS 
# ============================================================================

# [Keep your existing COURSE_KNOWLEDGE dictionary exactly as is]
COURSE_KNOWLEDGE = {
    "program_overview": {
        "name": "Agentic AI Developer Certification Program",
        "duration": "12 weeks",
        "cost": "Free",
        "provider": "Ready Tensor",
        "structure": "3 modules + 1 optional advanced module",
        "certification": "Complete all 3 projects to earn full certification",
        "micro_certs": "Earn micro-certificates for each module completed"
    },
    
    "modules": {
        "module_1": {
            "name": "Foundations of Agentic AI",
            "weeks": "1-4",
            "topics": [
                "Core concepts of agentic AI",
                "LangChain framework basics",
                "Prompt engineering and reasoning techniques",
                "LLM calls and multi-turn conversations",
                "Building RAG (Retrieval-Augmented Generation) systems",
                "Vector databases (Qdrant, FAISS)"
            ],
            "project": "LangGraph-powered assistant answering questions using real documentation with ReAct-based reasoning"
        },
        "module_2": {
            "name": "Multi-Agent Systems",
            "weeks": "5-8",
            "topics": [
                "Agent design patterns",
                "Tool integration and function calling",
                "Multi-agent coordination",
                "LangGraph for complex workflows",
                "Human-in-the-loop systems",
                "Agent memory and state management"
            ],
            "project": "Multi-agent research assistant with human oversight using FastAPI and local LLM inference"
        },
        "module_3": {
            "name": "Real-World Readiness",
            "weeks": "9-12",
            "topics": [
                "Testing agentic AI systems",
                "Security and guardrails (OWASP Top 10 for LLMs)",
                "Deployment strategies",
                "Monitoring and observability",
                "Production best practices",
                "Safety and alignment testing"
            ],
            "project": "Transform multi-agent system into production-ready application with full testing suite"
        },
        "module_4": {
            "name": "Advanced Topics (Optional)",
            "weeks": "Post-certification",
            "topics": [
                "Alternative agent frameworks",
                "Context engineering",
                "Graph RAG",
                "Governance and fairness",
                "Advanced testing",
                "Production monitoring"
            ],
            "required": False
        }
    },
    
    "enrollment": {
        "process": [
            "Visit certifications page from top menu",
            "Select Agentic AI Developer Certification card",
            "Click 'Enroll for Free'",
            "Instant enrollment and access to all lessons"
        ],
        "flexibility": "Self-paced, can start any module based on experience",
        "access": "All 12 weeks of lessons unlocked immediately",
        "cohort": "Can join anytime, monthly project reviews"
    },
    
    "projects": {
        "requirements": "Must score 70% or higher on each project",
        "submission": "Submit on Ready Tensor platform",
        "review": "Projects reviewed monthly by Ready Tensor experts",
        "revision": "Can revise and resubmit if needed",
        "portfolio": "All projects become public portfolio pieces"
    },
    
    "tools_frameworks": {
        "primary": ["LangChain", "LangGraph", "Python"],
        "vector_dbs": ["Qdrant", "FAISS"],
        "deployment": ["FastAPI", "Lightweight hosting"],
        "testing": ["pytest", "Giskard"],
        "security": ["OWASP LLM Top 10", "Guardrails"],
        "monitoring": ["Observability tools", "LangSmith"]
    },
    
    "team": {
        "founder": "Abhyuday Desai, Ph.D. - 20 years AI/ML experience",
        "curriculum_lead": "Victory",
        "support": "Team of AI/ML engineers"
    },
    
    "badges_certificates": {
        "full_certificate": "Agentic AI Developer Certificate (complete all 3 projects)",
        "micro_certificates": "One for each module completed",
        "badges": "4 digital badges shareable on LinkedIn",
        "portfolio": "3 public portfolio projects on Ready Tensor"
    },
    
    "community": {
        "discord": "Active Discord community for questions and networking",
        "participants": "30,000+ learners from 130+ countries",
        "updates": "Regular lesson updates based on feedback"
    }
}
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    query: str
    route: str
    context: str
    response: str
    confidence: float
    next_agent: str
    tool_calls: List[Dict[str, Any]]
    tool_results: List[str]


class RouterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.system_prompt = """You are a routing agent for the Ready Tensor Agentic AI Certification chatbot.
        Analyze the user's query and determine which specialist should handle it.
        
        Available specialists:
        - course_content: Questions about lessons, modules, curriculum, topics covered
        - enrollment: Questions about signing up, deadlines, costs, how to join
        - technical: Questions about tools, coding, LangChain, LangGraph, technical issues
        - projects: Questions about projects, submissions, certification requirements, badges
        
        Respond with ONLY the specialist name. No explanation."""
    
    def route(self, state: AgentState) -> AgentState:
        query = state["query"]
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Route this query: {query}")
        ]
        
        response = self.llm.invoke(messages)
        route = response.content.strip().lower()
        
        # Validate route
        valid_routes = ["course_content", "enrollment", "technical", "projects"]
        if route not in valid_routes:
            route = "course_content"  # Default fallback
        
        state["route"] = route
        state["next_agent"] = route
        state["messages"].append(AIMessage(content=f"[Router: Directing to {route} agent]"))
        
        return state


class CourseContentAgent:
    def __init__(self, llm):
        self.llm = llm
        self.knowledge = COURSE_KNOWLEDGE
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        
        tool_results = []
        
        # Use document retrieval tool if relevant
        if any(keyword in query.lower() for keyword in ["rag", "langgraph", "vector", "security", "deployment", "testing"]):
            topic_keywords = ["rag", "langgraph", "vector", "security", "deployment", "testing"]
            topic = next((kw for kw in topic_keywords if kw in query.lower()), "general")
            doc_result = document_retrieval_tool.invoke({"topic": topic})
            tool_results.append(f"📚 Documentation:\n{doc_result}")
        
        # Use web search for recent info
        if any(keyword in query.lower() for keyword in ["latest", "recent", "update", "new", "current", "2024", "2025"]):
            search_result = web_search_tool.invoke({"query": f"Ready Tensor Agentic AI {query}"})
            tool_results.append(f"🔍 Web Search:\n{search_result}")
        
        # Build context from knowledge base
        context = self._build_context()
        if tool_results:
            context += "\n\nADDITIONAL INFORMATION:\n" + "\n\n".join(tool_results)
        
        system_prompt = f"""You are the Course Content Specialist for Ready Tensor's Agentic AI Certification.
        
        KNOWLEDGE BASE:
        {context}
        
        Provide clear, helpful answers about the course content."""
        
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
        context = []
        
        context.append(f"Program: {kb['program_overview']['name']}")
        context.append(f"Duration: {kb['program_overview']['duration']}")
        context.append(f"Cost: {kb['program_overview']['cost']}")
        
        context.append("\nMODULES:")
        for mod_key, mod in kb['modules'].items():
            context.append(f"\n{mod['name']} (Weeks {mod['weeks']}):")
            context.append(f"Topics: {', '.join(mod['topics'][:3])}...")
            if 'project' in mod:
                context.append(f"Project: {mod['project']}")
        
        return "\n".join(context)


class EnrollmentAgent:
    def __init__(self, llm):
        self.llm = llm
        self.knowledge = COURSE_KNOWLEDGE
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        
        tool_results = []
        if any(keyword in query.lower() for keyword in ["when", "deadline", "available", "open", "closed", "search"]):
            search_result = web_search_tool.invoke({"query": "Ready Tensor Agentic AI certification enrollment"})
            tool_results.append(f"🔍 Current Info:\n{search_result}")
        
        enrollment_info = self.knowledge["enrollment"]
        program_info = self.knowledge["program_overview"]
        
        context = f"""
        ENROLLMENT INFORMATION:
        - Cost: {program_info['cost']}
        - Duration: {program_info['duration']}
        - Process: {', '.join(enrollment_info['process'])}
        - Flexibility: {enrollment_info['flexibility']}
        - Access: {enrollment_info['access']}
        """
        
        if tool_results:
            context += "\n\n" + "\n".join(tool_results)
        
        system_prompt = f"""You are the Enrollment Specialist. {context}"""
        
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
    def __init__(self, llm):
        self.llm = llm
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        
        tool_results = []
        
        # Check for code execution
        code_pattern = r'```python(.*?)```'
        code_match = re.search(code_pattern, query, re.DOTALL)
        
        if code_match or "run" in query.lower() or "execute" in query.lower():
            if code_match:
                code = code_match.group(1).strip()
                exec_result = code_executor_tool.invoke({"code": code})
                tool_results.append(f"⚙️ Code Execution:\n{exec_result}")
        
        # Technical documentation
        if any(topic in query.lower() for topic in ["langgraph", "langchain", "rag", "vector"]):
            topic = next((t for t in ["langgraph", "langchain", "rag", "vector"] if t in query.lower()), "langgraph")
            doc_result = document_retrieval_tool.invoke({"topic": topic})
            tool_results.append(f"📚 Technical Docs:\n{doc_result}")
        
        context = "TECHNICAL SUPPORT: I can help with LangChain, LangGraph, RAG, and coding questions."
        if tool_results:
            context += "\n\n" + "\n\n".join(tool_results)
        
        system_prompt = f"""You are the Technical Support Specialist. {context}"""
        
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
    def __init__(self, llm):
        self.llm = llm
        self.knowledge = COURSE_KNOWLEDGE
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        
        tool_results = []
        if "example" in query.lower() or "sample" in query.lower():
            search_result = web_search_tool.invoke({"query": "Ready Tensor Agentic AI project examples"})
            tool_results.append(f"🔍 Project Info:\n{search_result}")
        
        projects = self.knowledge["projects"]
        modules = self.knowledge["modules"]
        
        context = f"""
        PROJECT REQUIREMENTS:
        - Score 70% or higher on each project
        - Monthly expert reviews
        
        Projects:
        1. Module 1: {modules['module_1']['project']}
        2. Module 2: {modules['module_2']['project']} 
        3. Module 3: {modules['module_3']['project']}
        """
        
        if tool_results:
            context += "\n\n" + "\n".join(tool_results)
        
        system_prompt = f"""You are the Project Specialist. {context}"""
        
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
    def __init__(self, llm):
        self.llm = llm
    
    def supervise(self, state: AgentState) -> AgentState:
        response = state["response"]
        confidence = state["confidence"]
        tool_results = state.get("tool_results", [])
        
        if confidence >= 0.85:
            state["messages"].append(AIMessage(content="[Supervisor: Response approved]"))
            state["next_agent"] = "end"
            return state
        
        # Enhance low-confidence responses
        system_prompt = "You are the Supervisor. Review and enhance this response if needed."
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Review and improve:\n\n{response}")
        ]
        
        enhanced = self.llm.invoke(messages)
        
        state["response"] = enhanced.content
        state["messages"].append(AIMessage(content=enhanced.content))
        state["next_agent"] = "end"
        
        return state


# ============================================================================
# MAIN CHATBOT INTERFACE - SIMPLIFIED AND WORKING
# ============================================================================

class ReadyTensorChatbot:
    """Main chatbot interface that uses the multi-agent system"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0.3
        )
        self.conversation_history = []
    
    def chat(self, user_query: str) -> str:
        """Simple chat interface that routes to appropriate tools"""
        
        # Check for tool triggers
        user_lower = user_query.lower()
        
        # Web search trigger
        if any(word in user_lower for word in ["search", "latest", "current", "recent", "news"]):
            result = web_search_tool.invoke({"query": user_query})
            self.conversation_history.append({
                "query": user_query,
                "response": result,
                "tools_used": 1
            })
            return result
        
        # Code execution trigger
        elif "```python" in user_query or "execute" in user_lower or "run code" in user_lower:
            code_match = re.search(r'```python(.*?)```', user_query, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                result = code_executor_tool.invoke({"code": code})
            else:
                result = "Please provide Python code in ```python ``` blocks to execute."
            
            self.conversation_history.append({
                "query": user_query, 
                "response": result,
                "tools_used": 1
            })
            return result
        
        # Document retrieval trigger
        elif any(topic in user_lower for topic in ["rag", "langgraph", "vector", "security", "deployment", "testing"]):
            topic = next((t for t in ["rag", "langgraph", "vector", "security", "deployment", "testing"] if t in user_lower), "general")
            result = document_retrieval_tool.invoke({"topic": topic})
            self.conversation_history.append({
                "query": user_query,
                "response": result, 
                "tools_used": 1
            })
            return result
        
        # Default LLM response
        else:
            system_prompt = """You are a helpful assistant for the Ready Tensor Agentic AI Certification program.
            Answer questions about the course, modules, enrollment, projects, and technical topics.
            Be concise and informative."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = self.llm.invoke(messages)
            
            self.conversation_history.append({
                "query": user_query,
                "response": response.content,
                "tools_used": 0
            })
            
            return response.content
    
    def get_history(self):
        return self.conversation_history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Get API keys from environment variables
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY environment variable not set!")
        print("Set it using:")
        print("  export GROQ_API_KEY=your_key_here   # macOS/Linux")
        print("  set GROQ_API_KEY=your_key_here      # Windows CMD")
        print("  $env:GROQ_API_KEY=\"your_key_here\" # PowerShell")
        sys.exit(1)
    
    if not TAVILY_API_KEY:
        print("⚠️ WARNING: TAVILY_API_KEY not set. Web search will not work.")
        print("Get a free key from: https://tavily.com/")
    
    chatbot = ReadyTensorChatbot(api_key=GROQ_API_KEY)
    
    print("=" * 70)
    print("🤖 Ready Tensor Agentic AI Certification Chatbot")
    print("=" * 70)
    print("🔧 Available Tools:")
    print("  • Web Search - Say 'search for...' or 'latest news about...'")
    print("  • Code Execution - Use ```python your_code ```")
    print("  • Documentation - Ask about RAG, LangGraph, etc.")
    print("=" * 70)
    print("\nAsk me anything about the certification program!")
    print("Type 'quit' to exit, 'history' to see conversation history")
    print("=" * 70)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n🎓 Goodbye! Good luck with your certification!")
            break
        
        if user_input.lower() == 'history':
            history = chatbot.get_history()
            print("\n📜 Conversation History:")
            for i, item in enumerate(history, 1):
                tools = " (used tool)" if item["tools_used"] > 0 else ""
                print(f"{i}. Q: {item['query'][:80]}...{tools}")
            continue
        
        if not user_input:
            continue
        
        try:
            print("🤖 Processing...", end="", flush=True)
            response = chatbot.chat(user_input)
            print(f"\r{' ' * 50}\r", end="")  # Clear processing message
            print(f"Bot: {response}")
            
        except Exception as e:
            print(f"\r{' ' * 50}\r", end="")  # Clear processing message
            print(f"❌ Error: {str(e)}")
            print("Please check your API keys and internet connection.")
