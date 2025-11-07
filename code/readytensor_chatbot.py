"""
Multi-Agent Chatbot for Ready Tensor Agentic AI Certification
Uses LangGraph for agent orchestration with integrated tools
"""
#At line 893 you will put your API key1

import os
from typing import TypedDict, Annotated, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import operator
import json
import re
from datetime import datetime

# ============================================================================
# TOOLS DEFINITION
# ============================================================================

@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for real-time information about Ready Tensor courses,
    agentic AI, LangChain, LangGraph, and related topics.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as formatted text
    """
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return f"Search results for '{query}':\n\n{results}"
    except Exception as e:
        return f"Search failed: {str(e)}. Using cached knowledge instead."


@tool
def document_retrieval_tool(topic: str, module: str = "all") -> str:
    """
    Retrieve specific documentation about course topics, modules, or concepts.
    This simulates a vector database lookup for relevant course materials.
    
    Args:
        topic: The topic to search for (e.g., "RAG", "LangGraph", "vector databases")
        module: Specific module number (1, 2, 3, 4) or "all" for all modules
        
    Returns:
        Relevant documentation snippets
    """
    # Simulated document store (in production, this would be a vector DB)
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
        return f"No specific documentation found for '{topic}'. Try searching for: RAG, LangGraph, vector_databases, security, deployment, or testing."


@tool
def code_executor_tool(code: str, language: str = "python") -> str:
    """
    Execute simple Python code snippets to help users test LangChain/LangGraph concepts.
    For security, only allows safe operations (no file I/O, network, or system calls).
    
    Args:
        code: Python code to execute (safe operations only)
        language: Programming language (currently only "python" supported)
        
    Returns:
        Execution result or explanation
    """
    if language.lower() != "python":
        return f"Currently only Python execution is supported. You requested: {language}"
    
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
# KNOWLEDGE BASE - Ready Tensor Course Content
# ============================================================================

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
    tool_calls: List[Dict[str, Any]]
    tool_results: List[str]

# ============================================================================
# AGENT DEFINITIONS WITH TOOL INTEGRATION
# ============================================================================

class RouterAgent:
    """Routes user queries to appropriate specialist agents"""
    
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
    """Handles questions about course content, modules, and curriculum with tool support"""
    
    def __init__(self, llm):
        self.llm = llm
        self.knowledge = COURSE_KNOWLEDGE
        self.tools = [document_retrieval_tool, web_search_tool]
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        
        # Decide if we need tools
        should_retrieve_docs = any(keyword in query.lower() for keyword in 
            ["rag", "langgraph", "vector", "security", "deployment", "testing", "how to", "example"])
        
        should_search_web = any(keyword in query.lower() for keyword in 
            ["latest", "recent", "update", "new", "current", "2024", "2025"])
        
        tool_results = []
        
        # Use document retrieval tool if relevant
        if should_retrieve_docs:
            # Extract topic from query
            topic_keywords = ["rag", "langgraph", "vector", "security", "deployment", "testing"]
            topic = next((kw for kw in topic_keywords if kw in query.lower()), "general")
            
            doc_result = document_retrieval_tool.invoke({"topic": topic})
            tool_results.append(f"📚 Retrieved Documentation:\n{doc_result}")
        
        # Use web search if asking about recent info
        if should_search_web:
            search_query = f"Ready Tensor Agentic AI certification {query}"
            search_result = web_search_tool.invoke({"query": search_query})
            tool_results.append(f"🔍 Web Search:\n{search_result}")
        
        # Build context
        context = self._build_context()
        if tool_results:
            context += "\n\nTOOL RESULTS:\n" + "\n\n".join(tool_results)
        
        system_prompt = f"""You are the Course Content Specialist for Ready Tensor's Agentic AI Certification.
        Use the knowledge base and tool results to answer questions accurately.
        
        KNOWLEDGE BASE:
        {context}
        
        Provide clear, specific answers. Mention if you used tools to get current information."""
        
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
        """Build formatted context from knowledge base"""
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
        
        context.append(f"\nKey Tools: {', '.join(kb['tools_frameworks']['primary'])}")
        
        return "\n".join(context)


class EnrollmentAgent:
    """Handles enrollment, logistics, and program access questions"""
    
    def __init__(self, llm):
        self.llm = llm
        self.knowledge = COURSE_KNOWLEDGE
        self.tools = [web_search_tool]
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        
        # Use web search for current enrollment status
        should_search = any(keyword in query.lower() for keyword in 
            ["when", "deadline", "available", "open", "closed"])
        
        tool_results = []
        if should_search:
            search_result = web_search_tool.invoke({
                "query": "Ready Tensor Agentic AI certification enrollment 2025"
            })
            tool_results.append(f"🔍 Current Status:\n{search_result}")
        
        enrollment_info = self.knowledge["enrollment"]
        program_info = self.knowledge["program_overview"]
        community = self.knowledge["community"]
        
        context = f"""
        ENROLLMENT INFORMATION:
        - Cost: {program_info['cost']}
        - Duration: {program_info['duration']}
        - Enrollment Process: {', '.join(enrollment_info['process'])}
        - Flexibility: {enrollment_info['flexibility']}
        - Access: {enrollment_info['access']}
        - Community: {community['participants']} from 130+ countries
        
        URL: https://app.readytensor.ai/certifications/agentic-ai-cert-U7HxeL7a
        """
        
        if tool_results:
            context += "\n\nCURRENT INFO:\n" + "\n".join(tool_results)
        
        system_prompt = f"""You are the Enrollment Specialist for Ready Tensor's Agentic AI Certification.
        {context}
        
        Be encouraging and helpful. Emphasize that it's free, self-paced, and beginner-friendly."""
        
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
    """Handles technical questions with code execution and documentation retrieval"""
    
    def __init__(self, llm):
        self.llm = llm
        self.knowledge = COURSE_KNOWLEDGE
        self.tools = [code_executor_tool, document_retrieval_tool, web_search_tool]
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        
        tool_results = []
        
        # Check if user wants to execute code
        code_pattern = r'```python(.*?)```'
        code_match = re.search(code_pattern, query, re.DOTALL)
        
        if code_match or "run" in query.lower() or "execute" in query.lower():
            if code_match:
                code = code_match.group(1).strip()
                exec_result = code_executor_tool.invoke({"code": code})
                tool_results.append(f"⚙️ Code Execution:\n{exec_result}")
        
        # Check for technical documentation needs
        tech_topics = ["langgraph", "langchain", "rag", "vector", "agent", "tool"]
        needs_docs = any(topic in query.lower() for topic in tech_topics)
        
        if needs_docs:
            topic = next((t for t in tech_topics if t in query.lower()), "langgraph")
            doc_result = document_retrieval_tool.invoke({"topic": topic})
            tool_results.append(f"📚 Technical Docs:\n{doc_result}")
        
        # Search for latest technical info
        if any(word in query.lower() for word in ["latest", "new", "version", "update"]):
            search_result = web_search_tool.invoke({"query": f"LangChain LangGraph {query}"})
            tool_results.append(f"🔍 Latest Info:\n{search_result}")
        
        tools = self.knowledge["tools_frameworks"]
        
        context = f"""
        TECHNICAL STACK:
        - Primary: {', '.join(tools['primary'])}
        - Vector DBs: {', '.join(tools['vector_dbs'])}
        - Deployment: {', '.join(tools['deployment'])}
        - Testing: {', '.join(tools['testing'])}
        """
        
        if tool_results:
            context += "\n\nTOOL RESULTS:\n" + "\n\n".join(tool_results)
        
        system_prompt = f"""You are the Technical Support Specialist for Ready Tensor's Agentic AI Certification.
        {context}
        
        Provide practical guidance. If code was executed, explain the results."""
        
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
    """Handles questions about projects, submissions, and certification"""
    
    def __init__(self, llm):
        self.llm = llm
        self.knowledge = COURSE_KNOWLEDGE
        self.tools = [web_search_tool]
    
    def answer(self, state: AgentState) -> AgentState:
        query = state["query"]
        
        # Search for latest project guidelines
        tool_results = []
        if "example" in query.lower() or "sample" in query.lower():
            search_result = web_search_tool.invoke({
                "query": "Ready Tensor Agentic AI certification project examples"
            })
            tool_results.append(f"🔍 Project Examples:\n{search_result}")
        
        projects = self.knowledge["projects"]
        certs = self.knowledge["badges_certificates"]
        modules = self.knowledge["modules"]
        
        context = f"""
        PROJECT & CERTIFICATION INFO:
        
        Requirements:
        - Score 70% or higher on each project
        - Monthly reviews by experts
        
        Module Projects:
        1. Module 1: {modules['module_1']['project']}
        2. Module 2: {modules['module_2']['project']}
        3. Module 3: {modules['module_3']['project']}
        
        Certificates: {certs['full_certificate']}
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
    """Reviews and finalizes responses, ensures quality"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def supervise(self, state: AgentState) -> AgentState:
        response = state["response"]
        confidence = state["confidence"]
        tool_results = state.get("tool_results", [])
        
        # If confidence is high and tools were used appropriately, approve
        if confidence >= 0.85:
            if tool_results:
                state["messages"].append(AIMessage(
                    content=f"[Supervisor: Response approved - Used {len(tool_results)} tool(s)]"
                ))
            else:
                state["messages"].append(AIMessage(content="[Supervisor: Response approved]"))
            state["next_agent"] = "end"
            return state
        
        # Enhance low-confidence responses
        system_prompt = """You are the Supervisor. Review and enhance this response if needed.
        Keep responses concise but informative you must obey the following rules
        1. Do not Give any information outside any programming related concept or ready tensor publication given to you,
        simply say "I am a chatbot strictly for the ready tensor platform".
        2.Try to keep response as concise as possible"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Review:\n\n{response}")
        ]
        
        enhanced = self.llm.invoke(messages)
        
        state["response"] = enhanced.content
        state["messages"].append(AIMessage(content=enhanced.content))
        state["next_agent"] = "end"
        
        return state

# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

def create_chatbot(api_key: str, model: str = "llama-3.1-8b-instant"):
    """Create the multi-agent chatbot workflow with tools"""
    
    llm = ChatGroq(model=model, api_key=api_key, temperature=0.3)
    
    # Initialize agents with tool support
    router = RouterAgent(llm)
    course_agent = CourseContentAgent(llm)
    enrollment_agent = EnrollmentAgent(llm)
    technical_agent = TechnicalAgent(llm)
    project_agent = ProjectAgent(llm)
    supervisor = SupervisorAgent(llm)
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router.route)
    workflow.add_node("course_content", course_agent.answer)
    workflow.add_node("enrollment", enrollment_agent.answer)
    workflow.add_node("technical", technical_agent.answer)
    workflow.add_node("projects", project_agent.answer)
    workflow.add_node("supervisor", supervisor.supervise)
    
    def route_to_specialist(state: AgentState) -> str:
        return state["next_agent"]
    
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        route_to_specialist,
        {
            "course_content": "course_content",
            "enrollment": "enrollment",
            "technical": "technical",
            "projects": "projects"
        }
    )
    
    workflow.add_edge("course_content", "supervisor")
    workflow.add_edge("enrollment", "supervisor")
    workflow.add_edge("technical", "supervisor")
    workflow.add_edge("projects", "supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        route_to_specialist,
        {"end": END}
    )
    
    return workflow.compile()

# ============================================================================
# MAIN CHATBOT INTERFACE
# ============================================================================

class ReadyTensorChatbot:
    """Main chatbot interface with tool support"""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.graph = create_chatbot(api_key, model)
        self.conversation_history = []
    
    def chat(self, user_query: str) -> str:
        """Process a user query and return response"""
        
        initial_state = {
            "messages": [],
            "query": user_query,
            "route": "",
            "context": "",
            "response": "",
            "confidence": 0.0,
            "next_agent": "",
            "tool_calls": [],
            "tool_results": []
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        # Extract final response
        response = result["response"]
        
        # Store in conversation history with tool usage
        self.conversation_history.append({
            "query": user_query,
            "response": response,
            "route": result["route"],
            "tools_used": len(result.get("tool_results", []))
        })
        
        return response
    
    def get_history(self):
        """Get conversation history"""
        return self.conversation_history
    
    def reset(self):
        """Reset conversation history"""
        self.conversation_history = []


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize chatbot
    API_KEY = os.getenv("GROQ_API_KEY", "Your API key here")##########################
    
    chatbot = ReadyTensorChatbot(api_key=API_KEY)
    
    print("=" * 70)
    print("Ready Tensor Agentic AI Certification Chatbot (With Tools!)")
    print("=" * 70)
    print("🔧 Available Tools:")
    print("  1. Web Search - Real-time information")
    print("  2. Document Retrieval - Course documentation")
    print("  3. Code Executor - Test Python snippets")
    print("=" * 70)
    print("\nAsk me anything about the certification program!")
    print("Type 'quit' to exit, 'history' to see conversation history")
    print("=" * 70)
    print()
    
    # Show example queries
    print("💡 Example queries:")
    print("  • What is RAG? (uses document retrieval)")
    print("  • Search for latest LangGraph updates (uses web search)")
    print("  • Execute: print('Hello AI!') (uses code executor)")
    print("  • Tell me about Module 2")
    print()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! Good luck with your certification! 🚀")
            break
        
        if user_input.lower() == 'history':
            history = chatbot.get_history()
            print("\n--- Conversation History ---")
            for i, item in enumerate(history, 1):
                tools_used = item.get('tools_used', 0)
                tool_info = f" [Used {tools_used} tool(s)]" if tools_used > 0 else ""
                print(f"\n{i}. Q: {item['query']}")
                print(f"   Route: {item['route']}{tool_info}")
                print(f"   A: {item['response'][:150]}...")
            print()
            continue
        
        if not user_input:
            continue
        
        try:
            print("\n🤖 Processing", end="", flush=True)
            for _ in range(3):
                print(".", end="", flush=True)
                import time
                time.sleep(0.3)
            print()
            
            response = chatbot.chat(user_input)
            print(f"\nBot: {response}\n")
        except Exception as e:
            print(f"\nError: {str(e)}\n")
            print("Please make sure you have set your GROQ_API_KEY environment variable.\n")
