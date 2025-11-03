"""
Test script to demonstrate the multi-agent chatbot functionality
Run this to see example conversations and verify the system works
"""

import os
from readytensor_chatbot import ReadyTensorChatbot, COURSE_KNOWLEDGE
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
#it is used to print text
def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)
#this function below is used to show the user query and show the show the AI's reply in 1 secs
def print_response(query, response, delay=1):
    """Print query and response with formatting"""
    print(f"\n💬 Query: {query}")
    time.sleep(delay)
    print(f"🤖 Response: {response}\n")

def test_knowledge_base():
    """Test the knowledge base structure"""
    print_header("KNOWLEDGE BASE VALIDATION")
    
    kb = COURSE_KNOWLEDGE
    
    print("\n✅ Program Overview:")
    print(f"   - Name: {kb['program_overview']['name']}")
    print(f"   - Duration: {kb['program_overview']['duration']}")
    print(f"   - Cost: {kb['program_overview']['cost']}")
    
    print("\n✅ Modules:")
    for i, (key, mod) in enumerate(kb['modules'].items(), 1):
        print(f"   {i}. {mod['name']} ({mod.get('weeks', 'N/A')})")
    
    print("\n✅ Tools & Frameworks:")
    print(f"   - Primary: {', '.join(kb['tools_frameworks']['primary'])}")
    print(f"   - Vector DBs: {', '.join(kb['tools_frameworks']['vector_dbs'])}")
    
    print("\n✅ Enrollment Info:")
    print(f"   - Process: {len(kb['enrollment']['process'])} steps")
    print(f"   - Flexibility: {kb['enrollment']['flexibility']}")

def test_agent_routing():
    """Test the routing functionality"""
    print_header("AGENT ROUTING TEST")
    
    test_queries = [
        ("What topics are covered in Module 1?", "course_content"),
        ("How do I enroll in the program?", "enrollment"),
        ("What is LangGraph?", "technical"),
        ("What are the project requirements?", "projects"),
    ]
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found. Please set it in .env file.")
        return
    
    chatbot = ReadyTensorChatbot(api_key=api_key)
    
    print("\nTesting query routing to appropriate agents...\n")
    
    for query, expected_route in test_queries:
        try:
            print(f"📝 Query: '{query}'")
            response = chatbot.chat(query)
            history = chatbot.get_history()
            actual_route = history[-1]['route'] if history else 'unknown'
            
            # Check if routed correctly
            match = "✅" if actual_route == expected_route else "⚠️"
            print(f"{match} Routed to: {actual_route} (expected: {expected_route})")
            print(f"💬 Response preview: {response[:100]}...")
            print()
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")

def run_demo_conversation():
    """Run a comprehensive demo conversation"""
    print_header("DEMO CONVERSATION")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found. Please set it in .env file.")
        print("\nTo fix this:")
        print("1. Create a .env file")
        print("2. Add: GROQ_API_KEY=your-api-key-here")
        return
    
    print("\n🤖 Initializing chatbot...")
    chatbot = ReadyTensorChatbot(api_key=api_key)
    print("✅ Chatbot ready!\n")
    
    # Demo queries covering different agents
    demo_queries = [
        "What is the Ready Tensor Agentic AI Certification about?",
        "How long does the program take and is it free?",
        "What will I learn in Module 1?",
        "What tools and frameworks are used in the course?",
        "How do I enroll in the program?",
        "What are the certification requirements?",
    ]
    
    print("Running demo conversation with various queries...\n")
    print("-" * 70)
    
    for i, query in enumerate(demo_queries, 1):
        try:
            print(f"\n[Question {i}/{len(demo_queries)}]")
            print(f"💬 You: {query}")
            print("\n🤖 Bot: ", end="", flush=True)
            
            # Simulate typing effect
            time.sleep(0.5)
            
            response = chatbot.chat(query)
            print(response)
            
            print("-" * 70)
            time.sleep(1)  # Pause between queries
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print(f"   Query: {query}\n")
    
    # Show conversation summary
    print_header("CONVERSATION SUMMARY")
    history = chatbot.get_history()
    print(f"\n📊 Total queries: {len(history)}")
    print("\n📈 Agent usage:")
    
    routes = {}
    for item in history:
        route = item['route']
        routes[route] = routes.get(route, 0) + 1
    
    for route, count in routes.items():
        print(f"   - {route}: {count} queries")

def test_api_endpoints():
    """Test API endpoints (requires api.py to be running)"""
    print_header("API ENDPOINT TEST")
    
    try:
        import httpx
        
        base_url = "http://localhost:8000"
        
        print(f"\n🌐 Testing API at {base_url}...")
        print("   (Make sure api.py is running in another terminal)\n")
        
        # Test health endpoint
        print("1. Testing /health endpoint...")
        response = httpx.get(f"{base_url}/health", timeout=5.0)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}\n")
        
        # Test chat endpoint
        print("2. Testing /chat endpoint...")
        response = httpx.post(
            f"{base_url}/chat",
            json={"query": "What is Module 1 about?", "session_id": "test"},
            timeout=30.0
        )
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Route: {data.get('route')}")
        print(f"   Response: {data.get('response')[:100]}...\n")
        
        # Test knowledge endpoint
        print("3. Testing /knowledge/modules endpoint...")
        response = httpx.get(f"{base_url}/knowledge/modules", timeout=5.0)
        print(f"   Status: {response.status_code}")
        modules = response.json()
        print(f"   Modules found: {len(modules)}\n")
        
        print("✅ All API tests passed!")
        
    except httpx.ConnectError:
        print("⚠️  Could not connect to API server.")
        print("   Start the API server with: python api.py")
    except ImportError:
        print("⚠️  httpx not installed. Install with: pip install httpx")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def interactive_mode():
    """Run interactive chat mode"""
    print_header("INTERACTIVE MODE")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found. Please set it in .env file.")
        return
    
    chatbot = ReadyTensorChatbot(api_key=api_key)
    
    print("\n🤖 Ready Tensor Chatbot - Interactive Mode")
    print("\nCommands:")
    print("  • Type your question and press Enter")
    print("  • 'history' - Show conversation history")
    print("  • 'reset' - Clear conversation history")
    print("  • 'quit' - Exit")
    print("\n" + "-" * 70 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Good luck with your certification!\n")
                break
            
            if user_input.lower() == 'history':
                history = chatbot.get_history()
                if not history:
                    print("\n📝 No conversation history yet.\n")
                else:
                    print("\n" + "=" * 70)
                    print("  CONVERSATION HISTORY")
                    print("=" * 70)
                    for i, item in enumerate(history, 1):
                        print(f"\n{i}. Q: {item['query']}")
                        print(f"   Route: {item['route']}")
                        print(f"   A: {item['response'][:200]}...")
                    print()
                continue
            
            if user_input.lower() == 'reset':
                chatbot.reset()
                print("\n♻️  Conversation history cleared.\n")
                continue
            
            if not user_input:
                continue
            
            print("\n🤖 Bot: ", end="", flush=True)
            response = chatbot.chat(user_input)
            print(response + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")

def main():
    """Main test menu"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     Ready Tensor Chatbot - Test & Demo Suite                   ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nSelect a test to run:\n")
    print("1. Validate Knowledge Base")
    print("2. Test Agent Routing")
    print("3. Run Demo Conversation")
    print("4. Test API Endpoints")
    print("5. Interactive Chat Mode")
    print("6. Run All Tests")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-6): ").strip()
    
    if choice == "1":
        test_knowledge_base()
    elif choice == "2":
        test_agent_routing()
    elif choice == "3":
        run_demo_conversation()
    elif choice == "4":
        test_api_endpoints()
    elif choice == "5":
        interactive_mode()
    elif choice == "6":
        test_knowledge_base()
        test_agent_routing()
        run_demo_conversation()
        test_api_endpoints()
    elif choice == "0":
        print("\n👋 Goodbye!\n")
    else:
        print("\n❌ Invalid choice. Please try again.\n")
        main()

if __name__ == "__main__":
    main()
