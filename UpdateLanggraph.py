from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Any
import os
import time

# Try to import graphviz but don't fail if not available
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Note: Graphviz module not found. Will use ASCII chart instead.")

# ANSI escape code for formatting
GREEN = "\033[92;1m"
BLUE_BOLD = "\033[94;1m"
RESET = "\033[0m"

# Initialize the LLM
model_name = "gpt-4o-mini"
# Set API key directly
api_key = ""

# Set environment variable for OpenAI SDK
os.environ["OPENAI_API_KEY"] = api_key

# Initialize the LLM - using the API key directly
llm = ChatOpenAI(
    model=model_name,
    temperature=0,
    api_key=api_key
)

class Agent:
    def __init__(self, name: str, system_message: str):
        self.name = name
        self.system_message = system_message
        self.memory = []  # Store conversation history
    
    def send_message(self, message: str, sender_name: str = "Human"):
        """Add a message to this agent's memory and get a response"""
        # Create conversation history
        messages = [SystemMessage(content=self.system_message)]
        
        # Add conversation history
        for msg in self.memory:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=f"{msg['sender']}: {msg['content']}"))
            else:  # AI message
                messages.append(AIMessage(content=msg["content"]))
        
        # Add the new message
        messages.append(HumanMessage(content=f"{sender_name}: {message}"))
        
        try:
            # Get response from LLM
            response = llm.invoke(messages)
            
            # Store the exchange in memory
            self.memory.append({"role": "human", "sender": sender_name, "content": message})
            self.memory.append({"role": "ai", "content": response.content})
            
            return response.content
        except Exception as e:
            print(f"Error getting response from OpenAI: {e}")
            return f"Error: {e}"
    
    def initiate_chat(self, recipient, message: str):
        """Start a conversation with another agent"""
        print(f"\n{BLUE_BOLD}[{self.name}]{RESET} to {BLUE_BOLD}[{recipient.name}]{RESET}: {message}")
        recipient_response = recipient.send_message(message, self.name)
        print(f"\n{BLUE_BOLD}[{recipient.name}]{RESET}: {recipient_response}")
        return recipient_response

# Create all agents with the same system messages
product_owner = Agent(
    name="Product_Owner",
    system_message="""Represents the customer's needs, manages the product backlog, and prioritizes features for the book store platform.
                    Example tasks include defining shopping cart features, payment integrations, and user profile enhancements.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required:
                    - Total Features / Productivity = Total Duration
                    - e.g., 6 features / 3 features per week = 2 weeks
                    """
)

scrum_master = Agent(
    name="Scrum_Master",
    system_message="""Facilitates Scrum ceremonies, removes obstacles, and ensures team adherence to Agile principles for the book store platform.
                    Supports daily stand-ups, sprint planning, and retrospectives.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Days Required**:
                    - Total Ceremonies / Productivity = Total Duration
                    - e.g., 4 ceremonies / 1 ceremony per day = 4 days
                    """
)

ui_ux_designer = Agent(
    name="UI_UX_Designer",
    system_message="""Designs user interfaces and experiences for the book store platform.
                    Tasks include wireframes, prototypes, and mobile interfaces.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total Screens / Productivity = Total Duration
                    - e.g., 9 screens / 3 screens per week = 3 weeks
                    """
)

solution_architect = Agent(
    name="Solution_Architect",
    system_message="""Designs the system architecture for the book store platform including microservices and integrations.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total Components / Productivity = Total Duration
                    - e.g., 4 components / 1 per week = 4 weeks
                    """
)

developer = Agent(
    name="Developer",
    system_message="""Develops features, integrates APIs, and manages frontend/backend logic for the book store platform.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total SLOC / Productivity = Total Duration
                    - e.g., 1000 SLOC / 500 SLOC per week = 2 weeks
                    """
)

qa_engineer = Agent(
    name="QA_Engineer",
    system_message="""Tests features and validates functionalities for the book store platform.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Days Required**:
                    - Total Test Cases / Productivity = Total Duration
                    - e.g., 25 test cases / 5 per day = 5 days
                    """
)

technical_writer = Agent(
    name="Technical_Writer",
    system_message="""Writes user guides, API docs, and release notes for the book store platform.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total Pages / Productivity = Total Duration
                    - e.g., 8 pages / 4 pages per week = 2 weeks
                    """
)

# Define the message templates
customer_message = """I want to build a web-based mobile app for our bookstore where customers can browse books by genre, read previews, purchase books online, track their shipments, review books, and get personalized reading recommendations."""

def run_simulation():
    print(f"\n{GREEN}Running Book Store Project Simulation with LangChain (Sequential Implementation){RESET}")
    
    # Step 1: Product Owner initiates chat with Scrum Master
    print(f"\n{GREEN}Step 1: Product Owner initiates chat with Scrum Master{RESET}")
    product_owner.initiate_chat(
        scrum_master,
        customer_message
    )
    
    # Step 2: Scrum Master initiates chat with UI/UX Designer
    print(f"\n{GREEN}Step 2: Scrum Master initiates chat with UI/UX Designer{RESET}")
    scrum_master_to_designer = (
        "I have received the customer's requirements from the Product Owner for the book store project. Define user stories and acceptance criteria for the project. "
        "Organize at least 10 user stories, each with a unique ID (e.g., US-01, US-02). "
        "Provide work and effort estimates based on the number of stories documented for this sprint. "
        "Please show your detailed calculation steps for the estimate."
    )
    scrum_master.initiate_chat(
        ui_ux_designer,
        scrum_master_to_designer
    )
    
    # Step 3: Scrum Master discusses with Solution Architect
    print(f"\n{GREEN}Step 3: Scrum Master discusses with Solution Architect{RESET}")
    scrum_master_to_architect = (
        "The UI/UX Designer has completed the user stories for our book store application. Design the technical architecture to support these requirements, prioritizing security, scalability, and compliance. "
        "Include work and effort estimates based on the number of architectural components designed for this sprint. "
        "Please show your detailed calculation steps for the estimate."
    )
    scrum_master.initiate_chat(
        solution_architect,
        scrum_master_to_architect
    )
    
    # Step 4: Scrum Master discusses with Developer
    print(f"\n{GREEN}Step 4: Scrum Master discusses with Developer{RESET}")
    scrum_master_to_developer = (
        "The Architect has completed the design for our book store platform. Begin implementing the features based on the user stories and architectural components. "
        "Estimate the number of source lines of code (SLOC) and effort required for this sprint's development. "
        "Please show your detailed calculation steps for the estimate."
    )
    scrum_master.initiate_chat(
        developer,
        scrum_master_to_developer
    )
    
    # Step 5: Scrum Master discusses with QA Engineer
    print(f"\n{GREEN}Step 5: Scrum Master discusses with QA Engineer{RESET}")
    scrum_master_to_qa = (
        "The development phase is complete for our book store application. Create and execute test cases based on user stories. "
        "Provide work and effort estimates based on the number of test cases created and executed in this sprint. "
        "Please show your detailed calculation steps for the estimate."
    )
    scrum_master.initiate_chat(
        qa_engineer,
        scrum_master_to_qa
    )
    
    # Step 6: Scrum Master discusses with Technical Writer
    print(f"\n{GREEN}Step 6: Scrum Master discusses with Technical Writer{RESET}")
    scrum_master_to_tech_writer = (
        "Testing is complete for the book store platform. Prepare the user documentation and training materials based on the deliverables of this sprint. "
        "Provide work and effort estimates for documentation creation. "
        "Please show your detailed calculation steps for the estimate."
    )
    scrum_master.initiate_chat(
        technical_writer,
        scrum_master_to_tech_writer
    )
    
    # Final summary by the Scrum Master
    print(f"\n{GREEN}Final Summary: Scrum Master provides project summary{RESET}")
    summary_request = (
        "Please summarize the entire book store project based on all team members' contributions. "
        "Create a coherent project timeline that incorporates everyone's time estimates and provides a clear view of the overall project duration."
    )
    
    # We'll manually build a comprehensive context for the summary
    summary_context = []
    for agent in [product_owner, ui_ux_designer, solution_architect, developer, qa_engineer, technical_writer]:
        # Get the last AI message from each agent (their estimate)
        for msg in reversed(agent.memory):
            if msg["role"] == "ai":
                summary_context.append(f"{agent.name} estimate: {msg['content']}")
                break
    
    # Create a system message for the summary
    summary_system_message = """You are the Scrum Master summarizing the entire book store project. 
    Based on the time estimates from each team member, create a comprehensive project timeline 
    and identify the critical path. Calculate the total project duration by considering which tasks 
    can be done in parallel and which must be sequential."""
    
    messages = [SystemMessage(content=summary_system_message)]
    
    # Add the context from each team member
    for context in summary_context:
        messages.append(HumanMessage(content=context))
    
    # Add the final summary request
    messages.append(HumanMessage(content=summary_request))
    
    # Get the summary from the LLM
    summary_response = llm.invoke(messages)
    
    print(f"\n{BLUE_BOLD}[Scrum Master - Final Project Summary]{RESET}: {summary_response.content}")
    
    # Generate workflow flowchart
    generate_workflow_flowchart()
    
    print(f"\n{GREEN}Book Store Project Simulation Complete!{RESET}")

def generate_workflow_flowchart():
    """
    Generate a flowchart visualization of the agent workflow 
    """
    print(f"\n{GREEN}Generating Agent Workflow Flowchart...{RESET}")
    
    # If graphviz is not available, show ASCII chart
    if not HAS_GRAPHVIZ:
        print_ascii_workflow()
        return
    
    try:
        # Create a new directed graph
        dot = Digraph(comment='Book Store Project Agent Workflow')
        
        # Customize the graph appearance
        dot.attr('graph', rankdir='TB', size='8,5', ratio='fill', fontsize='16')
        dot.attr('node', shape='box', style='filled', fillcolor='lightblue', fontname='Arial', fontsize='12')
        dot.attr('edge', fontname='Arial', fontsize='10', fontcolor='#333333')
        
        # Add nodes (agents)
        dot.node('Customer', 'Customer', fillcolor='#FFCCCB')
        dot.node('Product_Owner', 'Product Owner', fillcolor='#ADD8E6')
        dot.node('Scrum_Master', 'Scrum Master', fillcolor='#90EE90')
        dot.node('UI_UX_Designer', 'UI/UX Designer', fillcolor='#FFFACD')
        dot.node('Solution_Architect', 'Solution Architect', fillcolor='#D8BFD8')
        dot.node('Developer', 'Developer', fillcolor='#FFE4B5')
        dot.node('QA_Engineer', 'QA Engineer', fillcolor='#E6E6FA')
        dot.node('Technical_Writer', 'Technical Writer', fillcolor='#F0FFF0')
        
        # Add edges (interaction flow)
        dot.edge('Customer', 'Product_Owner', label='Requirements')
        dot.edge('Product_Owner', 'Scrum_Master', label='Project\nRequirements')
        dot.edge('Scrum_Master', 'UI_UX_Designer', label='Define User\nStories')
        dot.edge('UI_UX_Designer', 'Scrum_Master', label='User Stories\n& Estimates')
        dot.edge('Scrum_Master', 'Solution_Architect', label='Design\nArchitecture')
        dot.edge('Solution_Architect', 'Scrum_Master', label='Architecture\n& Estimates')
        dot.edge('Scrum_Master', 'Developer', label='Implement\nFeatures')
        dot.edge('Developer', 'Scrum_Master', label='Implementation\n& Estimates')
        dot.edge('Scrum_Master', 'QA_Engineer', label='Test\nImplementation')
        dot.edge('QA_Engineer', 'Scrum_Master', label='Testing\n& Estimates')
        dot.edge('Scrum_Master', 'Technical_Writer', label='Create\nDocumentation')
        dot.edge('Technical_Writer', 'Scrum_Master', label='Documentation\n& Estimates')
        
        # Save flowchart to a file
        flowchart_filename = 'book_store_workflow'
        try:
            dot.render(flowchart_filename, format='png', view=False)
            print(f"\n{GREEN}Flowchart generated as '{flowchart_filename}.png'{RESET}")
            print(f"{GREEN}You can view it in the same directory as this script.{RESET}")
        except Exception as render_error:
            print(f"\n{GREEN}Could not render flowchart: {render_error}{RESET}")
            print(f"\n{GREEN}Trying to save just the DOT file...{RESET}")
            try:
                with open(f"{flowchart_filename}.dot", "w") as f:
                    f.write(dot.source)
                print(f"{GREEN}DOT file saved to '{flowchart_filename}.dot'{RESET}")
                print(f"{GREEN}You can render it using: dot -Tpng {flowchart_filename}.dot -o {flowchart_filename}.png{RESET}")
            except Exception as write_error:
                print(f"{GREEN}Could not save DOT file: {write_error}{RESET}")
                print_ascii_workflow()
        
    except Exception as e:
        print(f"\n{GREEN}Could not generate flowchart: {e}{RESET}")
        print_ascii_workflow()

def print_ascii_workflow():
    """Print a text-based ASCII workflow diagram"""
    print(f"\n{GREEN}Book Store Project Workflow (ASCII Diagram):{RESET}")
    print("""
            +------------+                              
            |  Customer  |                              
            +------------+                              
                  |  Requirements                       
                  v                                     
            +------------+                              
            |  Product   |                              
            |   Owner    |                              
            +------------+                              
                  |  Project Requirements               
                  v                                     
            +------------+                              
            |   Scrum    | ───────────────┐             
            |   Master   | <──────────────┤             
            +------------+                |             
             | ^   | ^   | ^   | ^   | ^   |             
             | |   | |   | |   | |   | |   |             
             v |   v |   v |   v |   v |   |             
    +-------+ | +-+ | +-+ | +-+ | +-+ | +-+             
    |UI/UX  | | |S| | |D| | |Q| | |T| | |              
    |Design | | |l| | |v| | | | | |c| | |              
    |Stories| | |o| | |e| | | | | |h| | |              
    |      |───|u|───|e|───|E|───| | |              
    |      | | |t| | |l| | |n| | | | | |              
    |      | | |i| | |o| | |g| | |W| | |              
    |      | | |o| | |p| | |i| | |r| | |              
    |      | | |n| | |e| | |n| | |i| | |              
    |      | | | | | |r| | |e| | |t| | |              
    |      | | |A| | | | | |e| | |e| | |              
    |      | | |r| | | | | |r| | |r| | |              
    |      | | |c| | | | | | | | | | | |              
    |      | | |h| | | | | | | | | | | |              
    +-------+ | +-+ | +-+ | +-+ | +-+ | Final          
              |     |     |     |     | Summary         
              |     |     |     |     v                 
              |     |     |     |    +-----------+      
              |     |     |     |    | Project   |      
              |     |     |     |    | Timeline  |      
              |     |     |     |    +-----------+      
              |     |     |     |                       
              |     |     |     |  Test Results         
              |     |     |     |  & Estimates          
              |     |     |     v                       
              |     |     |    +---------+              
              |     |     |    |   QA    |              
              |     |     |    | Engineer|              
              |     |     |    +---------+              
              |     |     |                             
              |     |     |  Implementation             
              |     |     |  & Estimates                
              |     |     v                             
              |     |    +---------+                    
              |     |    |Developer|                    
              |     |    +---------+                    
              |     |                                   
              |     |  Architecture                     
              |     |  & Estimates                      
              |     v                                   
              |    +----------+                         
              |    |  Solution |                        
              |    | Architect |                        
              |    +----------+                         
              |                                         
              |  User Stories                           
              |  & Estimates                            
              v                                         
         +----------+                                   
         |  UI/UX   |                                   
         | Designer |                                   
         +----------+                                   
    """)
    
    print(f"\n{GREEN}Workflow Description:{RESET}")
    print("""
    ► Customer → Product Owner: Provides requirements
    ► Product Owner → Scrum Master: Shares project requirements
    ► Scrum Master → UI/UX Designer: Requests user stories
    ► UI/UX Designer → Scrum Master: Delivers user stories & estimates
    ► Scrum Master → Solution Architect: Requests architecture design
    ► Solution Architect → Scrum Master: Delivers architecture & estimates
    ► Scrum Master → Developer: Requests implementation
    ► Developer → Scrum Master: Delivers implementation & estimates
    ► Scrum Master → QA Engineer: Requests testing
    ► QA Engineer → Scrum Master: Delivers testing results & estimates
    ► Scrum Master → Technical Writer: Requests documentation
    ► Technical Writer → Scrum Master: Delivers documentation & estimates
    ► Scrum Master: Provides final project summary
    """)

# Run the simulation
if __name__ == "__main__":
    run_simulation() 
