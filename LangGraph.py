#!/usr/bin/env python3
"""
Book Store Project Simulation using LangGraph

Required packages:
pip install langchain langchain_openai langgraph graphviz matplotlib networkx

Usage:
    python Experiment_4_proper_langgraph.py 
    python Experiment_4_proper_langgraph.py --api-key YOUR_API_KEY
    python Experiment_4_proper_langgraph.py --help

Note: 
1. Set your OpenAI API key before running:
   export OPENAI_API_KEY='your-key'
   
   Or use the --api-key parameter.
"""

import os
import time
import argparse
from typing import List, Dict, Any, TypedDict, Annotated, Sequence, Optional, Literal
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

# Try to import graphviz but don't fail if not available
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Note: Graphviz module not found. Will try matplotlib or fallback to ASCII chart.")

# Try to import matplotlib and networkx as alternative
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    if not HAS_GRAPHVIZ:
        print("Note: Neither Graphviz nor Matplotlib/NetworkX available. Will use ASCII chart.")

# ANSI escape code for formatting
GREEN = "\033[92;1m"
BLUE_BOLD = "\033[94;1m"
RESET = "\033[0m"

# Get the OpenAI API key from environment variable - will be overridden by command line if provided
api_key = os.environ.get("OPENAI_API_KEY", "")  

# Initialize the LLM variables - will be set properly after argument parsing
model_name = "gpt-4o-mini"

# Define llm as None initially, will be initialized after argument parsing
llm = None

def initialize_llm(api_key, model_name):
    """Initialize the LLM with the given API key and model name"""
    if not api_key:
        print("\033[93mNo OpenAI API key found in environment variables.\033[0m")
        user_input = input("Would you like to enter an OpenAI API key now? (yes/no): ")
        if user_input.lower() in ['yes', 'y']:
            api_key = input("Enter your OpenAI API key: ").strip()
        else:
            print("\033[91mNo valid API key provided. The script cannot run without an API key.\033[0m")
            print("You can get an API key from https://platform.openai.com/account/api-keys")
            print("You can set it permanently using: export OPENAI_API_KEY='your-key'")
            return None

    # Initialize the LLM - using the API key
    try:
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=api_key
        )
    except Exception as e:
        print(f"\033[91mError initializing OpenAI client: {e}\033[0m")
        print("Please check your API key and try again.")
        return None

# Define the roles that will be used in our workflow
class Role(str, Enum):
    PRODUCT_OWNER = "product_owner"
    SCRUM_MASTER = "scrum_master"
    UI_UX_DESIGNER = "ui_ux_designer"
    SOLUTION_ARCHITECT = "solution_architect"
    DEVELOPER = "developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    BACKEND_DEVELOPER = "backend_developer"
    RECOMMENDATION_DEVELOPER = "recommendation_developer"
    QA_ENGINEER = "qa_engineer"
    TECHNICAL_WRITER = "technical_writer"
    DEVOPS_ENGINEER = "devops_engineer"
    SECURITY_ENGINEER = "security_engineer"
    ECOMMERCE_SPECIALIST = "ecommerce_specialist"
    CUSTOMER = "customer"

# System messages for each role
SYSTEM_MESSAGES = {
    Role.PRODUCT_OWNER: """Represents the customer's needs, manages the product backlog, and prioritizes features for the book store platform.
                    Example tasks include defining shopping cart features, payment integrations, and user profile enhancements.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required:
                    - Total Features / Productivity = Total Duration
                    - e.g., 6 features / 3 features per week = 2 weeks
                    """,
    
    Role.SCRUM_MASTER: """Facilitates Scrum ceremonies, removes obstacles, and ensures team adherence to Agile principles for the book store platform.
                    Supports daily stand-ups, sprint planning, and retrospectives.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Days Required**:
                    - Total Ceremonies / Productivity = Total Duration
                    - e.g., 4 ceremonies / 1 ceremony per day = 4 days
                    """,
    
    Role.UI_UX_DESIGNER: """Designs user interfaces and experiences for the book store platform.
                    Tasks include wireframes, prototypes, and mobile interfaces.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total Screens / Productivity = Total Duration
                    - e.g., 9 screens / 3 screens per week = 3 weeks
                    """,
    
    Role.SOLUTION_ARCHITECT: """Designs the system architecture for the book store platform including microservices and integrations.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total Components / Productivity = Total Duration
                    - e.g., 4 components / 1 per week = 4 weeks
                    """,
    
    Role.DEVELOPER: """Develops features, integrates APIs, and manages frontend/backend logic for the book store platform.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total SLOC / Productivity = Total Duration
                    - e.g., 1000 SLOC / 500 SLOC per week = 2 weeks
                    """,
    
    Role.FRONTEND_DEVELOPER: """Implements responsive mobile web interfaces and develops interactive features like book previews and shopping cart for the book store platform.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total SLOC / Productivity = Total Duration
                    - e.g., 500 SLOC / 250 SLOC per week = 2 weeks
                    """,
    
    Role.BACKEND_DEVELOPER: """Creates APIs for book catalog, user management, and order processing, and implements business logic for the retail bookstore operations.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total SLOC / Productivity = Total Duration
                    - e.g., 500 SLOC / 250 SLOC per week = 2 weeks
                    """,
    
    Role.RECOMMENDATION_DEVELOPER: """Creates personalized book recommendation algorithms and implements user behavior tracking for relevant suggestions.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total SLOC / Productivity = Total Duration
                    - e.g., 400 SLOC / 200 SLOC per week = 2 weeks
                    """,
    
    Role.QA_ENGINEER: """Tests features and validates functionalities for the book store platform.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Days Required**:
                    - Total Test Cases / Productivity = Total Duration
                    - e.g., 25 test cases / 5 per day = 5 days
                    """,
    
    Role.TECHNICAL_WRITER: """Writes user guides, API docs, and release notes for the book store platform.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total Pages / Productivity = Total Duration
                    - e.g., 8 pages / 4 pages per week = 2 weeks
                    """,
    
    Role.DEVOPS_ENGINEER: """Handles CI/CD, infrastructure, and deployment automation for the book store platform.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total Tasks / Productivity = Total Duration
                    - e.g., 6 tasks / 2 tasks per week = 3 weeks
                    """,
    
    Role.SECURITY_ENGINEER: """Conducts code reviews, penetration testing, and secures sensitive data for the book store platform.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total Security Tasks / Productivity = Total Duration
                    - e.g., 3 tasks / 1 task per week = 3 weeks
                    """,
    
    Role.ECOMMERCE_SPECIALIST: """Provides best practices in book cataloging, checkout UX, and promotions for the book store platform.
                    IMPORTANT: When responding, always show your work in this format:
                    Estimated Weeks Required**:
                    - Total Areas / Productivity = Total Duration
                    - e.g., 6 areas / 2 per week = 3 weeks
                    """,
    
    Role.CUSTOMER: """You are a customer who wants a new book store app. You will describe your requirements for the application.
                    """
}

# Define the state of our workflow
class AgentState(TypedDict):
    """Represents the state of the workflow."""
    messages: Annotated[Sequence[Any], add_messages]
    sender: str
    receiver: str
    next_agent: Optional[str]
    done: bool
    summary: Optional[str]
    estimates: Dict[str, str]

# Define the function to initialize the agent state
def get_initial_state() -> AgentState:
    """Initialize the agent state."""
    return {
        "messages": [],
        "sender": Role.CUSTOMER,
        "receiver": Role.PRODUCT_OWNER,
        "next_agent": Role.PRODUCT_OWNER,
        "done": False,
        "summary": None,
        "estimates": {}
    }

# Function to create an agent that can process and respond to messages
def create_agent_node(role: Role):
    """Create an agent node for the workflow graph."""
    
    def agent_node(state: AgentState) -> AgentState:
        """Process messages and generate a response for this agent."""
        # Skip if this agent is not the intended receiver
        if role != state["receiver"]:
            return state
        
        print(f"\n{GREEN}Agent {role} is processing...{RESET}")
        
        # Create the prompt with system message and history
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_MESSAGES[role]),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Run the LLM
        messages = list(state["messages"])
        response = llm.invoke(prompt.format(messages=messages))
        
        # Print the response
        print(f"\n{BLUE_BOLD}[{role}]{RESET}: {response.content}")
        
        # Update the state with the response
        new_state = state.copy()
        new_state["messages"] = messages + [response]
        
        # Store the estimate if this is an expert providing an estimate
        if role != Role.CUSTOMER and role != Role.SCRUM_MASTER and role != Role.PRODUCT_OWNER:
            new_state["estimates"][role] = response.content
        
        # Determine the next step in the workflow
        determine_next_step(new_state, role)
        
        return new_state
    
    return agent_node

# Function to determine the next step in the workflow
def determine_next_step(state: AgentState, current_role: Role) -> None:
    """Determine the next agent and receiver based on the current role."""
    
    # Define the workflow transitions
    workflow = {
        Role.CUSTOMER: (Role.PRODUCT_OWNER, Role.SCRUM_MASTER),
        Role.PRODUCT_OWNER: (Role.SCRUM_MASTER, Role.UI_UX_DESIGNER),
        Role.UI_UX_DESIGNER: (Role.SCRUM_MASTER, Role.SOLUTION_ARCHITECT),
        Role.SOLUTION_ARCHITECT: (Role.SCRUM_MASTER, Role.FRONTEND_DEVELOPER),
        Role.FRONTEND_DEVELOPER: (Role.SCRUM_MASTER, Role.BACKEND_DEVELOPER),
        Role.BACKEND_DEVELOPER: (Role.SCRUM_MASTER, Role.RECOMMENDATION_DEVELOPER),
        Role.RECOMMENDATION_DEVELOPER: (Role.SCRUM_MASTER, Role.QA_ENGINEER),
        Role.QA_ENGINEER: (Role.SCRUM_MASTER, Role.TECHNICAL_WRITER),
        Role.TECHNICAL_WRITER: (Role.SCRUM_MASTER, Role.DEVOPS_ENGINEER),
        Role.DEVOPS_ENGINEER: (Role.SCRUM_MASTER, Role.SECURITY_ENGINEER),
        Role.SECURITY_ENGINEER: (Role.SCRUM_MASTER, Role.ECOMMERCE_SPECIALIST),
        Role.ECOMMERCE_SPECIALIST: (Role.SCRUM_MASTER, None),
        # Keep the original Developer role in the workflow for backward compatibility
        Role.DEVELOPER: (Role.SCRUM_MASTER, Role.QA_ENGINEER),
    }
    
    # Update the sender to the current role
    state["sender"] = current_role
    
    # If current role is in our workflow, set the next transitions
    if current_role in workflow:
        next_sender, next_receiver = workflow[current_role]
        state["receiver"] = next_sender
        
        # If we've reached the end of our workflow
        if next_receiver is None:
            # Time for the Scrum Master to provide a final summary
            if current_role == Role.ECOMMERCE_SPECIALIST:
                customer_message = "Please provide a final summary of the project timeline based on all the estimates collected."
                state["messages"] = state["messages"] + [HumanMessage(content=customer_message)]
                state["next_agent"] = "scrum_master"
                state["receiver"] = Role.SCRUM_MASTER
            else:
                state["done"] = True
                state["next_agent"] = "end"
        else:
            state["next_agent"] = next_sender.value
    elif current_role == Role.SCRUM_MASTER:
        # Handle Scrum Master's special role in coordinating
        if len(state["estimates"]) >= 7:  # All experts have provided estimates
            state["done"] = True
            state["next_agent"] = "end"
            state["summary"] = state["messages"][-1].content
        else:
            # Determine who the Scrum Master should talk to next based on collected estimates
            experts = [Role.UI_UX_DESIGNER, Role.SOLUTION_ARCHITECT, 
                      Role.FRONTEND_DEVELOPER, Role.BACKEND_DEVELOPER, Role.RECOMMENDATION_DEVELOPER,
                      Role.QA_ENGINEER, Role.TECHNICAL_WRITER, Role.DEVOPS_ENGINEER, 
                      Role.SECURITY_ENGINEER, Role.ECOMMERCE_SPECIALIST]
            
            for expert in experts:
                if expert.value not in state["estimates"]:
                    state["receiver"] = expert
                    state["next_agent"] = expert.value
                    
                    # Prepare a specific message for each role
                    messages = {
                        Role.UI_UX_DESIGNER: "I have received the customer's requirements from the Product Owner for the book store project. Define user stories and acceptance criteria for the project. Organize at least 10 user stories, each with a unique ID. Provide work and effort estimates based on the number of stories documented for this sprint. Please show your detailed calculation steps for the estimate.",
                        Role.SOLUTION_ARCHITECT: "The UI/UX Designer has completed the user stories for our book store application. Design the technical architecture to support these requirements, prioritizing security, scalability, and compliance. Include work and effort estimates based on the number of architectural components designed for this sprint. Please show your detailed calculation steps for the estimate.",
                        Role.FRONTEND_DEVELOPER: "The Architect has completed the design for our book store platform. Begin implementing the responsive mobile web interfaces and interactive features like book previews and shopping cart. Estimate the number of source lines of code (SLOC) and effort required for the frontend development. Please show your detailed calculation steps for the estimate.",
                        Role.BACKEND_DEVELOPER: "The Frontend Developer has started their work. Now we need APIs for book catalog, user management, and order processing. Implement the business logic for retail bookstore operations. Estimate the number of source lines of code (SLOC) and effort required for the backend development. Please show your detailed calculation steps for the estimate.",
                        Role.RECOMMENDATION_DEVELOPER: "With the frontend and backend underway, we now need to implement personalized book recommendation algorithms and user behavior tracking for relevant suggestions. Estimate the number of source lines of code (SLOC) and effort required for the recommendation system. Please show your detailed calculation steps for the estimate.",
                        Role.DEVELOPER: "The Architect has completed the design for our book store platform. Begin implementing the features based on the user stories and architectural components. Estimate the number of source lines of code (SLOC) and effort required for this sprint's development. Please show your detailed calculation steps for the estimate.",
                        Role.QA_ENGINEER: "The development phase is complete for our book store application. Create and execute test cases based on user stories. Provide work and effort estimates based on the number of test cases created and executed in this sprint. Please show your detailed calculation steps for the estimate.",
                        Role.TECHNICAL_WRITER: "Testing is complete for the book store platform. Prepare the user documentation and training materials based on the deliverables of this sprint. Provide work and effort estimates for documentation creation. Please show your detailed calculation steps for the estimate.",
                        Role.DEVOPS_ENGINEER: "Documentation is complete for the book store platform. Set up the CI/CD pipeline, infrastructure, and deployment automation. Provide work and effort estimates for DevOps setup and automation. Please show your detailed calculation steps for the estimate.",
                        Role.SECURITY_ENGINEER: "The CI/CD pipeline is set up for the book store platform. Conduct security reviews, implement security measures, and secure sensitive data. Provide work and effort estimates for security implementation. Please show your detailed calculation steps for the estimate.",
                        Role.ECOMMERCE_SPECIALIST: "The book store platform development is near completion. Provide best practices for book cataloging, checkout UX, and promotions. Provide work and effort estimates for implementing these best practices. Please show your detailed calculation steps for the estimate."
                    }
                    
                    if expert in messages:
                        state["messages"] = state["messages"] + [HumanMessage(content=messages[expert])]
                    
                    break
    else:
        # Default to ending the workflow if we don't know what's next
        state["done"] = True
        state["next_agent"] = "end"

# Function to check if the workflow is complete
def should_end(state: AgentState) -> str:
    """Determine if the workflow should continue or end."""
    if state["done"]:
        return "end"
    return state["next_agent"]

# Function to run the workflow simulation
def run_simulation():
    """Run the book store project simulation using LangGraph."""
    global llm  # Use the global llm variable
    print(f"\n{GREEN}Running Book Store Project Simulation with LangGraph{RESET}")
    
    # Create nodes for each agent
    nodes = {}
    for role in Role:
        nodes[role.value] = create_agent_node(role)
    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes to the graph
    for role in Role:
        workflow.add_node(role.value, nodes[role.value])
    
    # Add START node connected to customer
    workflow.add_edge(START, "customer")
    
    # Add conditional edge for routing from customer
    workflow.add_conditional_edges(
        "customer",
        should_end,
        {
            "product_owner": "product_owner",
            "scrum_master": "scrum_master",
            "ui_ux_designer": "ui_ux_designer",
            "solution_architect": "solution_architect",
            "developer": "developer",
            "qa_engineer": "qa_engineer",
            "technical_writer": "technical_writer",
            "devops_engineer": "devops_engineer",
            "security_engineer": "security_engineer",
            "ecommerce_specialist": "ecommerce_specialist",
            "end": END
        }
    )
    
    # Set up the conditional edges for the rest of the roles
    for role in Role:
        if role != Role.CUSTOMER:
            workflow.add_conditional_edges(
                role.value,
                should_end,
                {
                    "product_owner": "product_owner",
                    "scrum_master": "scrum_master",
                    "ui_ux_designer": "ui_ux_designer",
                    "solution_architect": "solution_architect",
                    "developer": "developer",
                    "frontend_developer": "frontend_developer",
                    "backend_developer": "backend_developer",
                    "recommendation_developer": "recommendation_developer",
                    "qa_engineer": "qa_engineer",
                    "technical_writer": "technical_writer",
                    "devops_engineer": "devops_engineer",
                    "security_engineer": "security_engineer",
                    "ecommerce_specialist": "ecommerce_specialist",
                    "end": END
                }
            )
    
    # Visualize the LangGraph workflow
    try:
        visualize_langgraph_workflow(workflow)
    except Exception as e:
        print(f"\n{GREEN}Could not visualize LangGraph workflow: {e}{RESET}")
    
    # Compile the graph
    app = workflow.compile()
    
    # Initialize the state
    state = get_initial_state()
    
    # Add the initial customer message
    customer_message = """I want to build a web-based mobile app for our bookstore where customers can browse books by genre, read previews, purchase books online, track their shipments, review books, and get personalized reading recommendations."""
    
    state["messages"] = [HumanMessage(content=customer_message)]
    
    # Run the workflow
    final_state = app.invoke(state)
    
    # Print the final summary
    if "summary" in final_state and final_state["summary"]:
        print(f"\n{GREEN}Final Project Summary:{RESET}")
        print(f"\n{BLUE_BOLD}[Scrum Master - Final Project Summary]{RESET}: {final_state['summary']}")
    
    # Generate workflow flowchart
    generate_workflow_flowchart()
    
    print(f"\n{GREEN}Book Store Project Simulation Complete!{RESET}")
    return final_state

def generate_workflow_flowchart():
    """
    Generate a flowchart visualization of the agent workflow 
    """
    print(f"\n{GREEN}Generating Agent Workflow Flowchart...{RESET}")
    
    # Try graphviz first if available
    if HAS_GRAPHVIZ:
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
            dot.node('Frontend_Developer', 'Frontend Developer', fillcolor='#FFA07A')
            dot.node('Backend_Developer', 'Backend Developer', fillcolor='#87CEFA')
            dot.node('Recommendation_Developer', 'Recommendation Developer', fillcolor='#98FB98')
            dot.node('QA_Engineer', 'QA Engineer', fillcolor='#E6E6FA')
            dot.node('Technical_Writer', 'Technical Writer', fillcolor='#F0FFF0')
            
            # Add new specialized roles
            dot.node('DevOps_Engineer', 'DevOps Engineer', fillcolor='#FFD700')
            dot.node('Security_Engineer', 'Security Engineer', fillcolor='#FF6347')
            dot.node('E_commerce_Specialist', 'E-commerce Specialist', fillcolor='#9370DB')
            
            # Add edges (interaction flow)
            dot.edge('Customer', 'Product_Owner', label='Requirements')
            dot.edge('Product_Owner', 'Scrum_Master', label='Project\nRequirements')
            dot.edge('Scrum_Master', 'UI_UX_Designer', label='Define User\nStories')
            dot.edge('UI_UX_Designer', 'Scrum_Master', label='User Stories\n& Estimates')
            dot.edge('Scrum_Master', 'Solution_Architect', label='Design\nArchitecture')
            dot.edge('Solution_Architect', 'Scrum_Master', label='Architecture\n& Estimates')
            dot.edge('Scrum_Master', 'Frontend_Developer', label='Implement\nFrontend')
            dot.edge('Frontend_Developer', 'Scrum_Master', label='Frontend\n& Estimates')
            dot.edge('Scrum_Master', 'Backend_Developer', label='Implement\nBackend')
            dot.edge('Backend_Developer', 'Scrum_Master', label='Backend\n& Estimates')
            dot.edge('Scrum_Master', 'Recommendation_Developer', label='Implement\nRecommendations')
            dot.edge('Recommendation_Developer', 'Scrum_Master', label='Recommendation\n& Estimates')
            dot.edge('Scrum_Master', 'QA_Engineer', label='Test\nImplementation')
            dot.edge('QA_Engineer', 'Scrum_Master', label='Testing\n& Estimates')
            dot.edge('Scrum_Master', 'Technical_Writer', label='Create\nDocumentation')
            dot.edge('Technical_Writer', 'Scrum_Master', label='Documentation\n& Estimates')
            
            # Add edges for new specialized roles
            dot.edge('Scrum_Master', 'DevOps_Engineer', label='Setup\nCI/CD')
            dot.edge('DevOps_Engineer', 'Scrum_Master', label='DevOps\n& Estimates')
            dot.edge('Scrum_Master', 'Security_Engineer', label='Implement\nSecurity')
            dot.edge('Security_Engineer', 'Scrum_Master', label='Security\n& Estimates')
            dot.edge('Scrum_Master', 'E_commerce_Specialist', label='Provide\nBest Practices')
            dot.edge('E_commerce_Specialist', 'Scrum_Master', label='Domain Expertise\n& Estimates')
            
            # Save flowchart to a file
            flowchart_filename = 'book_store_workflow_langgraph'
            try:
                dot.render(flowchart_filename, format='png', view=False)
                print(f"\n{GREEN}Flowchart generated as '{flowchart_filename}.png'{RESET}")
                print(f"{GREEN}You can view it in the same directory as this script.{RESET}")
                return True
            except Exception as render_error:
                print(f"\n{GREEN}Could not render Graphviz flowchart: {render_error}{RESET}")
                print(f"\n{GREEN}Trying to save just the DOT file...{RESET}")
                try:
                    with open(f"{flowchart_filename}.dot", "w") as f:
                        f.write(dot.source)
                    print(f"{GREEN}DOT file saved to '{flowchart_filename}.dot'{RESET}")
                    print(f"{GREEN}You can render it using: dot -Tpng {flowchart_filename}.dot -o {flowchart_filename}.png{RESET}")
                    return True
                except Exception as write_error:
                    print(f"{GREEN}Could not save DOT file: {write_error}{RESET}")
        except Exception as e:
            print(f"\n{GREEN}Error with Graphviz: {e}{RESET}")
            print(f"\n{GREEN}Trying alternative flowchart method...{RESET}")
    
    # Try matplotlib/networkx as an alternative
    if HAS_MATPLOTLIB:
        try:
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes (agents)
            roles = {
                'Customer': {'color': '#FFCCCB', 'pos': (0, 8)},
                'Product Owner': {'color': '#ADD8E6', 'pos': (0, 7)},
                'Scrum Master': {'color': '#90EE90', 'pos': (0, 6)},
                'UI/UX Designer': {'color': '#FFFACD', 'pos': (2, 5)},
                'Solution Architect': {'color': '#D8BFD8', 'pos': (2, 4)},
                'Frontend Developer': {'color': '#FFA07A', 'pos': (2, 3)},
                'Backend Developer': {'color': '#87CEFA', 'pos': (2, 2)},
                'Recommendation Developer': {'color': '#98FB98', 'pos': (2, 1)},
                'QA Engineer': {'color': '#E6E6FA', 'pos': (2, 0)},
                'Technical Writer': {'color': '#F0FFF0', 'pos': (2, -1)},
                'DevOps Engineer': {'color': '#FFD700', 'pos': (2, -2)},
                'Security Engineer': {'color': '#FF6347', 'pos': (2, -3)},
                'E-commerce Specialist': {'color': '#9370DB', 'pos': (2, -4)}
            }
            
            # Add nodes
            for role, attrs in roles.items():
                G.add_node(role, color=attrs['color'], pos=attrs['pos'])
            
            # Add edges with labels
            edges = [
                ('Customer', 'Product Owner', {'label': 'Requirements'}),
                ('Product Owner', 'Scrum Master', {'label': 'Project Requirements'}),
                ('Scrum Master', 'UI/UX Designer', {'label': 'Define User Stories'}),
                ('UI/UX Designer', 'Scrum Master', {'label': 'User Stories & Estimates'}),
                ('Scrum Master', 'Solution Architect', {'label': 'Design Architecture'}),
                ('Solution Architect', 'Scrum Master', {'label': 'Architecture & Estimates'}),
                ('Scrum Master', 'Frontend Developer', {'label': 'Implement Frontend'}),
                ('Frontend Developer', 'Scrum Master', {'label': 'Frontend & Estimates'}),
                ('Scrum Master', 'Backend Developer', {'label': 'Implement Backend'}),
                ('Backend Developer', 'Scrum Master', {'label': 'Backend & Estimates'}),
                ('Scrum Master', 'Recommendation Developer', {'label': 'Implement Recommendations'}),
                ('Recommendation Developer', 'Scrum Master', {'label': 'Recommendation & Estimates'}),
                ('Scrum Master', 'QA Engineer', {'label': 'Test Implementation'}),
                ('QA Engineer', 'Scrum Master', {'label': 'Testing & Estimates'}),
                ('Scrum Master', 'Technical Writer', {'label': 'Create Documentation'}),
                ('Technical Writer', 'Scrum Master', {'label': 'Documentation & Estimates'}),
                ('Scrum Master', 'DevOps Engineer', {'label': 'Setup CI/CD'}),
                ('DevOps Engineer', 'Scrum Master', {'label': 'DevOps & Estimates'}),
                ('Scrum Master', 'Security Engineer', {'label': 'Implement Security'}),
                ('Security Engineer', 'Scrum Master', {'label': 'Security & Estimates'}),
                ('Scrum Master', 'E-commerce Specialist', {'label': 'Provide Best Practices'}),
                ('E-commerce Specialist', 'Scrum Master', {'label': 'Domain Expertise & Estimates'})
            ]
            
            G.add_edges_from((u, v, d) for u, v, d in edges)
            
            # Set up the plot
            plt.figure(figsize=(12, 10))
            pos = nx.get_node_attributes(G, 'pos')
            
            # Draw nodes
            node_colors = [data['color'] for node, data in G.nodes(data=True)]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray', 
                                  connectionstyle='arc3,rad=0.1', arrowsize=20)
            
            # Draw node labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
            
            # Draw edge labels
            edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            
            # Set title
            plt.title('Book Store Project Workflow', size=15)
            plt.axis('off')
            
            # Save the figure
            flowchart_filename = 'book_store_workflow_matplotlib_langgraph.png'
            plt.savefig(flowchart_filename, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n{GREEN}Matplotlib flowchart generated as '{flowchart_filename}'{RESET}")
            print(f"{GREEN}You can view it in the same directory as this script.{RESET}")
            return True
            
        except Exception as e:
            print(f"\n{GREEN}Error with Matplotlib/NetworkX: {e}{RESET}")
    
    # If we reached here, neither method worked
    print_ascii_workflow()
    return False

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
    |Stories| | |o| | | | | | | | | |              
    |      |───|u|───|e|───|E|───|h| |              
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
    ► Scrum Master → Frontend Developer: Requests frontend implementation
    ► Frontend Developer → Scrum Master: Delivers frontend implementation & estimates
    ► Scrum Master → Backend Developer: Requests backend implementation
    ► Backend Developer → Scrum Master: Delivers backend implementation & estimates
    ► Scrum Master → Recommendation Developer: Requests recommendation system
    ► Recommendation Developer → Scrum Master: Delivers recommendation system & estimates
    ► Scrum Master → QA Engineer: Requests testing
    ► QA Engineer → Scrum Master: Delivers testing results & estimates
    ► Scrum Master → Technical Writer: Requests documentation
    ► Technical Writer → Scrum Master: Delivers documentation & estimates
    ► Scrum Master → DevOps Engineer: Requests CI/CD setup
    ► DevOps Engineer → Scrum Master: Delivers DevOps setup & estimates
    ► Scrum Master → Security Engineer: Requests security implementation
    ► Security Engineer → Scrum Master: Delivers security measures & estimates
    ► Scrum Master → E-commerce Specialist: Requests domain expertise
    ► E-commerce Specialist → Scrum Master: Delivers best practices & estimates
    ► Scrum Master: Provides final project summary
    """)

def visualize_langgraph_workflow(workflow):
    """Visualize the LangGraph workflow using LangGraph's built-in visualization."""
    try:
        from IPython.display import display
        print(f"\n{GREEN}Generating LangGraph Workflow Visualization...{RESET}")
        
        # Try to create and save the graph visualization directly as PNG
        try:
            graph_visualization_file = "langgraph_workflow.png"
            # Check which method is available for saving the graph
            if hasattr(workflow, "save_graph"):
                workflow.save_graph(graph_visualization_file, format="png")
                print(f"\n{GREEN}LangGraph workflow visualization saved to '{graph_visualization_file}'{RESET}")
                return
            
            # If save_graph doesn't work, try other methods
            if hasattr(workflow, "to_graph") or hasattr(workflow, "get_graph"):
                # Create a temporary DOT file
                import tempfile
                import subprocess
                
                with tempfile.NamedTemporaryFile(suffix='.dot', delete=False) as tmp:
                    tmp_filename = tmp.name
                    # Use the appropriate method to get the DOT content
                    if hasattr(workflow, "to_graph"):
                        tmp.write(workflow.to_graph().encode('utf-8'))
                    else:  # has get_graph
                        tmp.write(workflow.get_graph().encode('utf-8'))
                
                # Convert DOT to PNG using graphviz if available
                if HAS_GRAPHVIZ:
                    from graphviz import Source
                    graph = Source(open(tmp_filename).read())
                    graph.format = 'png'
                    graph.render(filename='langgraph_workflow', cleanup=True)
                    print(f"\n{GREEN}LangGraph workflow visualization saved to '{graph_visualization_file}'{RESET}")
                else:
                    # Try to use command-line graphviz if available
                    try:
                        subprocess.run(['dot', '-Tpng', tmp_filename, '-o', graph_visualization_file], 
                                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        print(f"\n{GREEN}LangGraph workflow visualization saved to '{graph_visualization_file}'{RESET}")
                    except (subprocess.SubprocessError, FileNotFoundError):
                        print(f"\n{GREEN}Could not convert DOT to PNG. Please install Graphviz.{RESET}")
                        print(f"{GREEN}DOT file saved to '{tmp_filename}'{RESET}")
                        # Use our alternative visualization
                        generate_workflow_flowchart()
                
                # Try to clean up the temp file
                try:
                    import os
                    os.unlink(tmp_filename)
                except:
                    pass
            else:
                print(f"\n{GREEN}This version of LangGraph doesn't support graph visualization export{RESET}")
                # Fall back to our custom visualization
                generate_workflow_flowchart()
                
        except Exception as e:
            print(f"\n{GREEN}Could not save LangGraph visualization: {str(e)}{RESET}")
            # Fall back to our custom visualization
            generate_workflow_flowchart()
                
    except ImportError:
        print(f"\n{GREEN}IPython not available, skipping LangGraph visualization{RESET}")
        # Fall back to our custom visualization
        generate_workflow_flowchart()
    except Exception as e:
        print(f"\n{GREEN}Error in LangGraph visualization: {e}{RESET}")
        # Fall back to our custom visualization
        generate_workflow_flowchart()

# Make sure the file is executable
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Book Store Project Simulation using LangGraph')
    parser.add_argument('--api-key', type=str, help='OpenAI API key to use')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use (default: gpt-4o-mini)')
    parser.add_argument('--debug', action='store_true', help='Show debug information')
    args = parser.parse_args()

    # Use command line API key if provided
    if args.api_key:
        api_key = args.api_key
        os.environ["OPENAI_API_KEY"] = api_key
        
    # Update model if provided
    if args.model:
        model_name = args.model
        
    # Initialize the LLM
    llm = initialize_llm(api_key, model_name)
    if llm is None:
        print(f"\n{GREEN}Exiting due to LLM initialization failure.{RESET}")
        exit(1)
        
    try:
        print(f"\n{GREEN}Starting Book Store Project Simulation with LangGraph{RESET}")
        print(f"{GREEN}Using model: {model_name}{RESET}")
        run_simulation()
    except KeyboardInterrupt:
        print(f"\n{GREEN}Simulation interrupted by user.{RESET}")
    except Exception as e:
        print(f"\n{GREEN}An error occurred: {e}{RESET}")
        if args.debug:
            import traceback
            traceback.print_exc() 
