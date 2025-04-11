from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Tuple, Any
import os
import time

# ANSI escape code for formatting
GREEN = "\033[92;1m"
BLUE_BOLD = "\033[94;1m"
RESET = "\033[0m"

# Initialize the LLM
model_name = "gpt-4o-mini"
api_key = ""  # Your API key

# Set environment variable
os.environ["OPENAI_API_KEY"] = api_key


# Initialize the LLM
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
        
        # Get response from LLM
        response = llm(messages)
        
        # Store the exchange in memory
        self.memory.append({"role": "human", "sender": sender_name, "content": message})
        self.memory.append({"role": "ai", "content": response.content})
        
        return response.content
    
    def initiate_chat(self, recipient, message: str):
        """Start a conversation with another agent"""
        print(f"\n{BLUE_BOLD}[{self.name}]{RESET} to {BLUE_BOLD}[{recipient.name}]{RESET}: {message}")
        recipient_response = recipient.send_message(message, self.name)
        print(f"\n{BLUE_BOLD}[{recipient.name}]{RESET}: {recipient_response}")
        return recipient_response

# Create all agents with the same system messages from the original code
product_owner_agent = Agent(
    name="Product_Owner",
    system_message="""Represents the customer's needs, manages the product backlog, and prioritizes features for the book store platform.
                      Example tasks include defining shopping cart features, payment integrations, and user profile enhancements.
                      IMPORTANT: When responding, always show your work in this format:
                      Estimated Weeks Required:
                      - Total Features / Productivity = Total Duration
                      - e.g., 6 features / 3 features per week = 2 weeks
                    """
)

scrum_master_agent = Agent(
    name="Scrum_Master",
    system_message="""Facilitates Scrum ceremonies, removes obstacles, and ensures team adherence to Agile principles for the book store platform.
                      Supports daily stand-ups, sprint planning, and retrospectives.
                      IMPORTANT: When responding, always show your work in this format:
                      Estimated Days Required**:
                      - Total Ceremonies / Productivity = Total Duration
                      - e.g., 4 ceremonies / 1 ceremony per day = 4 days
                    """
)

ui_ux_designer_agent = Agent(
    name="UI_UX_Designer",
    system_message="""Designs user interfaces and experiences for the book store platform.
                      Tasks include wireframes, prototypes, and mobile interfaces.
                      IMPORTANT: When responding, always show your work in this format:
                      Estimated Weeks Required**:
                      - Total Screens / Productivity = Total Duration
                      - e.g., 9 screens / 3 screens per week = 3 weeks
                    """
)

solution_architect_agent = Agent(
    name="Solution_Architect",
    system_message="""Designs the system architecture for the book store platform including microservices and integrations.
                      IMPORTANT: When responding, always show your work in this format:
                      Estimated Weeks Required**:
                      - Total Components / Productivity = Total Duration
                      - e.g., 4 components / 1 per week = 4 weeks
                    """
)

developer_agent = Agent(
    name="Developer",
    system_message="""Develops features, integrates APIs, and manages frontend/backend logic for the book store platform.
                      IMPORTANT: When responding, always show your work in this format:
                      Estimated Weeks Required**:
                      - Total SLOC / Productivity = Total Duration
                      - e.g., 1000 SLOC / 500 SLOC per week = 2 weeks
                    """
)

qa_engineer_agent = Agent(
    name="QA_Engineer",
    system_message="""Tests features and validates functionalities for the book store platform.
                      IMPORTANT: When responding, always show your work in this format:
                      Estimated Days Required**:
                      - Total Test Cases / Productivity = Total Duration
                      - e.g., 25 test cases / 5 per day = 5 days
                    """
)

technical_writer_agent = Agent(
    name="Technical_Writer",
    system_message="""Writes user guides, API docs, and release notes for the book store platform.
                      IMPORTANT: When responding, always show your work in this format:
                      Estimated Weeks Required**:
                      - Total Pages / Productivity = Total Duration
                      - e.g., 8 pages / 4 pages per week = 2 weeks
                    """
)

devops_engineer_agent = Agent(
    name="DevOps_Engineer",
    system_message="""Handles CI/CD, infrastructure, and deployment automation for the book store platform.
                      IMPORTANT: When responding, always show your work in this format:
                      Estimated Weeks Required**:
                      - Total Tasks / Productivity = Total Duration
                      - e.g., 6 tasks / 2 tasks per week = 3 weeks
                    """
)

security_engineer_agent = Agent(
    name="Security_Engineer",
    system_message="""Conducts code reviews, penetration testing, and secures sensitive data for the book store platform.
                      IMPORTANT: When responding, always show your work in this format:
                      Estimated Weeks Required**:
                      - Total Security Tasks / Productivity = Total Duration
                      - e.g., 3 tasks / 1 task per week = 3 weeks
                    """
)

bookstore_specialist_agent = Agent(
    name="Bookstore_Specialist",
    system_message="""Provides best practices in book cataloging, checkout UX, and promotions for the book store platform.
                      IMPORTANT: When responding, always show your work in this format:
                      Estimated Weeks Required**:
                      - Total Areas / Productivity = Total Duration
                      - e.g., 6 areas / 2 per week = 3 weeks
                    """
)

# List of agents for easy reference
bookstore_agents = [
    product_owner_agent, scrum_master_agent, ui_ux_designer_agent,
    solution_architect_agent, developer_agent, qa_engineer_agent,
    technical_writer_agent, devops_engineer_agent, security_engineer_agent,
    bookstore_specialist_agent
]

# Define the same message templates as in the original code
product_owner_message = (
    "I want to build a web-based mobile app for our bookstore where customers can browse books by genre, read previews, "
    "purchase books online, track their shipments, review books, and get personalized reading recommendations. "
    "Please provide your estimate including the detailed calculation steps for the refinement effort."
)

scrum_master_to_business_analyst_prompt = (
    "I have received the customer's requirements from the Product Owner. Define user stories and acceptance criteria for the project. "
    "Organize at least 10 user stories, each with a unique ID (e.g., US-01, US-02). "
    "Provide work and effort estimates based on the number of stories documented for this sprint. "
    "Please show your detailed calculation steps for the estimate."
)

business_analyst_to_scrum_master_response = (
    "I have documented the user stories with acceptance criteria, as requested. "
    "Here are my detailed calculation steps for the effort estimates based on the number of stories documented for this sprint:"
)

scrum_master_to_architect_prompt = (
    "The Business Analyst has completed the user stories. Design the technical architecture to support these requirements, prioritizing security, scalability, and compliance. "
    "Include work and effort estimates based on the number of architectural components designed for this sprint. "
    "Please show your detailed calculation steps for the estimate."
)

architect_to_scrum_master_response = (
    "I have completed the architectural design. Here are my detailed calculation steps for the effort estimates for the design phase:"
)

scrum_master_to_developer_prompt = (
    "The Architect has completed the design. Begin implementing the features based on the user stories and architectural components. "
    "Estimate the number of source lines of code (SLOC) and effort required for this sprint's development. "
    "Please show your detailed calculation steps for the estimate."
)

developer_to_scrum_master_response = (
    "I have developed the features based on the architecture and user stories. "
    "Here are my detailed calculation steps for the development phase effort estimates:"
)

scrum_master_to_qa_engineer_prompt = (
    "The development phase is complete. Create and execute test cases based on user stories. "
    "Provide work and effort estimates based on the number of test cases created and executed in this sprint. "
    "Please show your detailed calculation steps for the estimate."
)

qa_engineer_to_scrum_master_response = (
    "Testing is complete, and I have verified that the functionality meets the requirements. "
    "Here are my detailed calculation steps for the testing effort estimates:"
)

scrum_master_to_technical_writer_prompt = (
    "Testing is complete. Prepare the user documentation and training materials based on the deliverables of this sprint. "
    "Provide work and effort estimates for documentation creation. "
    "Please show your detailed calculation steps for the estimate."
)

technical_writer_to_scrum_master_response = (
    "Documentation and training materials are complete. Here are my detailed calculation steps for the documentation effort estimates:"
)

# Create a list of agent pairs to define the conversation flow in Scrum
conversation_flow_scrum = [
    (product_owner_agent, scrum_master_agent, product_owner_message),
    (scrum_master_agent, ui_ux_designer_agent, scrum_master_to_business_analyst_prompt),
    (ui_ux_designer_agent, scrum_master_agent, business_analyst_to_scrum_master_response),
    (scrum_master_agent, solution_architect_agent, scrum_master_to_architect_prompt),
    (solution_architect_agent, scrum_master_agent, architect_to_scrum_master_response),
    (scrum_master_agent, developer_agent, scrum_master_to_developer_prompt),
    (developer_agent, scrum_master_agent, developer_to_scrum_master_response),
    (scrum_master_agent, qa_engineer_agent, scrum_master_to_qa_engineer_prompt),
    (qa_engineer_agent, scrum_master_agent, qa_engineer_to_scrum_master_response),
    (scrum_master_agent, technical_writer_agent, scrum_master_to_technical_writer_prompt),
    (technical_writer_agent, scrum_master_agent, technical_writer_to_scrum_master_response)
]

class GroupChat:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.messages = []
    
    def add_message(self, sender_agent: Agent, message: str):
        """Add a message to the group chat history"""
        self.messages.append({
            "sender": sender_agent.name,
            "content": message
        })
        print(f"\n{BLUE_BOLD}[{sender_agent.name}]{RESET} to Group: {message}")
    
    def broadcast_message(self, sender_agent: Agent, message: str):
        """Send a message to all agents except the sender"""
        self.add_message(sender_agent, message)
        responses = {}
        
        for agent in self.agents:
            if agent != sender_agent:
                # Wait briefly to avoid rate limits
                time.sleep(1)
                response = agent.send_message(message, sender_agent.name)
                responses[agent.name] = response
                print(f"\n{BLUE_BOLD}[{agent.name}]{RESET}: {response}")
        
        return responses

class GroupChatManager:
    def __init__(self, groupchat: GroupChat):
        self.groupchat = groupchat
    
    def initiate_chat(self, agent: Agent, recipient: Agent = None, message: str = None):
        """Start a conversation in the group chat or between specific agents"""
        if recipient:
            # Direct conversation between two agents
            return agent.initiate_chat(recipient, message)
        else:
            # Group conversation
            return self.groupchat.broadcast_message(agent, message)

# Initialize GroupChat for Scrum simulation
groupchat_scrum = GroupChat(
    agents=[
        product_owner_agent,
        scrum_master_agent,
        ui_ux_designer_agent,
        solution_architect_agent,
        developer_agent,
        qa_engineer_agent,
        technical_writer_agent
    ]
)

manager_scrum = GroupChatManager(groupchat=groupchat_scrum)

# Define the initial customer message
customer_message = """I want to build a web-based mobile app for our bookstore where customers can browse books by genre, read previews, purchase books online, track their shipments, review books, and get personalized reading recommendations."""

# Run the simulation like in the original code
def run_simulation():
    print(f"\n{GREEN}Running Book Store Project Simulation with LangChain{RESET}")
    
    # First, the product owner initiates a chat with the scrum master
    print(f"\n{GREEN}Step 1: Product Owner initiates chat with Scrum Master{RESET}")
    product_owner_agent.initiate_chat(
        scrum_master_agent,
        customer_message
    )
    
    # The scrum master initiates a chat with the UI/UX designer (playing BA role)
    print(f"\n{GREEN}Step 2: Scrum Master initiates chat with UI/UX Designer{RESET}")
    scrum_master_agent.initiate_chat(
        ui_ux_designer_agent,
        scrum_master_to_business_analyst_prompt
    )
    
    # Continue with the rest of the conversation flow
    print(f"\n{GREEN}Step 3: UI/UX Designer responds to Scrum Master{RESET}")
    ui_ux_designer_agent.initiate_chat(
        scrum_master_agent,
        business_analyst_to_scrum_master_response
    )
    
    print(f"\n{GREEN}Step 4: Scrum Master discusses with Solution Architect{RESET}")
    scrum_master_agent.initiate_chat(
        solution_architect_agent,
        scrum_master_to_architect_prompt
    )
    
    print(f"\n{GREEN}Step 5: Solution Architect responds to Scrum Master{RESET}")
    solution_architect_agent.initiate_chat(
        scrum_master_agent,
        architect_to_scrum_master_response
    )
    
    print(f"\n{GREEN}Step 6: Scrum Master discusses with Developer{RESET}")
    scrum_master_agent.initiate_chat(
        developer_agent,
        scrum_master_to_developer_prompt
    )
    
    print(f"\n{GREEN}Step 7: Developer responds to Scrum Master{RESET}")
    developer_agent.initiate_chat(
        scrum_master_agent,
        developer_to_scrum_master_response
    )
    
    print(f"\n{GREEN}Step 8: Scrum Master discusses with QA Engineer{RESET}")
    scrum_master_agent.initiate_chat(
        qa_engineer_agent,
        scrum_master_to_qa_engineer_prompt
    )
    
    print(f"\n{GREEN}Step 9: QA Engineer responds to Scrum Master{RESET}")
    qa_engineer_agent.initiate_chat(
        scrum_master_agent,
        qa_engineer_to_scrum_master_response
    )
    
    print(f"\n{GREEN}Step 10: Scrum Master discusses with Technical Writer{RESET}")
    scrum_master_agent.initiate_chat(
        technical_writer_agent,
        scrum_master_to_technical_writer_prompt
    )
    
    print(f"\n{GREEN}Step 11: Technical Writer responds to Scrum Master{RESET}")
    technical_writer_agent.initiate_chat(
        scrum_master_agent,
        technical_writer_to_scrum_master_response
    )
    
    print(f"\n{GREEN}Book Store Project Simulation Complete!{RESET}")

# Run the simulation
if __name__ == "__main__":
    run_simulation() 
