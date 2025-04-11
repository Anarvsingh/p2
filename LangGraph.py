from langgraph.graph import StateGraph, END
from typing import Dict, List, TypedDict
import random

# ANSI escape codes for colored output
GREEN = "\033[92;1m"
BLUE_BOLD = "\033[94;1m"
RESET = "\033[0m"

# Define the state structure to hold data across nodes
class ScrumState(TypedDict):
    customer_request: str
    user_stories: List[Dict]
    sprint_tasks: List[Dict]

# Node functions for each role, mirroring the AutoGen agents

def product_owner_node(state: ScrumState) -> ScrumState:
    """Simulates Product Owner defining user stories and calculating effort."""
    user_stories = [
        {"id": f"US-{i:02d}", "story": f"User story {i} for bookstore app", "priority": "High", "story_points": random.randint(1, 8)}
        for i in range(1, 11)  # Generate 10 user stories
    ]
    total_features = len(user_stories)
    productivity = 3  # features per week
    duration = total_features / productivity
    print(f"{GREEN}Product Owner Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Features / Productivity = Total Duration")
    print(f"- {total_features} features / {productivity} features per week = {duration:.2f} weeks")
    state["user_stories"] = user_stories
    return state

def ui_ux_designer_node(state: ScrumState) -> ScrumState:
    """Simulates UI/UX Designer generating design tasks and calculating effort."""
    total_screens = 3 * len(state["user_stories"])  # 3 screens per user story
    productivity = 3  # screens per week
    duration = total_screens / productivity
    print(f"{GREEN}UI/UX Designer Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Screens / Productivity = Total Duration")
    print(f"- {total_screens} screens / {productivity} screens per week = {duration:.2f} weeks")
    for story in state["user_stories"]:
        task = {
            "task_id": f"SPRINT1-TASK-{len(state['sprint_tasks']) + 1:03d}",
            "story_id": story["id"],
            "description": f"Design UI for {story['story']}",
            "role": "UI_UX_Designer",
            "effort_hours": 20
        }
        state["sprint_tasks"].append(task)
    return state

def solution_architect_node(state: ScrumState) -> ScrumState:
    """Simulates Solution Architect designing architecture and calculating effort."""
    total_components = 2 * len(state["user_stories"])  # 2 components per user story
    productivity = 1  # component per week
    duration = total_components / productivity
    print(f"{GREEN}Solution Architect Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Components / Productivity = Total Duration")
    print(f"- {total_components} components / {productivity} component per week = {duration:.2f} weeks")
    for story in state["user_stories"]:
        task = {
            "task_id": f"SPRINT1-TASK-{len(state['sprint_tasks']) + 1:03d}",
            "story_id": story["id"],
            "description": f"Define architecture for {story['story']}",
            "role": "Solution_Architect",
            "effort_hours": 15
        }
        state["sprint_tasks"].append(task)
    return state

def developer_node(state: ScrumState) -> ScrumState:
    """Simulates Developer implementing features and calculating effort."""
    total_points = sum(story["story_points"] for story in state["user_stories"])
    sloc_per_point = 100
    total_sloc = total_points * sloc_per_point
    productivity = 500  # SLOC per week
    duration = total_sloc / productivity
    print(f"{GREEN}Developer Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total SLOC / Productivity = Total Duration")
    print(f"- {total_sloc} SLOC / {productivity} SLOC per week = {duration:.2f} weeks")
    for story in state["user_stories"]:
        effort_hours = 8 * story["story_points"]  # 8 hours per story point
        task = {
            "task_id": f"SPRINT1-TASK-{len(state['sprint_tasks']) + 1:03d}",
            "story_id": story["id"],
            "description": f"Develop feature for {story['story']}",
            "role": "Developer",
            "effort_hours": effort_hours
        }
        state["sprint_tasks"].append(task)
    return state

def qa_engineer_node(state: ScrumState) -> ScrumState:
    """Simulates QA Engineer testing features and calculating effort."""
    total_test_cases = 5 * len(state["user_stories"])  # 5 test cases per user story
    productivity = 5  # test cases per day
    duration_days = total_test_cases / productivity
    print(f"{GREEN}QA Engineer Estimate:{RESET}")
    print(f"Estimated Days Required:")
    print(f"- Total Test Cases / Productivity = Total Duration")
    print(f"- {total_test_cases} test cases / {productivity} test cases per day = {duration_days:.2f} days")
    for story in state["user_stories"]:
        task = {
            "task_id": f"SPRINT1-TASK-{len(state['sprint_tasks']) + 1:03d}",
            "story_id": story["id"],
            "description": f"Test feature for {story['story']}",
            "role": "QA_Engineer",
            "effort_hours": 20
        }
        state["sprint_tasks"].append(task)
    return state

def technical_writer_node(state: ScrumState) -> ScrumState:
    """Simulates Technical Writer creating documentation and calculating effort."""
    total_pages = 4 * len(state["user_stories"])  # 4 pages per user story
    productivity = 4  # pages per week
    duration = total_pages / productivity
    print(f"{GREEN}Technical Writer Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Pages / Productivity = Total Duration")
    print(f"- {total_pages} pages / {productivity} pages per week = {duration:.2f} weeks")
    for story in state["user_stories"]:
        task = {
            "task_id": f"SPRINT1-TASK-{len(state['sprint_tasks']) + 1:03d}",
            "story_id": story["id"],
            "description": f"Write documentation for {story['story']}",
            "role": "Technical_Writer",
            "effort_hours": 10
        }
        state["sprint_tasks"].append(task)
    return state

def devops_engineer_node(state: ScrumState) -> ScrumState:
    """Simulates DevOps Engineer setting up infrastructure and calculating effort."""
    total_tasks = 2 * len(state["user_stories"])  # 2 tasks per user story
    productivity = 2  # tasks per week
    duration = total_tasks / productivity
    print(f"{GREEN}DevOps Engineer Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Tasks / Productivity = Total Duration")
    print(f"- {total_tasks} tasks / {productivity} tasks per week = {duration:.2f} weeks")
    for story in state["user_stories"]:
        task = {
            "task_id": f"SPRINT1-TASK-{len(state['sprint_tasks']) + 1:03d}",
            "story_id": story["id"],
            "description": f"Set up infrastructure for {story['story']}",
            "role": "DevOps_Engineer",
            "effort_hours": 15
        }
        state["sprint_tasks"].append(task)
    return state

def security_engineer_node(state: ScrumState) -> ScrumState:
    """Simulates Security Engineer securing features and calculating effort."""
    total_security_tasks = 1 * len(state["user_stories"])  # 1 security task per user story
    productivity = 1  # task per week
    duration = total_security_tasks / productivity
    print(f"{GREEN}Security Engineer Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Security Tasks / Productivity = Total Duration")
    print(f"- {total_security_tasks} tasks / {productivity} task per week = {duration:.2f} weeks")
    for story in state["user_stories"]:
        task = {
            "task_id": f"SPRINT1-TASK-{len(state['sprint_tasks']) + 1:03d}",
            "story_id": story["id"],
            "description": f"Secure feature for {story['story']}",
            "role": "Security_Engineer",
            "effort_hours": 10
        }
        state["sprint_tasks"].append(task)
    return state

def ecommerce_specialist_node(state: ScrumState) -> ScrumState:
    """Simulates E-commerce Specialist optimizing features and calculating effort."""
    total_areas = 3 * len(state["user_stories"])  # 3 areas per user story
    productivity = 2  # areas per week
    duration = total_areas / productivity
    print(f"{GREEN}E-commerce Specialist Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Areas / Productivity = Total Duration")
    print(f"- {total_areas} areas / {productivity} areas per week = {duration:.2f} weeks")
    for story in state["user_stories"]:
        task = {
            "task_id": f"SPRINT1-TASK-{len(state['sprint_tasks']) + 1:03d}",
            "story_id": story["id"],
            "description": f"Optimize e-commerce for {story['story']}",
            "role": "Ecommerce_Specialist",
            "effort_hours": 10
        }
        state["sprint_tasks"].append(task)
    return state

def scrum_master_node(state: ScrumState) -> ScrumState:
    """Simulates Scrum Master facilitating sprint planning and calculating effort."""
    total_ceremonies = 4  # e.g., planning, review, retrospective
    productivity = 1  # ceremony per day
    duration_days = total_ceremonies / productivity
    print(f"{GREEN}Scrum Master Estimate:{RESET}")
    print(f"Estimated Days Required:")
    print(f"- Total Ceremonies / Productivity = Total Duration")
    print(f"- {total_ceremonies} ceremonies / {productivity} ceremony per day = {duration_days:.2f} days")
    task = {
        "task_id": f"SPRINT1-TASK-{len(state['sprint_tasks']) + 1:03d}",
        "story_id": "N/A",
        "description": "Facilitate Sprint 1 planning and ceremonies",
        "role": "Scrum_Master",
        "effort_hours": 8 * duration_days  # 8 hours per day
    }
    state["sprint_tasks"].append(task)
    return state

# Build the LangGraph workflow
workflow = StateGraph(ScrumState)
workflow.add_node("product_owner", product_owner_node)
workflow.add_node("ui_ux_designer", ui_ux_designer_node)
workflow.add_node("solution_architect", solution_architect_node)
workflow.add_node("developer", developer_node)
workflow.add_node("qa_engineer", qa_engineer_node)
workflow.add_node("technical_writer", technical_writer_node)
workflow.add_node("devops_engineer", devops_engineer_node)
workflow.add_node("security_engineer", security_engineer_node)
workflow.add_node("ecommerce_specialist", ecommerce_specialist_node)
workflow.add_node("scrum_master", scrum_master_node)

# Define the sequence of nodes, mirroring the AutoGen conversation flow
workflow.set_entry_point("product_owner")
workflow.add_edge("product_owner", "ui_ux_designer")
workflow.add_edge("ui_ux_designer", "solution_architect")
workflow.add_edge("solution_architect", "developer")
workflow.add_edge("developer", "qa_engineer")
workflow.add_edge("qa_engineer", "technical_writer")
workflow.add_edge("technical_writer", "devops_engineer")
workflow.add_edge("devops_engineer", "security_engineer")
workflow.add_edge("security_engineer", "ecommerce_specialist")
workflow.add_edge("ecommerce_specialist", "scrum_master")
workflow.add_edge("scrum_master", END)

# Compile the workflow
app = workflow.compile()

# Function to simulate the Scrum plan
def simulate_scrum_plan(customer_request: str) -> tuple[List[Dict], List[Dict]]:
    print(f"{BLUE_BOLD}=== LangGraph Scrum Plan Simulation ==={RESET}")
    initial_state = {"customer_request": customer_request, "user_stories": [], "sprint_tasks": []}
    final_state = app.invoke(initial_state)
    return final_state["user_stories"], final_state["sprint_tasks"]

# Run the simulation with the customer request
customer_request = (
    "I want to build a web-based mobile app for our bookstore where customers can browse books by genre, "
    "read previews, purchase books online, track their shipments, review books, and get personalized reading recommendations."
)
user_stories, sprint_tasks = simulate_scrum_plan(customer_request)

# Display the results
print(f"\n{BLUE_BOLD}=== Generated User Stories ==={RESET}")
for story in user_stories:
    print(f"ID: {story['id']}")
    print(f"Story: {story['story']}")
    print(f"Priority: {story['priority']}")
    print(f"Story Points: {story['story_points']}")
    print("-" * 40)

print(f"\n{BLUE_BOLD}=== Generated Sprint 1 Tasks ==={RESET}")
for task in sprint_tasks:
    print(f"Task ID: {task['task_id']}")
    print(f"Story ID: {task['story_id']}")
    print(f"Description: {task['description']}")
    print(f"Role: {task['role']}")
    print(f"Effort Hours: {task['effort_hours']}")
    print("-" * 40)
