from langgraph.graph import StateGraph, END
from typing import Dict, List, TypedDict
import json
import random

# Define state
class ScrumState(TypedDict):
    customer_request: str
    user_stories: List[Dict]
    sprint_tasks: List[Dict]

# Mock LLM
def mock_llm_node(prompt):
    if "user stories" in prompt.lower():
        return [
            {"id": "US-001", "story": "As a customer, I want to browse books by category so I can find books easily.", "priority": "High", "story_points": 5, "acceptance_criteria": ["Genre filters", "Search <2s"]},
            {"id": "US-002", "story": "As a customer, I want to purchase books online so I can pay securely.", "priority": "High", "story_points": 8, "acceptance_criteria": ["Secure gateway", "Card support"]}
        ]
    elif "tasks" in prompt.lower():
        tasks = []
        for story_id in ["US-001", "US-002"]:
            role_tasks = {
                "UIUXDesigner": f"Design {'browsing UI' if story_id == 'US-001' else 'checkout UI'}",
                "SolutionArchitect": f"Define {'catalog API' if story_id == 'US-001' else 'payment API'}",
                "Developer": f"Code {'browsing feature' if story_id == 'US-001' else 'payment integration'}",
                "QAEngineer": f"Test {'browsing functionality' if story_id == 'US-001' else 'payment security'}",
                "TechnicalWriter": f"Write {'browsing guide' if story_id == 'US-001' else 'payment guide'}",
                "DevOpsEngineer": f"Set up {'cloud hosting' if story_id == 'US-001' else 'payment gateway'}",
                "SecurityEngineer": f"Secure {'catalog API' if story_id == 'US-001' else 'payment API'}",
                "EcommerceSpecialist": f"Optimize {'search flow' if story_id == 'US-001' else 'checkout flow'}"
            }
            task_id = len(tasks) + 1
            for role, desc in role_tasks.items():
                effort = random.randint(1, 4) if role == "TechnicalWriter" else random.randint(3, 8) if role in ["UIUXDesigner", "DevOpsEngineer"] else random.randint(4, 10)
                tasks.append({
                    "task_id": f"SPRINT1-TASK-{task_id:03d}",
                    "story_id": story_id,
                    "description": desc,
                    "role": role,
                    "effort_hours": effort
                })
                task_id += 1
        tasks.append({
            "task_id": f"SPRINT1-TASK-{task_id:03d}",
            "story_id": "N/A",
            "description": "Plan Sprint 1 tasks",
            "role": "ScrumMaster",
            "effort_hours": random.randint(2, 4)
        })
        return tasks
    return []

# Define nodes
def generate_stories(state: ScrumState) -> ScrumState:
    state["user_stories"] = mock_llm_node(f"Generate user stories for: {state['customer_request']}")
    return state

def generate_tasks(state: ScrumState) -> ScrumState:
    state["sprint_tasks"] = mock_llm_node(f"Generate Sprint 1 tasks for: {json.dumps(state['user_stories'])}")
    return state

# Build graph
workflow = StateGraph(ScrumState)
workflow.add_node("generate_stories", generate_stories)
workflow.add_node("generate_tasks", generate_tasks)
workflow.add_edge("generate_stories", "generate_tasks")
workflow.add_edge("generate_tasks", END)
workflow.set_entry_point("generate_stories")
app = workflow.compile()

# Simulate Scrum plan and print output directly
def simulate_scrum_plan(customer_request):
    print("=== Experiment 4: LangGraph Scrum Plan ===")
    print(f"Generating Scrum Plan for: {customer_request}\n")
    
    state = app.invoke({"customer_request": customer_request, "user_stories": [], "sprint_tasks": []})
    
    print("User Stories:")
    for story in state["user_stories"]:
        print(f"- {story['id']}: {story['story']}")
        print(f"  Priority: {story['priority']}, Story Points: {story['story_points']}")
        print("  Acceptance Criteria:")
        for crit in story["acceptance_criteria"]:
            print(f"    * {crit}")
    
    print("\nSprint 1 Tasks:")
    for task in state["sprint_tasks"]:
        print(f"- {task['task_id']}: {task['description']}")
        print(f"  Role: {task['role']}, Effort: {task['effort_hours']} hours")
    
    print(f"\nSummary: {len(state['user_stories'])} User Stories, {len(state['sprint_tasks'])} Tasks for Sprint 1")
    return state["user_stories"], state["sprint_tasks"]

# Run experiment
customer_request = "Create a web-based mobile app for a bookstore where customers can browse and purchase books, pay online, track orders, leave reviews, and get recommendations."
simulate_scrum_plan(customer_request)