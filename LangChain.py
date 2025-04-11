from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional
import json
import random

# Mock LLM
class MockLLM(LLM):
    def _llm_type(self) -> str:
        return "mock"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if "user stories" in prompt.lower():
            return json.dumps([
                {"id": "US-001", "story": "As a customer, I want to browse books by category so I can find books easily.", "priority": "High", "story_points": 5},
                {"id": "US-002", "story": "As a customer, I want to purchase books online so I can pay securely.", "priority": "High", "story_points": 8}
            ])
        elif "criteria" in prompt.lower():
            return json.dumps({"US-001": ["Genre filters", "Search <2s"], "US-002": ["Secure gateway", "Card support"]})
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
            return json.dumps(tasks)
        return "[]"

llm = MockLLM()

# Define chains
story_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["request"], template="Generate Scrum user stories for: {request}"),
    output_key="user_stories"
)
criteria_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["user_stories"], template="Add acceptance criteria to: {user_stories}"),
    output_key="criteria"
)
task_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["user_stories"], template="Generate Sprint 1 tasks for all roles: {user_stories}"),
    output_key="sprint_tasks"
)
overall_chain = SequentialChain(
    chains=[story_chain, criteria_chain, task_chain],
    input_variables=["request"],
    output_variables=["user_stories", "criteria", "sprint_tasks"]
)

# Simulate Scrum plan and print output directly
def simulate_scrum_plan(customer_request):
    print("=== Experiment 3: LangChain Scrum Plan ===")
    print(f"Generating Scrum Plan for: {customer_request}\n")
    
    result = overall_chain({"request": customer_request})
    user_stories = json.loads(result["user_stories"])
    criteria = json.loads(result["criteria"])
    sprint_tasks = json.loads(result["sprint_tasks"])
    
    for story in user_stories:
        story["acceptance_criteria"] = criteria.get(story["id"], [])
    
    print("User Stories:")
    for story in user_stories:
        print(f"- {story['id']}: {story['story']}")
        print(f"  Priority: {story['priority']}")
        print(f"  Story Points: {story['story_points']}")
        print("  Acceptance Criteria:")
        for crit in story["acceptance_criteria"]:
            print(f"    * {crit}")
    
    print("\nSprint 1 Tasks:")
    for task in sprint_tasks:
        print(f"- {task['task_id']}: {task['description']}")
        print(f"  Story ID: {task['story_id']}")
        print(f"  Role: {task['role']}")
        print(f"  Effort: {task['effort_hours']} hours")
    
    print(f"\nSummary: {len(user_stories)} User Stories, {len(sprint_tasks)} Tasks for Sprint 1")
    return user_stories, sprint_tasks

# Run experiment
customer_request = "Create a web-based mobile app for a bookstore where customers can browse and purchase books, pay online, track orders, leave reviews, and get recommendations."
simulate_scrum_plan(customer_request)