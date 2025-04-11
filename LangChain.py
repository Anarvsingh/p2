from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional
import json
import random

# Color codes for console output
GREEN = "\033[92m"
RESET = "\033[0m"

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
            features = [
                "browse books by genre", "read book previews", "purchase books online",
                "track shipments", "review books", "get personalized recommendations",
                "manage my profile", "view order history", "receive promotional offers",
                "contact customer support"
            ]
            return json.dumps([
                {"id": f"US-{i:03d}", "story": f"As a customer, I want to {feature} so I can manage my bookstore experience effectively.", "priority": "High", "story_points": random.randint(1, 8)}
                for i, feature in enumerate(features, start=1)
            ])
        elif "criteria" in prompt.lower():
            return json.dumps({
                f"US-{i:03d}": [f"Criteria 1 for story {i}", f"Criteria 2 for story {i}"]
                for i in range(1, 11)
            })
        elif "tasks" in prompt.lower():
            tasks = []
            for story_id in [f"US-{i:03d}" for i in range(1, 11)]:
                role_tasks = {
                    "UIUXDesigner": f"Design UI for story {story_id}",
                    "SolutionArchitect": f"Define architecture for story {story_id}",
                    "Developer": f"Code feature for story {story_id}",
                    "QAEngineer": f"Test feature for story {story_id}",
                    "TechnicalWriter": f"Write guide for story {story_id}",
                    "DevOpsEngineer": f"Set up infrastructure for story {story_id}",
                    "SecurityEngineer": f"Secure feature for story {story_id}",
                    "EcommerceSpecialist": f"Optimize e-commerce for story {story_id}"
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

# Function to calculate and print role efforts
def calculate_role_efforts(user_stories):
    num_stories = len(user_stories)
    total_points = sum(story["story_points"] for story in user_stories)
    
    # Product Owner
    total_features = num_stories
    productivity_po = 3  # features per week
    duration_po = total_features / productivity_po
    print(f"{GREEN}Product Owner Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Features / Productivity = Total Duration")
    print(f"- {total_features} features / {productivity_po} features per week = {duration_po:.2f} weeks")
    
    # UI/UX Designer
    total_screens = 3 * num_stories
    productivity_ui = 3  # screens per week
    duration_ui = total_screens / productivity_ui
    print(f"{GREEN}UI/UX Designer Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Screens / Productivity = Total Duration")
    print(f"- {total_screens} screens / {productivity_ui} screens per week = {duration_ui:.2f} weeks")
    
    # Solution Architect
    total_components = 2 * num_stories
    productivity_arch = 1  # components per week
    duration_arch = total_components / productivity_arch
    print(f"{GREEN}Solution Architect Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Components / Productivity = Total Duration")
    print(f"- {total_components} components / {productivity_arch} component per week = {duration_arch:.2f} weeks")
    
    # Developer
    total_sloc = total_points * 100
    productivity_dev = 500  # SLOC per week
    duration_dev = total_sloc / productivity_dev
    print(f"{GREEN}Developer Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total SLOC / Productivity = Total Duration")
    print(f"- {total_sloc} SLOC / {productivity_dev} SLOC per week = {duration_dev:.2f} weeks")
    
    # QA Engineer
    total_test_cases = 5 * num_stories
    productivity_qa = 5  # test cases per day
    duration_qa = total_test_cases / productivity_qa
    print(f"{GREEN}QA Engineer Estimate:{RESET}")
    print(f"Estimated Days Required:")
    print(f"- Total Test Cases / Productivity = Total Duration")
    print(f"- {total_test_cases} test cases / {productivity_qa} test cases per day = {duration_qa:.2f} days")
    
    # Technical Writer
    total_pages = 4 * num_stories
    productivity_tw = 4  # pages per week
    duration_tw = total_pages / productivity_tw
    print(f"{GREEN}Technical Writer Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Pages / Productivity = Total Duration")
    print(f"- {total_pages} pages / {productivity_tw} pages per week = {duration_tw:.2f} weeks")
    
    # DevOps Engineer
    total_tasks_devops = 2 * num_stories
    productivity_devops = 2  # tasks per week
    duration_devops = total_tasks_devops / productivity_devops
    print(f"{GREEN}DevOps Engineer Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Tasks / Productivity = Total Duration")
    print(f"- {total_tasks_devops} tasks / {productivity_devops} tasks per week = {duration_devops:.2f} weeks")
    
    # Security Engineer
    total_security_tasks = 1 * num_stories
    productivity_sec = 1  # tasks per week
    duration_sec = total_security_tasks / productivity_sec
    print(f"{GREEN}Security Engineer Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Security Tasks / Productivity = Total Duration")
    print(f"- {total_security_tasks} tasks / {productivity_sec} task per week = {duration_sec:.2f} weeks")
    
    # E-commerce Specialist
    total_areas = 3 * num_stories
    productivity_ecom = 2  # areas per week
    duration_ecom = total_areas / productivity_ecom
    print(f"{GREEN}E-commerce Specialist Estimate:{RESET}")
    print(f"Estimated Weeks Required:")
    print(f"- Total Areas / Productivity = Total Duration")
    print(f"- {total_areas} areas / {productivity_ecom} areas per week = {duration_ecom:.2f} weeks")
    
    # Scrum Master
    total_ceremonies = 4
    productivity_sm = 1  # ceremony per day
    duration_sm = total_ceremonies / productivity_sm
    print(f"{GREEN}Scrum Master Estimate:{RESET}")
    print(f"Estimated Days Required:")
    print(f"- Total Ceremonies / Productivity = Total Duration")
    print(f"- {total_ceremonies} ceremonies / {productivity_sm} ceremony per day = {duration_sm:.2f} days")

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
    
    # Calculate and print role efforts
    calculate_role_efforts(user_stories)
    
    print("\nUser Stories:")
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
