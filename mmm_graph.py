from typing import TypedDict, Optional
from pipeline import TaskPlan,run_planner,build_tool_registry ,TaskPlan ,run_executor,run_synthesizer  # reuse your existing Pydantic model
from langgraph.types import interrupt
from src.db import get_collection
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import uuid
from langgraph.types import Command


class MMMState(TypedDict):
    question : str
    plan    : Optional[dict]  # planner writes this
    execution_log : Optional[list]     # list of tools called
    answer : Optional[dict]       # synthesizer writes this
    human_feedback : Optional[str]  # human can write this to trigger replan
    error: Optional[str]  # any node can write this
    tools_used: Optional[list]  # track which tools were used


def planner_node(state: MMMState) -> dict:  # MMMstate works like a slate to be written on by various nodes
    question = state["question"]
    plan_object = run_planner(question)  # planner logic returns a Pydantic model (TaskPlan)
    plan_dict = plan_object.model_dump() # convert Pydantic model to dict
    return {'plan': plan_dict} #only updates the 'plan' field in state, other fields remain unchanged


def human_approval_node(state: MMMState) -> dict:
    # in real life, this would be a UI where a human reviews the plan and either approves or provides feedback
    proposed_plan = state["plan"]    
    feedback = interrupt({
    "question":  state["question"],
    "objective": proposed_plan["objective"],
    "steps":     [f"Step {i+1}: {s['task']}" for i, s in enumerate(proposed_plan["subtasks"])],
    "message":   "Review the plan above. Type 'approve' to proceed or 'reject' to cancel."
                        })
    print(f"> Resumed with input: {feedback}")
    return {"human_feedback": feedback}  # write human feedback back to state  

def route_after_approval(state: MMMState) -> str:
    feedback = state.get("human_feedback", "").lower()
    if feedback == "approve":
        return "executor"  # proceed to execution
    else:
        return "rejected"  
def rejected_node(state: MMMState) -> dict:
    return {"error": f"Plan rejected by human. Feedback: {state.get('human_feedback', 'no feedback')}"}

def executor_node(state: MMMState) -> dict:
    collection = get_collection() 
    tool_registry = build_tool_registry(collection)
    plan = TaskPlan(**state["plan"])
    execution_log = run_executor(plan, tool_registry)
    return {"execution_log": execution_log, "tools_used": [entry["tool_name"] for entry in execution_log]}



def synthesizer_node(state: MMMState) -> dict:
    question = state["question"]
    execution_log = state["execution_log"]
    answer = run_synthesizer(question, execution_log)
    return {"answer": answer}
    

builder = StateGraph(MMMState)
builder.add_node("planner", planner_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("rejected", rejected_node)
builder.add_node("executor", executor_node)
builder.add_node("synthesizer", synthesizer_node)
builder.set_entry_point("planner")
builder.add_edge("planner", "human_approval")
builder.add_conditional_edges("human_approval", route_after_approval, {"executor": "executor", "rejected": "rejected"})
builder.add_edge("executor", "synthesizer")
builder.add_edge("synthesizer", END) 
builder.add_edge("rejected", END)



def run_with_approval(question: str):
    with SqliteSaver.from_conn_string("mmm_checkpoints.db") as checkpointer:
        app = builder.compile(checkpointer=checkpointer)
        thread_id = str(uuid.uuid4())
        config    = {"configurable": {"thread_id": thread_id}}

        result = app.invoke({"question": question}, config=config)
    # check if graph paused at interrupt
        if "__interrupt__" in result:
            payload = result["__interrupt__"][0].value

            # print the plan for human review
            print("\n" + "=" * 50)
            print("MMM COPILOT — PLAN APPROVAL")
            print("=" * 50)
            print(f"Question : {payload['question']}")
            print(f"Objective: {payload['objective']}\n")
            print("Proposed steps:")
            for step in payload["steps"]:
                print(f"  {step}")
            print(f"\n{payload['message']}")

            # get human input
            feedback = input("\nYour decision: ").strip().lower()

            # resume the graph with feedback
            
            final_result = app.invoke(
                Command(resume=feedback),
                config=config    # same thread_id — critical
            )

            # handle outcome
            if final_result.get("answer"):
                from pipeline import print_answer
                print_answer(final_result["answer"])
            elif final_result.get("error"):
                print(f"\nPipeline ended: {final_result['error']}")

if __name__ == "__main__":
    run_with_approval("What is the ROI of HCP channels for oncology brands?")