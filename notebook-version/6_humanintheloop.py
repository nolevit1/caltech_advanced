import os
import openai

# If needed, set your OPENAI_API_KEY here for demonstration, not recommended in production:
# os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_KEY>"

from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.types import Command

###############################################################################
# 1. Define the State
###############################################################################

class ExampleState(TypedDict):
    """State with a possible user_input key, plus an output for demonstration."""
    user_input: str
    output: str

###############################################################################
# 2. Create Node Functions
###############################################################################

def human_input_node(state: ExampleState) -> ExampleState:
    """
    In a real setting, you'd do `interrupt()` here or 
    handle user input in some special way. 
    For demonstration, we'll just read from state['user_input'] 
    as if the user typed something in.
    """
    print(f"[human_input_node] Received input: {state['user_input']}")
    return {
        "user_input": state["user_input"],
        "output": state["output"]
    }

def process_node(state: ExampleState) -> ExampleState:
    """
    A trivial node that calls GPT-4o or just manipulates the input.
    We'll just say we processed the user_input in some way.
    """
    # Example: calling GPT-4o (this is optional)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm.invoke([{"role": "user", "content": f"Received input: {state['user_input']}"}])
    new_output = response.content

    return {
        "user_input": state["user_input"],
        "output": f"[Processed] => {new_output}"
    }

###############################################################################
# 3. Build the Graph
###############################################################################

builder = StateGraph(ExampleState)
builder.add_node("human_input", human_input_node)
builder.add_node("process_node", process_node)

builder.add_edge(START, "human_input")
builder.add_edge("human_input", "process_node")

memory = MemorySaver()

# Compile the graph with a breakpoint BEFORE human_input.
# This means the graph halts execution right before we run 'human_input' 
# so that we can update the state with a "user_input" from an external source.
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["human_input"]
)

###############################################################################
# 4. Running the Graph Step-by-Step
###############################################################################

def run_example():
    # We'll define some minimal initial state with an empty user_input
    initial_state = {
        "user_input": "",
        "output": ""
    }
    config = {"configurable": {"thread_id": "example_thread"}}

    # 1) Stream the graph up to the breakpoint
    print("=== Stream until the breakpoint ===")
    for event in graph.stream(initial_state, config, stream_mode="values"):
        print(event)

    # At this point, the graph has paused BEFORE 'human_input' runs.
    # So we can 'inject' the real user input or changes:
    user_input_text = "Hello supply chain advanced team!"
    print(f"Injecting user_input: '{user_input_text}'")

    # 2) Use update_state(...) to inject the user_input 
    #    into the node 'human_input' specifically:
    graph.update_state(
        config,
        {"user_input": user_input_text},
        as_node="human_input"
    )

    # 3) Continue execution from that checkpoint 
    #    so it runs 'human_input' next
    print("=== Resuming Execution ===")
    for event in graph.stream(None, config, stream_mode="values"):
        print(event)

    # 4) Finally, let's see the final state
    final_state = graph.get_state(config)
    print("=== Final State ===")
    print(final_state.values)

if __name__ == "__main__":
    run_example()