import os
import openai

# If needed, set your OpenAI API key:
# os.environ["OPENAI_API_KEY"] = "<YOUR_KEY>"

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from typing import TypedDict

###############################################################################
# 1. Define Our Shared State (with memory).
###############################################################################

class SupplyChainState(TypedDict):
    # We keep a conversation log plus a "stock_level" key for demonstration
    messages: list
    stock_level: int

###############################################################################
# 2. Create Two Specialized Agents (stock and delivery).
###############################################################################

def check_stock_tool(stock_level: int) -> str:
    """
    Simple 'tool' that just returns a string about our stock level.
    This might be part of a real 'StockAgent' in practice.
    """
    if stock_level > 0:
        return f"[StockAgent] We have {stock_level} items in stock."
    else:
        return "[StockAgent] Out of stock!"

def schedule_delivery_tool(order_id: str) -> str:
    """
    A 'DeliveryAgent' tool that confirms scheduling if we have items.
    """
    return f"[DeliveryAgent] Delivery scheduled for order {order_id}"

###############################################################################
# 3. Build Each Sub-Agent Using ReAct for Single-Step
#    (Though each is just a single tool for demonstration).
###############################################################################

llm_stock = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
stock_agent = create_react_agent(model=llm_stock, tools=[check_stock_tool])

llm_delivery = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
delivery_agent = create_react_agent(model=llm_delivery, tools=[schedule_delivery_tool])

###############################################################################
# 4. The CoordinatingAgent: A Simple Node that Decides Which Sub-Agent to Call
###############################################################################

def coordinator_node(state: SupplyChainState):
    """
    The coordinator node checks if we have stock. If no stock, it calls the stock agent
    to confirm 'Out of stock!' Then updates the stock. Next, it calls the delivery agent 
    to schedule an order.
    """
    # 1) First call the stock sub-agent:
    # We imagine user wants to see how many are in stock
    result_1 = stock_agent.invoke(
        {"messages": state["messages"], "stock_level": state["stock_level"]},
        config={"configurable": {"thread_id": "example"}}
    )
    # Merge the output back into our state's messages.
    # The sub-agent typically returns a final text or tool output as a string
    # for demonstration, let's store it.
    msg_1 = result_1["messages"][-1].content  # The last message from the sub-agent
    # For demonstration, if out of stock, we "restock" it:
    if "Out of stock" in msg_1:
        state["stock_level"] = 10  # we restock

    # 2) Then call the delivery sub-agent:
    result_2 = delivery_agent.invoke(
        {"messages": state["messages"], "order_id": "#1234"},
        config={"configurable": {"thread_id": "example"}}
    )
    msg_2 = result_2["messages"][-1].content

    # Combine results into a single final message
    final_message = f"{msg_1}\n{msg_2}"
    # We'll update the 'messages' in state with a new AI message:
    new_state = {
        "messages": state["messages"] + [final_message],
        "stock_level": state["stock_level"]
    }
    return new_state

###############################################################################
# 5. Build a Simple Graph with a Single Node for Our Coordinator.
###############################################################################

builder = StateGraph(SupplyChainState)
builder.add_node("coordinator", coordinator_node)
builder.add_edge(START, "coordinator")

memory = MemorySaver()  # We'll keep state across multiple runs
graph = builder.compile(checkpointer=memory)

###############################################################################
# 6. Run the Graph!
###############################################################################

def run_example():
    initial_state = {
        "messages": ["User wants to check stock then schedule a delivery."],
        "stock_level": 0
    }
    config = {"configurable": {"thread_id": "supply_thread"}}

    output = graph.invoke(initial_state, config=config)
    print("=== Final Output ===")
    print("Messages:", output["messages"])
    print("Stock level:", output["stock_level"])

if __name__ == "__main__":
    run_example()