import os
import openai

# Optionally set your OpenAI API key in code (not recommended for production):
# os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_KEY>"

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool


###############################################################################
# 1. Supply Chain Tools: check_inventory, update_inventory, schedule_delivery
###############################################################################

@tool
def check_inventory(item: str) -> str:
    """
    Mock function to query item stock levels.
    In production, you'd query a real database or ERP system.
    """
    mock_db = {"widget-a": 20, "widget-b": 0, "widget-c": 100}
    key = item.lower()
    if key in mock_db:
        qty = mock_db[key]
        if qty > 0:
            return f"[Inventory] '{item}' in stock: {qty} units."
        else:
            return f"[Inventory] '{item}' is out of stock."
    return f"[Inventory] No data for '{item}'."

@tool
def update_inventory(args: str) -> str:
    """
    Mock function to 'add' a certain quantity to a given item.
    Usage example: "widget-a, 50"
    In a real system, you'd parse carefully or use a Pydantic model + DB updates.
    """
    try:
        item_str, qty_str = args.split(",")
        item_str = item_str.strip().lower()
        qty = int(qty_str.strip())
        return f"[Update] Successfully added {qty} units of '{item_str}'."
    except Exception as e:
        return f"[Update] Error parsing input: {args}. ({e})"

@tool
def schedule_delivery(order_id: str) -> str:
    """
    Mock function to schedule a delivery for a given order ID.
    In practice, you'd call a shipping or 3PL API.
    """
    if not order_id:
        return "[Delivery] Missing order ID."
    return f"[Delivery] Scheduled shipment for order '{order_id}' on next truck run."


###############################################################################
# 2. Create a ReAct agent with ChatOpenAI from langchain_openai
###############################################################################

def main():
    # You can adjust the model name, or pass additional parameters if needed
    llm = ChatOpenAI(
        model="gpt-4o",    # or "gpt-4" / "gpt-3.5-turbo" / custom endpoint
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # api_key="...",     # Only if you prefer passing the key directly
        # base_url="...",    # e.g. self-hosted instance or custom proxy
        # organization="...",# or other config parameters
    )

    # Optional memory for multi-turn conversations
    memory = MemorySaver()

    # Create a ReAct agent with the defined supply chain tools
    agent_app = create_react_agent(
        model=llm,
        tools=[check_inventory, update_inventory, schedule_delivery],
        checkpointer=memory
    )

    # We'll reuse a single thread_id to simulate a multi-turn conversation
    THREAD_ID = 42

    print("\n--- EXAMPLE 1: Check Inventory ---")
    user_input_1 = "Check inventory for widget-a please."
    result_1 = agent_app.invoke(
        {"messages": [{"role": "user", "content": user_input_1}]},
        config={"configurable": {"thread_id": THREAD_ID}}
    )
    print("AGENT RESPONSE:", result_1["messages"][-1].content)

    print("\n--- EXAMPLE 2: Update Inventory ---")
    user_input_2 = "Add 50 units of widget-b to inventory."
    result_2 = agent_app.invoke(
        {"messages": [{"role": "user", "content": user_input_2}]},
        config={"configurable": {"thread_id": THREAD_ID}}
    )
    print("AGENT RESPONSE:", result_2["messages"][-1].content)

    print("\n--- EXAMPLE 3: Schedule a Delivery ---")
    user_input_3 = "Now schedule a delivery for order #12345."
    result_3 = agent_app.invoke(
        {"messages": [{"role": "user", "content": user_input_3}]},
        config={"configurable": {"thread_id": THREAD_ID}}
    )
    print("AGENT RESPONSE:", result_3["messages"][-1].content)

    print("\n--- EXAMPLE 4: Combined Query ---")
    user_input_4 = "Check if widget-b is still out of stock, and if not schedule a delivery for order #98765."
    result_4 = agent_app.invoke(
        {"messages": [{"role": "user", "content": user_input_4}]},
        config={"configurable": {"thread_id": THREAD_ID}}
    )
    print("AGENT RESPONSE:", result_4["messages"][-1].content)


if __name__ == "__main__":
    main()