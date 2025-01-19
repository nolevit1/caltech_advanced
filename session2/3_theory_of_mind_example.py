import os
import openai

# OPTIONAL: set your OpenAI API key if not already in environment:
# os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_KEY>"

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
# Not using MemorySaver to keep single-turn usage
from langchain_core.tools import tool

###############################################################################
# 1. Mock Supply Chain Data
###############################################################################

WAREHOUSE_STOCK = {"widget-a": 20, "widget-b": 0, "widget-c": 100}
OUTBOUND_ORDERS = []

###############################################################################
# 2. Tools (One Parameter Each, Docstrings for Each Param)
###############################################################################

@tool
def reactive_check_stock(item: str) -> str:
    """
    Checks the current stock for a single item.
    
    :param item: Name of the item to check (e.g. "widget-a").
    :return: A string indicating how many units are in stock or if out of stock.
    """
    key = item.lower()
    qty = WAREHOUSE_STOCK.get(key, 0)
    if qty > 0:
        return f"[Reactive] '{item}' in stock: {qty} units."
    else:
        return f"[Reactive] '{item}' is out of stock."

@tool
def deliberative_schedule(item_qty: str) -> str:
    """
    Attempts to schedule a delivery for 'item, quantity'.
    
    :param item_qty: A string like "widget-a, 15". 
    :return: A message about scheduling or needing restock.
    """
    try:
        item_str, qty_str = item_qty.split(",")
        item_str = item_str.strip().lower()
        qty = int(qty_str.strip())

        cur = WAREHOUSE_STOCK.get(item_str, 0)
        if cur >= qty:
            WAREHOUSE_STOCK[item_str] = cur - qty
            OUTBOUND_ORDERS.append((item_str, qty))
            return (f"[Deliberative] Scheduled {qty} of '{item_str}'. "
                    f"Remaining stock: {WAREHOUSE_STOCK[item_str]}")
        else:
            needed = qty - cur
            return (f"[Deliberative] Not enough stock for '{item_str}' "
                    f"(need {needed} more). Restock required.")
    except Exception as e:
        return f"[Deliberative] Error: {e}"

@tool
def theory_of_mind(agent_input: str) -> str:
    """
    Simulates 'theory-of-mind' logic about urgency.
    
    :param agent_input: A string like "urgent=yes, item=widget-b".
    :return: A message about whether the item is urgent or normal.
    """
    urgent = False
    item = None
    segments = [seg.strip() for seg in agent_input.split(",")]
    for seg in segments:
        if "=" in seg:
            key, val = seg.split("=")
            if key.strip().lower() == "urgent" and val.strip().lower() in ("yes", "true"):
                urgent = True
            elif key.strip().lower() == "item":
                item = val.strip().lower()

    if urgent and item:
        return (f"[Theory-of-Mind] The other agent treats '{item}' as urgent. "
                "We should expedite it.")
    else:
        return "[Theory-of-Mind] No special urgency found."

###############################################################################
# 3. Create ReAct Agent Without prefix=, Single-Step Only
###############################################################################

def main():
    llm = ChatOpenAI(
        model="gpt-4o",  # or "gpt-4", etc.
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    # Single-turn approach
    agent_app = create_react_agent(
        model=llm,
        tools=[reactive_check_stock, deliberative_schedule, theory_of_mind]
    )

    SYSTEM_MSG = """
    You are a supply chain AI with three tools, each requiring a single string argument:
      1) reactive_check_stock(item)
      2) deliberative_schedule(item_qty)
      3) theory_of_mind(agent_input)

    Each user query calls EXACTLY ONE tool. 
    - If user wants stock info: reactive_check_stock("widget-a")
    - If user wants to schedule: deliberative_schedule("widget-a, 15")
    - If user references urgent or item: theory_of_mind("urgent=yes, item=widget-b")

    Return only the tool's result. No disclaimers or extra text.
    """

    # We'll do single-step queries, each referencing one tool.
    # No memory, so each is a fresh call.
    print("\n--- Single-Step 1: Reactive Check Stock ---")
    user_query_1 = "Check if widget-a is in stock."
    result_1 = agent_app.invoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_query_1}
            ]
        },
        config={
            "configurable": {
                "recursion_limit": 80
            }
        }
    )
    print("AGENT RESPONSE:", result_1["messages"][-1].content)

    print("\n--- Single-Step 2: Deliberative Schedule ---")
    user_query_2 = "Schedule 'widget-a, 15'"
    result_2 = agent_app.invoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_query_2}
            ]
        },
        config={
            "configurable": {
                "recursion_limit": 80
            }
        }
    )
    print("AGENT RESPONSE:", result_2["messages"][-1].content)

    print("\n--- Single-Step 3: Theory of Mind ---")
    user_query_3 = "urgent=yes, item=widget-b"
    result_3 = agent_app.invoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user_query_3}
            ]
        },
        config={
            "configurable": {
                "recursion_limit": 80
            }
        }
    )
    print("AGENT RESPONSE:", result_3["messages"][-1].content)

if __name__ == "__main__":
    main()