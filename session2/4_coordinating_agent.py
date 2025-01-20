import os
import openai

# If needed, set your OPENAI_API_KEY in environment for GPT-4o usage:
# os.environ["OPENAI_API_KEY"] = "<YOUR_OPENAI_KEY>"

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

###############################################################################
# 1. Mock Supply Chain Data
###############################################################################

WAREHOUSE_STOCK = {"widget-a": 20, "widget-b": 0, "widget-c": 100}
OUTBOUND_ORDERS = []

###############################################################################
# 2. Specialized Tools for Each Agent
###############################################################################
#
#  StockAgent Tools
#

@tool
def check_stock(item: str) -> str:
    """
    Checks the current stock for a single item.

    :param item: e.g. "widget-a"
    :return: A message indicating in-stock or out-of-stock.
    """
    key = item.lower()
    qty = WAREHOUSE_STOCK.get(key, 0)
    if qty > 0:
        return f"[Stock] '{item}' in stock: {qty} units."
    else:
        return f"[Stock] '{item}' is out of stock."

@tool
def update_stock(item_qty: str) -> str:
    """
    Updates warehouse stock by adding quantity.

    :param item_qty: e.g. "widget-b, 50"
    :return: Confirmation or error message.
    """
    try:
        it, qty_str = item_qty.split(",")
        it = it.strip().lower()
        qty = int(qty_str.strip())
        WAREHOUSE_STOCK[it] = WAREHOUSE_STOCK.get(it, 0) + qty
        return f"[Stock] Successfully added {qty} units of '{it}'."
    except Exception as e:
        return f"[Stock] Update error: {e}"

#
#  DeliveryAgent Tools
#

@tool
def schedule_delivery(order_id: str) -> str:
    """
    Schedules a delivery for a given order ID.

    :param order_id: e.g. "#12345"
    :return: A message confirming the scheduled delivery or missing ID.
    """
    if not order_id:
        return "[Delivery] Missing order ID."
    OUTBOUND_ORDERS.append(order_id)
    return f"[Delivery] Scheduled shipment for order '{order_id}'."

@tool
def estimate_shipping_cost(args: str) -> str:
    """
    Mock shipping cost estimation. Expects "widget-a, distance=100"

    :param args: e.g. "widget-a, distance=100"
    :return: A cost estimate message.
    """
    try:
        parts = args.split(",")
        item_str = parts[0].strip().lower()
        dist_info = parts[1].strip().lower()  # e.g. "distance=100"
        dist_val = int(dist_info.split("=")[1])
        cost = dist_val * 0.5
        return f"[Delivery] Cost to ship '{item_str}' {dist_val} km: ${cost:.2f}"
    except Exception as e:
        return f"[Delivery] Cost estimation error: {e}"

#
#  PriorityAgent (Theory-of-Mind) Tools
#

@tool
def check_urgency(args: str) -> str:
    """
    Basic 'theory of mind' for urgent requests.

    :param args: e.g. "urgent=yes, item=widget-b"
    :return: A message describing urgency handling.
    """
    urgent = False
    item = None
    segments = [seg.strip() for seg in args.split(",")]
    for seg in segments:
        if "=" in seg:
            k, v = seg.split("=")
            if k.strip().lower() == "urgent" and v.strip().lower() in ("yes", "true"):
                urgent = True
            elif k.strip().lower() == "item":
                item = v.strip().lower()

    if urgent and item:
        return (f"[Priority] The agent with item '{item}' is urgent. "
                "We should expedite handling or coordinate carefully.")
    else:
        return "[Priority] No special urgency found."

###############################################################################
# 3. Create the Three Specialized Agents (GPT-4o)
###############################################################################

def create_specialized_agents():
    llm_stock = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_delivery = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_priority = ChatOpenAI(model="gpt-4o", temperature=0)

    # StockAgent
    stock_agent = create_react_agent(
        model=llm_stock,
        tools=[check_stock, update_stock]
    )

    # DeliveryAgent
    delivery_agent = create_react_agent(
        model=llm_delivery,
        tools=[schedule_delivery, estimate_shipping_cost]
    )

    # PriorityAgent
    priority_agent = create_react_agent(
        model=llm_priority,
        tools=[check_urgency]
    )

    return stock_agent, delivery_agent, priority_agent

###############################################################################
# 4. The CoordinatingAgent Tools
###############################################################################

@tool
def call_stock_agent(query: str) -> str:
    """
    The coordinator calls StockAgent in single-step usage for inventory queries.
    """
    return "[Coordinator] (No real implementation yet)"

@tool
def call_delivery_agent(query: str) -> str:
    """
    The coordinator calls DeliveryAgent in single-step usage for shipping tasks.
    """
    return "[Coordinator] (No real implementation yet)"

@tool
def call_priority_agent(query: str) -> str:
    """
    The coordinator calls PriorityAgent in single-step usage for urgent checks.
    """
    return "[Coordinator] (No real implementation yet)"

###############################################################################
# 5. Create the CoordinatingAgent (GPT-4o) that orchestrates sub-calls
###############################################################################

def create_coordinating_agent():
    # We'll do a ReAct agent that can call call_stock_agent, etc.
    llm_coord = ChatOpenAI(model="gpt-4o", temperature=0)
    coordinating_agent = create_react_agent(
        model=llm_coord,
        tools=[call_stock_agent, call_delivery_agent, call_priority_agent]
    )
    return coordinating_agent

###############################################################################
# 6. Wire Up the Actual Calls
###############################################################################

def main():
    # Create specialized agents (Stock, Delivery, Priority)
    stock_agent, delivery_agent, priority_agent = create_specialized_agents()

    # Create the coordinating agent
    coordinating_agent = create_coordinating_agent()

    # Now we override the call_* agent tools so they actually call the specialized agent

    def call_stock_agent_impl(query: str) -> str:
        """
        Actually calls StockAgent in single-step usage.
        """
        SYSTEM_MSG_STOCK = """
        You are the StockAgent. 
        You have 'check_stock(item)' and 'update_stock(item_qty)'.
        Call exactly one of them based on the user's query. Return the tool's output.
        """
        result = stock_agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG_STOCK},
                    {"role": "user",   "content": query}
                ]
            },
            config={"configurable": {"recursion_limit": 50}}
        )
        return result["messages"][-1].content

    def call_delivery_agent_impl(query: str) -> str:
        """
        Actually calls DeliveryAgent in single-step usage.
        """
        SYSTEM_MSG_DELIVERY = """
        You are the DeliveryAgent.
        You have 'schedule_delivery(order_id)' and 'estimate_shipping_cost(args)'.
        Call one based on user's request. Return the tool's output.
        """
        result = delivery_agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG_DELIVERY},
                    {"role": "user", "content": query}
                ]
            },
            config={"configurable": {"recursion_limit": 50}}
        )
        return result["messages"][-1].content

    def call_priority_agent_impl(query: str) -> str:
        """
        Actually calls PriorityAgent in single-step usage.
        """
        SYSTEM_MSG_PRIORITY = """
        You are the PriorityAgent.
        You have 'check_urgency(args)' to handle urgent or priority tasks.
        Call it exactly once. Return the tool's output.
        """
        result = priority_agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG_PRIORITY},
                    {"role": "user", "content": query}
                ]
            },
            config={"configurable": {"recursion_limit": 50}}
        )
        return result["messages"][-1].content

    # Patch docstrings & function references onto the existing 'call_*' tools
    call_stock_agent.__doc__ = call_stock_agent_impl.__doc__
    call_stock_agent.func = call_stock_agent_impl

    call_delivery_agent.__doc__ = call_delivery_agent_impl.__doc__
    call_delivery_agent.func = call_delivery_agent_impl

    call_priority_agent.__doc__ = call_priority_agent_impl.__doc__
    call_priority_agent.func = call_priority_agent_impl

    # Now the CoordinatingAgent can call these sub-agent tools in sequence if it wants:
    COORDINATOR_SYSTEM_MSG = """
    You are the CoordinatingAgent for a supply chain. 
    You have:
      1) call_stock_agent(query)
      2) call_delivery_agent(query)
      3) call_priority_agent(query)

    The user might request multiple steps: 
      - Checking or updating stock
      - Scheduling deliveries
      - Checking if something is urgent
    You can call them in sequence to solve the user's request. 
    Summarize final results for the user. 
    No disclaimers. 
    """

    # Example user queries that might need multiple sub-agent calls
    user_queries = [
        "Check if widget-b is in stock, and if out of stock, add 40 units. Then schedule a delivery for order #111.",
        "We have urgent=yes, item=widget-a. But first see if widget-a is in stock, then estimate shipping cost if it's in stock."
    ]

    for idx, query in enumerate(user_queries, start=1):
        result = coordinating_agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": COORDINATOR_SYSTEM_MSG},
                    {"role": "user",   "content": query}
                ]
            },
            config={"configurable": {"recursion_limit": 80}}
        )
        print(f"\n--- CoordinatingAgent Query #{idx} ---")
        print("User:", query)
        print("Agent Response:", result["messages"][-1].content)


if __name__ == "__main__":
    main()