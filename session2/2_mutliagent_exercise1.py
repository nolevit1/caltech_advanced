import os
import openai

# OPTIONAL: set your OPENAI_API_KEY here for demonstration
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
# 2. Define Tools for Each Specialized Agent
###############################################################################

# --- Tools for StockAgent ---
@tool
def check_stock(item: str) -> str:
    """
    Checks the current stock for a given item.

    :param item: e.g., "widget-a"
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
    Updates the stock by adding some quantity.

    :param item_qty: A string like "widget-b, 50"
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

# --- Tools for DeliveryAgent ---
@tool
def schedule_delivery(order_id: str) -> str:
    """
    Schedules a delivery for a given order ID.

    :param order_id: e.g., "#12345"
    :return: A message confirming the scheduled delivery or missing ID.
    """
    if not order_id:
        return "[Delivery] Missing order ID."
    OUTBOUND_ORDERS.append(order_id)
    return f"[Delivery] Scheduled shipment for order '{order_id}'."

@tool
def estimate_shipping_cost(args: str) -> str:
    """
    Mock cost estimation. Expects "widget-a, distance=100"

    :param args: e.g. "widget-a, distance=100"
    :return: A cost estimate message.
    """
    try:
        parts = args.split(",")
        item_str = parts[0].strip().lower()
        distance_info = parts[1].strip().lower()  # e.g. distance=100
        dist_val = int(distance_info.split("=")[1])
        cost = dist_val * 0.5  # Some silly formula
        return f"[Delivery] Estimated cost for shipping '{item_str}' over {dist_val} km: ${cost:.2f}"
    except Exception as e:
        return f"[Delivery] Cost estimation error: {e}"

# --- Tools for PriorityAgent (theory-of-mind) ---
@tool
def check_urgency(args: str) -> str:
    """
    Basic 'theory of mind' approach. Expects 'urgent=yes, item=widget-b'
    and returns how to handle if urgent.

    :param args: e.g., "urgent=yes, item=widget-b"
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
        return (f"[Priority] The other agent treats '{item}' as urgent. "
                "We should expedite handling or coordinate carefully.")
    else:
        return "[Priority] No special urgency found."

###############################################################################
# 3. Create Three Separate Agents on GPT-4o
###############################################################################

def create_agents():
    # Each agent uses model="gpt-4o"
    llm_stock = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_delivery = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_priority = ChatOpenAI(model="gpt-4o", temperature=0)

    # StockAgent -> Tools for stock checks and updates
    stock_agent = create_react_agent(
        model=llm_stock,
        tools=[check_stock, update_stock]
    )

    # DeliveryAgent -> Tools for scheduling and cost estimates
    delivery_agent = create_react_agent(
        model=llm_delivery,
        tools=[schedule_delivery, estimate_shipping_cost]
    )

    # PriorityAgent -> Tools for urgent constraints (theory-of-mind)
    priority_agent = create_react_agent(
        model=llm_priority,
        tools=[check_urgency]
    )

    return stock_agent, delivery_agent, priority_agent

###############################################################################
# 4. Coordinator to Route User Queries to the Right Agent
###############################################################################

def coordinator(user_input: str) -> str:
    """
    A naive function that decides which agent to call
    based on keywords in user_input.
    """
    txt_lower = user_input.lower()

    if "stock" in txt_lower or "update" in txt_lower:
        return "stock"
    elif "delivery" in txt_lower or "schedule" in txt_lower or "cost" in txt_lower:
        return "delivery"
    elif "urgent" in txt_lower or "priority" in txt_lower:
        return "priority"
    else:
        return "none"

###############################################################################
# 5. Single-Step Usage for Each Agent
###############################################################################

def main():
    stock_agent, delivery_agent, priority_agent = create_agents()

    # System messages to each agent
    SYSTEM_MSG_STOCK = """
    You are the 'StockAgent' in a supply chain system.
    Call either check_stock(item) or update_stock(item_qty)
    exactly once based on the user's request. Return tool output only.
    """

    SYSTEM_MSG_DELIVERY = """
    You are the 'DeliveryAgent' in a supply chain system.
    Call either schedule_delivery(order_id) or estimate_shipping_cost(args)
    exactly once based on the user's request. Return tool output only.
    """

    SYSTEM_MSG_PRIORITY = """
    You are the 'PriorityAgent' in a supply chain system.
    Call check_urgency(args) exactly once based on the user's request.
    Return the tool's output only.
    """

    # Example user queries
    user_queries = [
        "Check if widget-a is in stock.",
        "Update stock: 'widget-b, 50'",
        "Schedule a delivery for order #12345",
        "Estimate shipping cost for 'widget-a, distance=250'",
        "We have urgent=yes, item=widget-b. Please advise."
    ]

    for idx, query in enumerate(user_queries, start=1):
        agent_type = coordinator(query)
        if agent_type == "stock":
            result = stock_agent.invoke(
                {"messages": [
                    {"role": "system", "content": SYSTEM_MSG_STOCK},
                    {"role": "user",   "content": query}
                ]},
                config={"configurable": {"recursion_limit": 50}}
            )
            print(f"\n--- Query #{idx} (StockAgent) ---")
            print("User:", query)
            print("Agent Response:", result["messages"][-1].content)

        elif agent_type == "delivery":
            result = delivery_agent.invoke(
                {"messages": [
                    {"role": "system", "content": SYSTEM_MSG_DELIVERY},
                    {"role": "user",   "content": query}
                ]},
                config={"configurable": {"recursion_limit": 50}}
            )
            print(f"\n--- Query #{idx} (DeliveryAgent) ---")
            print("User:", query)
            print("Agent Response:", result["messages"][-1].content)

        elif agent_type == "priority":
            result = priority_agent.invoke(
                {"messages": [
                    {"role": "system", "content": SYSTEM_MSG_PRIORITY},
                    {"role": "user",   "content": query}
                ]},
                config={"configurable": {"recursion_limit": 50}}
            )
            print(f"\n--- Query #{idx} (PriorityAgent) ---")
            print("User:", query)
            print("Agent Response:", result["messages"][-1].content)

        else:
            print(f"\n--- Query #{idx} (No Agent) ---")
            print("User:", query)
            print("Agent Response: No suitable agent found.")

if __name__ == "__main__":
    main()