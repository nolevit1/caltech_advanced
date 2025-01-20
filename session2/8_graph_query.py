import os
import openai
import requests

# this is an example but far from production ready

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, initialize_agent, AgentType

###############################################################################
# 1) THE WIKIDATA QUERY TOOL
###############################################################################

def wikidata_query_tool(sparql_query: str) -> str:
    '''
    Tool to run SPARQL queries on Wikidata. Input must be a SPARQL query string.
    dont use backticks or triple quotes.
    '''
    url = "https://query.wikidata.org/sparql"
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(url, params={"query": sparql_query}, headers=headers)

    if not response.ok:
        return f"[Error] SPARQL query failed: {response.text}"

    data = response.json()
    print(data)
    return data


wikidata_tool = Tool(
    name="WikidataQuery",
    func=wikidata_query_tool,
    description=(
        "Use this tool to run SPARQL queries on Wikidata. Input must be a SPARQL query string. avoid bcakticks or triple quotes. Look up IDS when needed"
        "It returns a textual summary of the results."
        "ensure valid SPARQL syntax and avoid long-running queries."
        # "enclose the query in triple quotes for multi-line queries, but dont use ``` or similar."
        "dont write sparql: before the query. Directly write the query."
        "dont use backticks or triple quotes."
        ),
)

###############################################################################
# 2) BUILD THE AGENT
###############################################################################

def create_wikidata_agent():
    """
    Creates a Zero-Shot ReAct style agent, with handle_parsing_errors=True 
    so it attempts to fix or retry if the LLM returns invalid ReAct text.
    """
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    tools = [wikidata_tool]

    # The magic is adding `handle_parsing_errors=True` to let the agent 
    # recover from unparseable text:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent

###############################################################################
# 3) EXAMPLE USE
###############################################################################

if __name__ == "__main__":
    agent = create_wikidata_agent()

    #example question

    # question = (
    #     "Find all official languages in Belgium and switzerland using Wikidata. "
    #     "Return their labels and relevant IDs."
    # )

    # ask user input to get the questions
    question = input("Enter the question: ")

    print("\n=== Agent's Answer ===")
    try:
        response = agent.invoke(question)
        print(response)
    except Exception as e:
        print("[Error] Something else happened:", e)