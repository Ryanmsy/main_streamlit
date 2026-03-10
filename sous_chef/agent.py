"""
Importable agent setup for the Streamlit page.
Defines the LangGraph Sous Chef agent without executing it.
"""
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import requests
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

sous_chef_prompt = """
You are a highly skilled Sous Chef Agent.
Your goal is to help users find recipes based on the ingredients they have.

GUIDELINES:
1. Always check if the user has provided specific ingredients.
2. If ingredients are missing or the request is vague, ask clarifying questions.
3. When you find a recipe, summarize it clearly: Name, Time, and Key Ingredients.
4. Be encouraging and helpful, like a friendly cooking instructor.

You have access to tools to search for recipes. USE THEM.
Do not make up recipes unless explicitly asked to 'create' one from scratch.
"""

@tool
def search_by_ingredients(ingredients: list[str]) -> str:
    """
    Search for recipes based on a list of available ingredients.

    Args:
        ingredients: A list of ingredient names (e.g., ["chicken", "rice"]).

    Returns:
        A string containing a list of matching recipes with titles and IDs.
    """
    SPOON_API_KEY = os.getenv("SPOON_API_KEY")
    formatted_ingredients = ",".join(ingredients)
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = {
        "ingredients": formatted_ingredients,
        "number": 5,
        "apiKey": SPOON_API_KEY,
    }
    try:
        response = requests.get(url=url, params=params)
        response.raise_for_status()
        return str(response.json())
    except requests.exceptions.RequestException as e:
        return f"Error fetching recipes: {e}"

tools = [search_by_ingredients]
tool_node = ToolNode(tools)

def _build_app():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: AgentState) -> Literal["tools", END]:
        if state["messages"][-1].tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile()

def get_app():
    return _build_app()
