import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from sous_chef.agent import get_app, sous_chef_prompt

@st.cache_resource
def load_app():
    return get_app()

# --- UI ---
st.title("Sous Chef Agent")
st.write("Tell me what ingredients you have and I'll find real recipes for you.")

st.sidebar.markdown("""
**About this project**

An AI agent powered by **Gemini 2.5 Flash** using LangGraph.

It searches the **Spoonacular API** for real recipes based on your available ingredients.
""")

user_input = st.text_area(
    "What ingredients do you have?",
    height=120,
    placeholder="e.g., I have chicken breast, garlic, and lemon."
)

if st.button("Find Recipes", type="primary", use_container_width=True):
    if not user_input:
        st.warning("Please describe your ingredients first.")
    else:
        try:
            with st.spinner("The chef is thinking..."):
                final_state = load_app().invoke({
                    "messages": [
                        SystemMessage(content=sous_chef_prompt),
                        HumanMessage(content=user_input),
                    ]
                })
            st.divider()
            st.subheader("Sous Chef says:")
            st.write(final_state["messages"][-1].content)
        except ValueError as e:
            st.error(str(e))
