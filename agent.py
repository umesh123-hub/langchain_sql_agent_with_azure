import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
import os
import pandas as pd
load_dotenv()
class SQLCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_query = None
    def on_tool_start(self, tool, input_str, **kwargs):
        if isinstance(tool, dict):
            tool_name = tool.get('name')
        else:
            tool_name = getattr(tool, 'name', None)
        if tool_name == "sql_db_query":
            self.sql_query = input_str
# --------------------------------------------- Database Connection ---------------------------------------------------
username = os.getenv("DB_USERNAME")
password = quote_plus(os.getenv("DB_PASSWORD"))
host = os.getenv("DB_HOST")
port = 5432
database = os.getenv("DB_NAME")
connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
engine = create_engine(connection_string)
db = SQLDatabase(engine=engine)
# -------------------------------------------- LangChain LLM setup -----------------------------------------------------
llm = AzureChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT"),
    temperature=0,
)
# ----------------------------------------- Create SQL agent ------------------------------------------------------------
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    handle_parsing_errors=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
# ----------------------------------------- Streamlit UI setup ----------------------------------------------------------
st.set_page_config(page_title="SQL Agent", layout="centered")
st.title("💬 SQL Agent")
st.markdown("Ask your database questions. Previous Q&A will be remembered during this session.")
# ----------------------------------------- Initialize chat history in session_state -------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sql_query" not in st.session_state:
    st.session_state.sql_query = None
def new_chat():
    st.session_state.chat_history = []
    st.session_state.sql_query = None
st.sidebar.button("New Chat", on_click=new_chat)
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
prompt = st.chat_input("Say something")
if prompt:
    
    modified_prompt = f"{prompt} create query for above and give optimised sql query only"
    st.session_state.chat_history.append({"role": "user", "content": prompt, "query": None})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        user_question = modified_prompt.strip()
        try:
            context = ""
            for chat in st.session_state.chat_history:
                if chat["role"] == "user":
                    context += f"Q: {chat['content']}\n"
                elif chat["role"] == "assistant":
                    context += f"A: {chat['content']}\n"
            full_prompt = f"{context}Q: {user_question}"
            sql_callback = SQLCallbackHandler()
            response = agent_executor.run(full_prompt, callbacks=[sql_callback])
            if sql_callback.sql_query:
                st.session_state.sql_query = sql_callback.sql_query
                st.code(sql_callback.sql_query, language="sql")
            else:
                st.warning("No SQL query was captured.")
            full_response = response
        except Exception as e:
            full_response = f"⚠️ Error: {str(e)}"
        message_placeholder.markdown(full_response)
    st.session_state.chat_history.append({"role": "assistant", "content": full_response, "query": st.session_state.sql_query})
if st.session_state.sql_query:
    if st.button("Execute Query"):
        try:
            query_without_limit = st.session_state.sql_query.split("LIMIT")[0].strip()
            if not query_without_limit.endswith(";"):
                query_without_limit += ";"
            with engine.connect() as connection:
                result = connection.execute(text(query_without_limit))
                rows = result.fetchall()
                columns = result.keys()
            df = pd.DataFrame(rows, columns=columns)
            st.success("✅ Query Results Without Limit")
            st.dataframe(df)
        except Exception as e:
            st.error(f"⚠️ Error executing query: {str(e)}")
with st.sidebar:
    st.markdown("### 🧠 Chat History")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Agent:** {message['content']}")
            if message["query"]:
                st.code(message["query"], language="sql")
