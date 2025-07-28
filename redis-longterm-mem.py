from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore
from langgraph.store.base import BaseStore
from langchain_google_genai import ChatGoogleGenerativeAI
import uuid
from langgraph.config import get_store

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
# llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0.0)

DB_URI = "redis://default:KNABCEEdtG7vSBNODLaOpY84OPYN2XGf@redis-10758.crce174.ca-central-1-1.ec2.redns.redis-cloud.com:10758"

with (
    RedisStore.from_conn_string(DB_URI) as store,
    RedisSaver.from_conn_string(DB_URI) as checkpointer,
):
    store.setup()
    checkpointer.setup()

    def call_model(
        state: MessagesState,
        config: RunnableConfig,
        *,
        store: BaseStore,
    ):
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)

        # Access the DB, retrieve user's memories, append memories to system prompt
        related_memory = read_user_memories(user_id, store, str(state["messages"][-1].content))
        system_msg = (
            f"You are a helpful assistant talking to the user. User info: {related_memory}"
        )

        # Check latest message, if it contains "remember", add new memory to DB
        last_message = state["messages"][-1]
        if "remember" in last_message.content.lower():
            # memory = "User likes to play football"
            memory = state["messages"][-1].content
            store.put(namespace, str(uuid.uuid4()), {"data": memory})

        # Call llm with memory from DB AND messages in current state
        response = llm.invoke(
            [{"role": "system", "content": system_msg}] + state["messages"]
        )
        return {"messages": response}
    
    def read_user_memories(user_id: str, store: BaseStore, query_natural_lang: str) -> str:
        namespace = ("memories", user_id)
        memories = store.search(namespace, query=query_natural_lang)
        info = "\n".join([d.value["data"] for d in memories])
        return info

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )

    config = {
        "configurable": {
            "thread_id": "1",
            "user_id": "1",
        }
    }
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "Hi! Remember: I like to play football"}]},
        config,
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()

    config = {
        "configurable": {
            "thread_id": "2",
            "user_id": "1",
        }
    }

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "what is my favorite sport?"}]},
        config,
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()
