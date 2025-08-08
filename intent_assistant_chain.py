
# --- Intent elicitation agent (custom runnable sequence) --------------------
# NOTE: This migrates the prebuilt `create_react_agent` into an LCEL sequence
# so we can host it here and reuse it as a node in other graphs.
from typing import Any
from langchain_core.runnables import (
    RunnableLambda,
    RunnableBranch,
    RunnablePassthrough,
    RunnableConfig,
)
from langgraph.prebuilt import ToolNode  # Assumes default expects `messages` key

def build_intent_elicitation_sequence(model, tools, prompt_fn):
    """
    Build a ReAct-style runnable sequence for the intent elicitation workflow.

    The sequence:
    1) Formats messages via `prompt_fn(state, config)` (uses state like
       `flight_requirements`, `flight_requirements_messages`, etc.).
    2) Calls the LLM bound with tools.
    3) If the LLM requests a tool, we route into a ToolNode.
       - We map the state's message stream into a standard `messages` key
         (alias added in our state) and append the AI message before tool exec.
       - We mirror resulting ToolMessage into BOTH `messages` and
         `flight_requirements_messages` so prompts remain consistent.

    Returns a runnable that **emits state updates** compatible with a LangGraph
    node (dict of partial state).
    """

    def _call_llm(state: dict, config: RunnableConfig) -> dict:
        # The prompt_fn returns a list[BaseMessage]
        msgs = prompt_fn(state, config)
        ai_msg = model.bind_tools(tools).invoke(msgs, config)
        return {"ai_msg": ai_msg, "state": state}

    def _needs_tool(payload: dict) -> bool:
        ai = payload["ai_msg"]
        # Some chat models use `.tool_calls`, others `.additional_kwargs`.
        # This check is defensive.
        return bool(getattr(ai, "tool_calls", None))

    def _prep_toolnode_input(payload: dict) -> dict:
        state = payload["state"].copy()
        prior = state.get("messages", [])  # alias stream
        # Append the AIMessage so ToolNode can read its tool_calls
        return {
            **state,
            "messages": prior + [payload["ai_msg"]],
        }

    tool_node = ToolNode(tools=tools)  # TODO: If a future API allows
    # configuring the message key explicitly, we could pass it here instead of
    # aliasing to `messages`.

    def _after_tool(tool_out: dict) -> dict:
        """Map ToolNode output to our state updates.

        NOTE: I'm assuming ToolNode returns any `Command(update=...)` updates
        from tools alongside `messages`. If that ever changes, we may need to
        explicitly forward those updates from the executor. (Not 100% sure.)
        """
        updates: dict[str, Any] = {}
        if "messages" in tool_out and tool_out["messages"]:
            last = tool_out["messages"][-1]
            updates["messages"] = [last]
            updates["flight_requirements_messages"] = [last]
        # Forward any other updates from tools (e.g., our Command(update=...))
        for k, v in tool_out.items():
            if k != "messages":
                updates[k] = v
        return updates

    def _no_tool_updates(payload: dict) -> dict:
        ai = payload["ai_msg"]
        # Mirror AI message into both streams so prompts and ToolNode stay in sync
        return {"messages": [ai], "flight_requirements_messages": [ai]}

    sequence = (
        RunnableLambda(_call_llm)
        | RunnableBranch(
            (_needs_tool, RunnableLambda(_prep_toolnode_input) | tool_node | RunnableLambda(_after_tool)),
            (lambda _x: True, RunnableLambda(_no_tool_updates)),
        )
    )
    return sequence
