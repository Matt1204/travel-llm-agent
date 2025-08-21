import uuid
import textwrap
from datetime import datetime

# from graph_1 import graph
from graph import graph
from langgraph.types import Command
from utils import chat_bot_print


def _print_new_messages_from_state(state, prev_counts):
    """Pretty-print only newly added messages for tracked keys using chat-bot style.

    Tracked keys: 'messages' and 'flight_discovery_messages'.
    """
    for key in ("messages", "flight_discovery_messages"):
        cur_msg_list = state.get(key)
        if not isinstance(cur_msg_list, list):
            continue
        prev_len = prev_counts.get(key, 0)
        cur_len = len(cur_msg_list)
        if cur_len != prev_len:
            new_items = cur_msg_list[prev_len:]
            for msg in new_items:
                try:
                    # Only print AIMessages that don't have tool calls
                    if hasattr(msg, "type") and msg.type == "ai":
                        # Check if the message has tool calls
                        has_tool_calls = False
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            has_tool_calls = True
                        elif hasattr(
                            msg, "additional_kwargs"
                        ) and msg.additional_kwargs.get("tool_calls"):
                            has_tool_calls = True

                        # Only print if it's an AI message without tool calls
                        if not has_tool_calls:
                            # Use our new chat_bot_print function
                            chat_bot_print(msg, is_human_msg=False)

                    # Still print human messages for context
                    elif hasattr(msg, "type") and msg.type == "human":
                        chat_bot_print(msg, is_human_msg=True)

                except Exception:
                    # Fallback safe print
                    print(f"[Error displaying message]: {str(msg)}")
        # Reset tracking if list shrank (e.g., resettable lists)
        prev_counts[key] = cur_len


if __name__ == "__main__":
    # Demonstrate the chat_bot_print function with different message types
    print("=" * 60)
    print("FLIGHT DISCOVERY CHAT ASSISTANT")
    print("=" * 60)

    # Demo with string input
    chat_bot_print(
        "Welcome! I'm your Flight Assistant.\n"
        "Type 'exit' to quit the chat session.\n",
        is_human_msg=False,
    )

    # Demo showing different message types that would come from your graph
    # chat_bot_print("Type 'exit' to quit the chat session.", is_human_msg=False)

    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "passenger_id": "3442 587242",
            "thread_id": thread_id,
        },
        "recursion_limit": 100,
    }

    while True:
        q = input("human input: ")
        if q == "exit":
            break

        # Initialize counters using current state before sending the new input
        pre_state = graph.get_state(config)
        prev_counts = {
            "messages": (
                len(pre_state.values.get("messages", []))
                if hasattr(pre_state, "values")
                else len(pre_state.get("messages", []))
            ),
            "flight_discovery_messages": (
                len(pre_state.values.get("flight_discovery_messages", []))
                if hasattr(pre_state, "values")
                else len(pre_state.get("flight_discovery_messages", []))
            ),
        }

        stream = graph.stream({"messages": ("user", q)}, config, stream_mode=["values"])
        for event in stream:
            state = event[-1]
            _print_new_messages_from_state(state, prev_counts)
        cur_state = graph.get_state(config)

        # if graph hasn't reach "END" node, but aborted due to "interrupt()"
        while cur_state.next:
            try:
                user_input = input(
                    "Do you approve of the above actions? Type 'y' to continue;"
                    " otherwise, explain your requested changed.\n\n"
                )
            except:
                user_input = "y"
            result = None
            if user_input.strip() == "y":
                # Just continue
                # Reset counters to current values before resuming
                pre_state = graph.get_state(config)
                prev_counts = {
                    "messages": (
                        len(pre_state.values.get("messages", []))
                        if hasattr(pre_state, "values")
                        else len(pre_state.get("messages", []))
                    ),
                    "flight_discovery_messages": (
                        len(pre_state.values.get("flight_discovery_messages", []))
                        if hasattr(pre_state, "values")
                        else len(pre_state.get("flight_discovery_messages", []))
                    ),
                }

                result = graph.invoke(
                    Command(resume=True),
                    config,
                    stream_mode=["values"],
                )
            else:
                # Satisfy the tool invocation by
                # providing instructions on the requested changes / change of mind
                # Reset counters to current values before resuming with instructions
                pre_state = graph.get_state(config)
                prev_counts = {
                    "messages": (
                        len(pre_state.values.get("messages", []))
                        if hasattr(pre_state, "values")
                        else len(pre_state.get("messages", []))
                    ),
                    "flight_discovery_messages": (
                        len(pre_state.values.get("flight_discovery_messages", []))
                        if hasattr(pre_state, "values")
                        else len(pre_state.get("flight_discovery_messages", []))
                    ),
                }

                result = graph.invoke(
                    Command(resume=user_input),
                    config,
                    stream_mode=["values"],
                )
            for event in result:
                state = event[-1]
                _print_new_messages_from_state(state, prev_counts)

            cur_state = graph.get_state(config)
            # print(cur_state)
