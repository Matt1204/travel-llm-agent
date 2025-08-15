"""
Simple Custom Tool Node for Debugging

A minimal implementation to bypass the official ToolNode for debugging purposes.
Handles tool execution with basic error handling and logging.
"""

from typing import Dict, Any, List, Union
import uuid
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from fd_baseline_chain import baseline_tools


def custom_tool_node(tools: List, messages_key: str = "messages"):
    """
    Create a simple custom tool node that bypasses the official ToolNode.

    Args:
        tools: List of tools to execute
        messages_key: Key in state where messages are stored

    Returns:
        A function that can be used as a graph node
    """
    # Create tool name to function mapping; ignore entries without a `.name` (e.g., sentinel Pydantic models)
    tool_map = {}
    for tool in tools:
        name = getattr(tool, "name", None)
        if name:
            tool_map[name] = tool
        else:
            # Non-executable tool spec (e.g., Pydantic model used only for routing); skip
            try:
                debug_name = getattr(tool, "__name__", str(tool))
            except Exception:
                debug_name = str(tool)
            print(
                f"🔧 [CustomToolNode] Skipping non-executable tool without .name: {debug_name}"
            )

    def tool_node_func(
        state: Dict[str, Any], config: RunnableConfig = None
    ) -> Dict[str, Any]:
        """Execute tool calls from the last message"""
        print(
            f"🔧 [CustomToolNode] Starting execution with {len(tools)} available tools"
        )

        # Get messages from state
        messages = state.get(messages_key, [])
        if not messages:
            print("🔧 [CustomToolNode] No messages found in state")
            return {messages_key: []}

        # Find the last message with tool calls
        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            print("🔧 [CustomToolNode] No tool calls found in last message")
            return {messages_key: []}

        print(
            f"🔧 [CustomToolNode] Found {len(last_message.tool_calls)} tool calls to execute"
        )

        # Execute each tool call
        tool_results = []
        state_updates: Dict[str, Any] = {}
        final_goto = None

        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name")
            tool_call_id = (
                tool_call.get("id")
                or tool_call.get("tool_call_id")
                or str(uuid.uuid4())
            )
            tool_args = tool_call.get("args", {})
            if not tool_name:
                error_msg = "Tool call missing 'name'"
                print(f"🔧 [CustomToolNode] ERROR: {error_msg}")
                tool_results.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                )
                continue

            print(f"🔧 [CustomToolNode] Executing: {tool_name} (ID: {tool_call_id})")

            # Check if tool exists
            if tool_name not in tool_map:
                error_msg = f"Tool '{tool_name}' not found in available tools"
                print(f"🔧 [CustomToolNode] ERROR: {error_msg}")

                tool_results.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                )
                continue

            try:
                # Get the tool function
                tool_func = tool_map[tool_name]

                # Execute the tool
                print(f"🔧 [CustomToolNode] Calling {tool_name} with args: {tool_args}")
                payload = {
                    "type": "tool_call",
                    "name": tool_name,
                    "tool_call_id": tool_call_id,
                    "args": {
                        **(tool_args or {}),
                        "tool_call_id": tool_call_id,
                        "state": state,
                        "config": config,
                    },
                }
                # result = tool_func.invoke(payload, config=config)
                result = tool_func.invoke(
                    {
                        **(tool_args or {}),
                        "tool_call_id": tool_call_id,  # required because your tool expects InjectedToolCallId
                        "state": state,
                        "config": config,
                    }
                )

                # Handle different result types
                if isinstance(result, Command):
                    # Merge state updates from Command
                    if hasattr(result, "update") and result.update:
                        for key, value in result.update.items():
                            if (
                                key in state_updates
                                and isinstance(state_updates[key], list)
                                and isinstance(value, list)
                            ):
                                state_updates[key].extend(value)
                            else:
                                state_updates[key] = value
                    # Capture goto if provided
                    if hasattr(result, "goto") and result.goto:
                        if final_goto and final_goto != result.goto:
                            print(
                                f"🔧 [CustomToolNode] Multiple goto targets detected: {final_goto} -> {result.goto}. Using latest."
                            )
                        final_goto = result.goto
                elif isinstance(result, dict):
                    # Treat dict as direct state updates
                    for key, value in result.items():
                        if (
                            key in state_updates
                            and isinstance(state_updates[key], list)
                            and isinstance(value, list)
                        ):
                            state_updates[key].extend(value)
                        else:
                            state_updates[key] = value
                elif isinstance(result, ToolMessage):
                    tool_results.append(result)
                else:
                    # Convert anything else to a ToolMessage for visibility
                    tool_results.append(
                        ToolMessage(content=str(result), tool_call_id=tool_call_id)
                    )

                print(f"🔧 [CustomToolNode] Tool {tool_name} executed successfully")

            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                print(f"🔧 [CustomToolNode] ERROR: {error_msg}")

                # Print traceback for debugging
                import traceback

                traceback.print_exc()

                tool_results.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                )

        print(
            f"🔧 [CustomToolNode] Completed execution, returning {len(tool_results)} results"
        )

        # Return the tool results as new messages, merged with state updates
        if tool_results:
            if messages_key in state_updates:
                try:
                    # Extend if existing updates already contain messages
                    state_updates[messages_key].extend(tool_results)
                except Exception:
                    state_updates[messages_key] = tool_results
            else:
                state_updates[messages_key] = tool_results
        # If any tool requested a goto, return a Command so the graph can route
        if final_goto:
            print(f"🔧 [CustomToolNode] Returning Command with goto: {final_goto}")
            return Command(goto=final_goto, update=state_updates)
        return state_updates

    return tool_node_func


def create_simple_tool_node(tools: List, messages_key: str = "messages"):
    """
    Simple factory function to create a custom tool node.

    Args:
        tools: List of tools to make available
        messages_key: Key in state where messages are stored

    Returns:
        A tool node function
    """
    return custom_tool_node(tools, messages_key)


# Example usage for debugging:
if __name__ == "__main__":
    # This is just for testing the module
    create_simple_tool_node(baseline_tools)
