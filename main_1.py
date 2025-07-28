import uuid

from langchain_core.language_models.fake_chat_models import FakeChatModel
from graph_1 import graph
from langgraph.types import Command


def run_question(question: str):
    """Run the graph for a single user question, handling sensitive tool interrupts."""
    # try:
    # Normal streaming execution
    stream = graph.stream(
        {"messages": ("user", question)}, config, stream_mode=["values"]
    )
    for event in stream:
        event["messages"][-1].pretty_print()

    # except GraphRunInterrupted as interrupt:
    #     # We were interrupted right before a sensitive tool call
    #     state = interrupt.state
    #     # If interrupted before a (sensitive) tool call, ask user if they want to proceed
    #     if len(state["messages"][-1].tool_call) > 0:
    #         print("HITTT interrupt before tool call")
    #         tool_call = state["messages"][-1].tool_call[0]
    #         answer = (
    #             input(f"tool '{tool_call['name']}' requested. Proceed? (y/n): ")
    #             .strip()
    #             .lower()
    #         )

    #         if answer == "y":
    #             # Resume execution and actually run the sensitive tool
    #             resumed = graph.stream(None, config, stream_mode=["values"])
    #             # resumed = graph.resume(state, config, stream_mode=["values"], run_id=interrupt.run_id)
    #         else:
    #             resumed = graph.stream(
    #                 {
    #                     "messages": [
    #                         ToolMessage(
    #                             tool_call_id=state["messages"][-1].tool_call[0]["id"],
    #                             content=f"tool call denied by user. Continue assisting, accounting for the user's input.",
    #                         )
    #                     ]
    #                 },
    #                 config,
    #             )
    #         # Continue streaming the remaining events after resumption
    #         for event in resumed:
    #             event["messages"][-1].pretty_print()


if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "passenger_id": "3442 587242",
            "thread_id": thread_id,
        }
    }

    # Example conversation
    demo_questions = [
        "Hi there, what time is my flight?",
        "Am I allowed to update my flight to something sooner? I want to leave later today.",
        "Update my flight to sometime next week then",
        "The next available option is great",
    ]

    for q in demo_questions:
        # run_question(q)
        stream = graph.stream({"messages": ("user", q)}, config, stream_mode=["values"])
        for event in stream:
            # print(event)
            event[-1]["messages"][
                -1
            ].pretty_print()  # chunk: contains accumulated messages
        cur_state = graph.get_state(config)
        # print(cur_state["messages"][-1].pretty_print())

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
                result = graph.stream(
                    Command(resume=True),
                    config,
                    stream_mode=["updates"],
                )
            else:
                # Satisfy the tool invocation by
                # providing instructions on the requested changes / change of mind
                result = graph.stream(
                    Command(resume=False),
                    config,
                    stream_mode=["updates"],
                )
            for event in result:
                event["messages"][-1].pretty_print()
            cur_state = graph.get_state(config)
            print(cur_state)
