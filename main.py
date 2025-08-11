import uuid

# from graph_1 import graph
from graph import graph
from langgraph.types import Command


if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "passenger_id": "3442 587242",
            "thread_id": thread_id,
        }
    }

    while True:
        q = input("human input: ")
        if q == "exit":
            break

        stream = graph.stream({"messages": ("user", q)}, config, stream_mode=["values"])
        for event in stream:
            event[-1]["messages"][-1].pretty_print()  # chunk: contains accumulated messages
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
                result = graph.invoke(
                    Command(resume=True),
                    config,
                    stream_mode=["values"],
                )
            else:
                # Satisfy the tool invocation by
                # providing instructions on the requested changes / change of mind
                result = graph.invoke(
                    Command(resume=user_input),
                    config,
                    stream_mode=["values"],
                )
            for event in result:
                event[-1]["messages"][-1].pretty_print()

            cur_state = graph.get_state(config)
            # print(cur_state)
