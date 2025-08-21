import textwrap
from datetime import datetime

def chat_bot_print(message, is_human_msg, width=80):
    """
    Print messages in a chat-bot style format with visual appeal.

    Args:
        message: Can be an AIMessage object, string, dict, or any object with content
        is_human_msg: Boolean indicating if the message is from a human
        width: Maximum width of the chat bubble (default: 80)
    """
    # Extract content from message - handle various input types
    if hasattr(message, "content"):
        # Handle AIMessage or similar objects
        content = str(message.content)
    elif isinstance(message, dict):
        # Handle dict inputs from graph nodes
        content = message.get("content", message.get("message", str(message)))
    elif isinstance(message, (str, int, float)):
        # Handle direct string/number inputs
        content = str(message)
    else:
        # Handle any other object type
        content = str(message)

    # Colors for terminal output
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Get current timestamp
    timestamp = datetime.now().strftime("%H:%M")

    # Different styles for user vs AI
    if is_human_msg:
        # User messages: cyan color with person emoji
        header = f"{BOLD}{CYAN}👤 Human{RESET} {DIM}• {timestamp}{RESET}"
        border_color = CYAN
        bubble_chars = ("╭", "─", "╮", "│", "╰", "╯")
    else:
        # AI messages: green color with robot emoji
        header = f"{BOLD}{GREEN}🤖 AI Assistant{RESET} {DIM}• {timestamp}{RESET}"
        border_color = GREEN
        bubble_chars = ("┌", "─", "┐", "│", "└", "┘")

    # Preserve original formatting including newlines
    # Split by existing newlines first
    original_lines = content.split("\n")
    wrapped_lines = []

    # Account for border characters and padding
    content_width = width - 6  # 2 for borders + 4 for padding

    for line in original_lines:
        if len(line) <= content_width:
            # Line fits, keep as is
            wrapped_lines.append(line)
        else:
            # Line too long, wrap it but preserve intentional breaks
            wrapped_parts = textwrap.wrap(line, width=content_width)
            wrapped_lines.extend(wrapped_parts)

    # Handle empty content
    if not wrapped_lines:
        wrapped_lines = [""]

    # Calculate the actual width needed
    max_line_length = max(len(line) for line in wrapped_lines)
    bubble_width = min(max_line_length + 4, width)  # 4 for padding

    # Print the chat bubble with appropriate style
    print(f"\n{header}")
    print(
        f"{border_color}{bubble_chars[0]}{bubble_chars[1] * bubble_width}{bubble_chars[2]}{RESET}"
    )

    for line in wrapped_lines:
        padding = bubble_width - len(line) - 2  # 2 for side borders
        print(
            f"{border_color}{bubble_chars[3]}{RESET} {line}{' ' * padding} {border_color}{bubble_chars[3]}{RESET}"
        )

    print(
        f"{border_color}{bubble_chars[4]}{bubble_chars[1] * bubble_width}{bubble_chars[5]}{RESET}"
    )
    print()  # Add some spacing after the message

