# Flight Search Agent Runner

This project provides two different ways to run the flight search agent for multi-round conversations.

## Files

1. **`run_flight_search_agent.py`** - Full-featured runner with graph-based architecture
2. **`simple_flight_search_runner.py`** - Simplified runner for direct agent interaction

## Prerequisites

Make sure you have all the required dependencies installed:

```bash
pip install -r requirement.txt
```

You'll also need to set up your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Option 1: Simple Runner (Recommended for testing)

The simple runner provides a straightforward way to interact with the flight search agent:

```bash
python simple_flight_search_runner.py
```

**Features:**
- Direct interaction with the flight search agent
- Sample flight requirements for testing
- Simple command interface
- No complex graph setup required

**Commands:**
- `help` - Show available commands
- `requirements` - Display current flight requirements
- `clear` - Clear conversation history
- `sample` - Load sample flight requirements
- `exit` - Exit the conversation

### Option 2: Full Graph Runner

The full runner uses the complete graph architecture:

```bash
python run_flight_search_agent.py
```

**Features:**
- Complete graph-based architecture
- Database integration for flight requirements
- More sophisticated state management
- Better for production use

## Sample Conversation

Here's an example of how to interact with the agent:

```
🎯 Simple Flight Search Agent
==================================================
Choose an option:
1. Start with sample requirements
2. Start with empty requirements

Enter your choice (1-2): 1

🚀 Simple Flight Search Agent started!
💬 You can now have a conversation with the agent.
📝 Type 'exit' to quit, 'help' for commands.
==================================================
📋 Current Flight Requirements:
{
  "departure_airport": {
    "priority_1": "YUL"
  },
  "arrival_airport": {
    "priority_1": "YYZ"
  },
  "departure_time": {
    "priority_1": ["2024-01-15 10:00:00", "2024-01-22 10:00:00"]
  },
  "budget": {
    "priority_1": 500
  }
}
==================================================

👤 You: Can you find flights from Montreal to Toronto?

🤖 Agent: I'll search for flights from Montreal (YUL) to Toronto (YYZ) within your specified time window and budget.

🔧 Using tools: ['search_flights']

👤 You: What are the results?

🤖 Agent: Here are the flights I found for you:

**Base Filter Results (Priority 1 criteria):**
- Flight: AC123, Departure: 2024-01-16 08:30, Arrival: 2024-01-16 10:15
  - Economy: $450
  - Business: $650

- Flight: AC456, Departure: 2024-01-18 14:00, Arrival: 2024-01-18 15:45
  - Economy: $380
  - Business: $580

Would you like me to search with different criteria or modify any of your requirements?
```

## Flight Requirements Structure

The agent works with flight requirements that have priority levels:

```json
{
  "departure_airport": {
    "priority_1": "YUL",  // Most preferred
    "priority_2": "YHU"   // Alternative option
  },
  "arrival_airport": {
    "priority_1": "YYZ"
  },
  "departure_time": {
    "priority_1": ["2024-01-15 10:00:00", "2024-01-22 10:00:00"],
    "priority_2": ["2024-01-15 08:00:00", "2024-01-25 18:00:00"]
  },
  "budget": {
    "priority_1": 500,
    "priority_2": 800
  }
}
```

## How It Works

1. **Priority-based Search**: The agent first searches using priority_1 criteria
2. **Progressive Relaxation**: If results are unsatisfactory, it tries priority_2, priority_3, etc.
3. **Multi-filter Results**: Presents results from at least 3 different filter combinations
4. **Interactive Refinement**: Allows users to modify requirements and re-search

## Troubleshooting

### Common Issues

1. **OpenAI API Key**: Make sure your API key is set correctly
2. **Database Connection**: The full runner requires a database with flight data
3. **Dependencies**: Ensure all packages are installed correctly

### Error Messages

- `❌ Error: Could not load requirements with ID: xxx` - Database connection issue
- `❌ Error: Invalid datetime value` - Date format issue
- `🤔 Agent is thinking...` - Normal processing, wait for response

## Development

To modify the agent behavior, edit the following files:

- `flight_search_agent.py` - Main agent logic and tools
- `intent_model.py` - Flight requirements data model
- `simple_flight_search_runner.py` - Simple runner interface
- `run_flight_search_agent.py` - Full graph runner

## Database Setup

If using the full runner with database integration, ensure you have:

1. SQLite database with flight data
2. Proper table structure for flights and ticket_flights
3. Flight requirements table for persistence

The simple runner doesn't require database setup and works with in-memory requirements. 