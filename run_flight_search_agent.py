#!/usr/bin/env python3
"""
Flight Search Agent Runner

This script allows you to have multi-round conversations with the flight search agent.
It creates a simple interface to interact with the agent and manage conversation state.
"""

import uuid
import json
from typing import Dict, Any
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

from flight_search_agent import (
    flight_search_entry_node,
    flight_search_agent,
    register_flight_search_graph
)
from primary_assistant_chain import State
from intent_model import FlightRequirements
from intent import load_flight_requirements_from_db


class FlightSearchAgentRunner:
    def __init__(self):
        """Initialize the flight search agent runner."""
        self.checkpointer = InMemorySaver()
        self.thread_id = str(uuid.uuid4())
        self.config = {
            "configurable": {
                "thread_id": self.thread_id,
            }
        }
        
        # Build the graph
        self.graph_builder = StateGraph(State)
        register_flight_search_graph(self.graph_builder)
        self.graph = self.graph_builder.compile(checkpointer=self.checkpointer)
        
        # Initialize state
        self.current_state = {
            "messages": [],
            "flight_requirements": None,
            "requirement_id": None
        }
    
    def create_sample_requirements(self) -> FlightRequirements:
        """Create sample flight requirements for testing."""
        from datetime import datetime, timedelta
        
        # Create sample requirements
        requirements = FlightRequirements()
        
        # Add departure airport (Montreal)
        requirements.departure_airport["priority_1"] = "YUL"
        
        # Add arrival airport (Toronto)
        requirements.arrival_airport["priority_1"] = "YYZ"
        
        # Add departure time window (next week)
        tomorrow = datetime.now() + timedelta(days=1)
        next_week = tomorrow + timedelta(days=7)
        requirements.departure_time["priority_1"] = [tomorrow, next_week]
        
        # Add budget (optional)
        requirements.budget["priority_1"] = 500
        
        return requirements
    
    def start_conversation(self, requirement_id: str = None, use_sample: bool = False):
        """Start a new conversation with the flight search agent."""
        if use_sample:
            # Use sample requirements
            requirements = self.create_sample_requirements()
            self.current_state["flight_requirements"] = requirements
            self.current_state["requirement_id"] = None
        elif requirement_id:
            # Load requirements from database
            requirements = load_flight_requirements_from_db(requirement_id)
            if requirements:
                self.current_state["flight_requirements"] = requirements
                self.current_state["requirement_id"] = requirement_id
            else:
                print(f"❌ Could not load requirements with ID: {requirement_id}")
                return
        else:
            # Start with empty requirements
            self.current_state["flight_requirements"] = FlightRequirements()
            self.current_state["requirement_id"] = None
        
        print("🚀 Flight Search Agent started!")
        print("💬 You can now have a conversation with the agent.")
        print("📝 Type 'exit' to quit, 'help' for commands.")
        print("=" * 50)
        
        if self.current_state["flight_requirements"]:
            print("📋 Current Flight Requirements:")
            print(json.dumps(self.current_state["flight_requirements"].to_json(), indent=2))
            print("=" * 50)
    
    def send_message(self, message: str) -> str:
        """Send a message to the agent and get response."""
        # Add user message to state
        self.current_state["messages"].append(HumanMessage(content=message))
        
        try:
            # Run the graph
            stream = self.graph.stream(
                self.current_state,
                self.config,
                stream_mode=["values"]
            )
            
            # Collect the response
            response = ""
            for event in stream:
                if "messages" in event[-1]:
                    messages = event[-1]["messages"]
                    if messages and len(messages) > 0:
                        last_message = messages[-1]
                        if hasattr(last_message, 'content'):
                            response = last_message.content
                        elif hasattr(last_message, 'tool_calls'):
                            # Handle tool calls
                            response = f"🔧 Using tools: {[call['name'] for call in last_message.tool_calls]}"
            
            # Update current state
            current_graph_state = self.graph.get_state(self.config)
            self.current_state = current_graph_state.values
            
            return response if response else "🤔 Agent is thinking..."
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def show_current_requirements(self):
        """Display current flight requirements."""
        if self.current_state.get("flight_requirements"):
            print("\n📋 Current Flight Requirements:")
            print(json.dumps(self.current_state["flight_requirements"].to_json(), indent=2))
        else:
            print("\n📋 No flight requirements loaded.")
    
    def show_help(self):
        """Show help information."""
        print("\n📖 Available Commands:")
        print("  help                    - Show this help")
        print("  requirements            - Show current flight requirements")
        print("  exit                    - Exit the conversation")
        print("  clear                   - Clear conversation history")
        print("  sample                  - Load sample flight requirements")
        print("\n💡 You can also just type your questions normally!")


def main():
    """Main function to run the flight search agent."""
    runner = FlightSearchAgentRunner()
    
    print("🎯 Flight Search Agent")
    print("=" * 50)
    print("Choose an option:")
    print("1. Start with sample requirements")
    print("2. Start with empty requirements")
    print("3. Load requirements by ID")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        runner.start_conversation(use_sample=True)
    elif choice == "2":
        runner.start_conversation()
    elif choice == "3":
        req_id = input("Enter requirement ID: ").strip()
        runner.start_conversation(requirement_id=req_id)
    else:
        print("❌ Invalid choice. Starting with empty requirements.")
        runner.start_conversation()
    
    # Main conversation loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() == 'exit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                runner.show_help()
                continue
            elif user_input.lower() == 'requirements':
                runner.show_current_requirements()
                continue
            elif user_input.lower() == 'clear':
                runner.current_state["messages"] = []
                print("🗑️  Conversation history cleared.")
                continue
            elif user_input.lower() == 'sample':
                runner.start_conversation(use_sample=True)
                continue
            elif not user_input:
                continue
            
            # Send message to agent
            response = runner.send_message(user_input)
            print(f"\n🤖 Agent: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main() 