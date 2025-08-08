#!/usr/bin/env python3
"""
Simple Flight Search Agent Runner

A simplified version that directly uses the flight search agent for multi-round conversations.
"""

import json
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from flight_search_agent import flight_search_agent
from intent_model import FlightRequirements


class SimpleFlightSearchRunner:
    def __init__(self):
        """Initialize the simple flight search runner."""
        self.messages = []
        self.flight_requirements = None
        
    def create_sample_requirements(self) -> FlightRequirements:
        """Create sample flight requirements for testing."""
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
    
    def start_conversation(self, use_sample: bool = False):
        """Start a new conversation."""
        if use_sample:
            self.flight_requirements = self.create_sample_requirements()
        else:
            self.flight_requirements = FlightRequirements()
        
        print("🚀 Simple Flight Search Agent started!")
        print("💬 You can now have a conversation with the agent.")
        print("📝 Type 'exit' to quit, 'help' for commands.")
        print("=" * 50)
        
        if self.flight_requirements:
            print("📋 Current Flight Requirements:")
            print(json.dumps(self.flight_requirements.to_json(), indent=2))
            print("=" * 50)
    
    def send_message(self, message: str) -> str:
        """Send a message to the agent and get response."""
        # Add user message
        self.messages.append(HumanMessage(content=message))
        
        try:
            # Create state for the agent
            state = {
                "messages": self.messages,
                "flight_requirements": self.flight_requirements
            }
            
            # Invoke the agent
            result = flight_search_agent.invoke(state)
            
            # Add agent response to messages
            if hasattr(result, 'content') and result.content:
                self.messages.append(AIMessage(content=result.content))
                return result.content
            elif hasattr(result, 'tool_calls') and result.tool_calls:
                # Handle tool calls
                tool_names = [call.get('name', 'unknown') for call in result.tool_calls]
                response = f"🔧 Using tools: {tool_names}"
                self.messages.append(AIMessage(content=response))
                return response
            else:
                return "🤔 Agent is thinking..."
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def show_current_requirements(self):
        """Display current flight requirements."""
        if self.flight_requirements:
            print("\n📋 Current Flight Requirements:")
            print(json.dumps(self.flight_requirements.to_json(), indent=2))
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
    
    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
        print("🗑️  Conversation history cleared.")


def main():
    """Main function to run the simple flight search agent."""
    runner = SimpleFlightSearchRunner()
    
    print("🎯 Simple Flight Search Agent")
    print("=" * 50)
    print("Choose an option:")
    print("1. Start with sample requirements")
    print("2. Start with empty requirements")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == "1":
        runner.start_conversation(use_sample=True)
    else:
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
                runner.clear_history()
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