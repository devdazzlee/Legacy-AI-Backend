#!/usr/bin/env python3
"""
Test script for the chatbot endpoint
"""
import requests
import json

# Test the chatbot endpoint
def test_chatbot():
    url = "http://localhost:8000/chatbot/ask"
    
    # Test data
    test_data = {
        "message": "How do I fill out the outings & appointments section?",
        "context": {
            "currentSection": "Outings & Appointments",
            "currentStep": 1,
            "currentNoteStep": 1,
            "selectedServices": ["Transportation", "Appointment Scheduling"]
        }
    }
    
    try:
        response = requests.post(url, json=test_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            chatbot_response = response.json().get("response", "")
            if "I'm here to help with" in chatbot_response and "Please ask me a specific question" in chatbot_response:
                print("\n❌ ERROR: Still getting fallback response!")
                print("This means there's still an error in the chatbot function.")
            else:
                print("\n✅ SUCCESS: Getting proper AI response!")
        else:
            print(f"\n❌ ERROR: HTTP {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server. Make sure FastAPI is running on localhost:8000")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_chatbot()
