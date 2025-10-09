import requests
import json

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}

def test_create_thread():
    """Test creating a new thread"""
    try:
        response = requests.post(f"{BASE_URL}/threads/create")
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}

def test_send_message(thread_id):
    """Test sending a message"""
    try:
        data = {
            "thread_id": thread_id,
            "message": "Hello, how are you?"
        }
        response = requests.post(f"{BASE_URL}/messages/send", json=data)
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}

def test_get_thread_messages(thread_id):
    """Test getting thread messages"""
    try:
        response = requests.get(f"{BASE_URL}/threads/{thread_id}/messages")
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}

def test_validate_response():
    """Test response validation endpoint"""
    try:
        # Test with a short response
        data = {
            "response": "good",
            "question_type": "day_description"
        }
        response = requests.post(f"{BASE_URL}/validate-response", json=data)
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": str(e)}

def test_existing_endpoints():
    """Test the existing endpoints"""
    try:
        # Test root endpoint
        response = requests.get(f"{BASE_URL}/")
        root_status = response.status_code
        root_data = response.json()
        
        # Test items endpoint
        response = requests.get(f"{BASE_URL}/items/123?q=test")
        items_status = response.status_code
        items_data = response.json()
        
        return {
            "root": {"status": root_status, "data": root_data},
            "items": {"status": items_status, "data": items_data}
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("🚀 Testing Backend Azure OpenAI Chat API")
    print("==================================================")
    
    # Test existing endpoints
    print("0. Testing Existing Endpoints")
    existing_results = test_existing_endpoints()
    print(f"Root endpoint: {existing_results.get('root', {})}")
    print(f"Items endpoint: {existing_results.get('items', {})}")
    print()
    
    # 1. Health Check
    health_status, health_response = test_health_check()
    print(f"1. Health Check\nStatus: {health_status}\nResponse: {json.dumps(health_response, indent=2)}\n")
    
    # 1.5. Response Validation Test
    validation_status, validation_response = test_validate_response()
    print(f"1.5. Response Validation Test\nStatus: {validation_status}\nResponse: {json.dumps(validation_response, indent=2)}\n")
    
    # 2. Creating Thread
    thread_status, thread_response = test_create_thread()
    thread_id = None
    if thread_status == 200 and thread_response.get("success"):
        thread_id = thread_response["thread_id"]
    print(f"2. Creating Thread\nStatus: {thread_status}\nResponse: {json.dumps(thread_response, indent=2)}\n")
    
    if thread_id:
        # 3. Sending Message
        message_status, message_response = test_send_message(thread_id)
        print(f"3. Sending Message\nStatus: {message_status}\nResponse: {json.dumps(message_response, indent=2)}\n")
        
        # 4. Getting Thread Messages
        messages_status, messages_response = test_get_thread_messages(thread_id)
        print(f"4. Getting Thread Messages\nStatus: {messages_status}\nResponse: {json.dumps(messages_response, indent=2)}\n")
    else:
        print("Skipping message sending and retrieval as thread creation failed.")
