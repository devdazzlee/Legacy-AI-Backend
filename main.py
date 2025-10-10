from typing import Union, List, Dict, Any
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
import uuid

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Azure OpenAI Chat API",
    description="FastAPI application with Azure OpenAI integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class MessageRequest(BaseModel):
    thread_id: str
    message: str

class MessageResponse(BaseModel):
    success: bool
    message: str
    response: str
    thread_id: str

class ThreadResponse(BaseModel):
    success: bool
    thread_id: str
    message: str

class ThreadMessagesResponse(BaseModel):
    success: bool
    thread_id: str
    messages: List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status: str
    message: str
    service: str

class ResponseValidationRequest(BaseModel):
    response: str
    question_type: str = "day_description"

class ResponseValidationResponse(BaseModel):
    is_detailed: bool
    suggestion: str
    message: str

class ServiceValidationRequest(BaseModel):
    response: str
    selected_services: list[str]
    question_type: str = "client_response_outcomes"

class ServiceValidationResponse(BaseModel):
    is_relevant: bool
    suggestion: str
    message: str

class ChatbotRequest(BaseModel):
    message: str
    context: dict

class ChatbotResponse(BaseModel):
    response: str

class PrescreenerRequest(BaseModel):
    message: str
    clientId: str
    context: dict = {}

class PrescreenerResponse(BaseModel):
    isApproved: bool
    status: str  # "approved" or "rejected"
    userMessage: str  # Original message
    exampleMessage: str  # AI-generated example (only shown if rejected)
    feedback: str  # Clear acknowledgement
    detectedIssues: List[str]  # Safety incidents detected
    requiredActions: List[str]  # What user must do next
    canSubmit: bool  # Whether user can submit the current message

# Azure OpenAI Service
class AzureOpenAIService:
    """Service class for Azure OpenAI integration"""
    
    def __init__(self):
        self.client = None
        self._initialized = False
        self.conversations = {}  # Store conversations in memory
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")  # Store deployment name
    
    def _initialize_client(self):
        """Initialize the Azure OpenAI client"""
        try:
            # Your Azure OpenAI configuration - loaded from environment variables
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_KEY")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            
            if not endpoint or not api_key:
                raise ValueError("Azure OpenAI credentials not found in environment variables")
            
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )
            
            self._initialized = True
            logger.info("Azure OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise
    
    def send_message(self, thread_id: str, message: str) -> str:
        """Send a message to Azure OpenAI and get response"""
        try:
            if not self._initialized:
                self._initialize_client()
            
            # Initialize conversation history if not exists
            if thread_id not in self.conversations:
                self.conversations[thread_id] = [
                    {
                        "role": "system",
                        "content": "You are an AI assistant that helps people find information."
                    }
                ]
            
            # Add user message to conversation
            self.conversations[thread_id].append({
                "role": "user",
                "content": message
            })
            
            # Get response from Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=self.conversations[thread_id],
                max_tokens=6553,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Add assistant response to conversation
            self.conversations[thread_id].append({
                "role": "assistant",
                "content": response_content
            })
            
            return response_content
            
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
    
    def get_thread_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a thread"""
        if thread_id not in self.conversations:
            return []
        
        return self.conversations[thread_id]
    
    def validate_response_detail(self, response: str, question_type: str = "day_description") -> Dict[str, Any]:
        """Validate if a response is detailed enough and provide suggestions"""
        try:
            if not self._initialized:
                self._initialize_client()
            
            # Create a validation prompt based on question type
            if question_type == "day_description":
                validation_prompt = f"""
You are a healthcare assistant helping to ensure detailed progress notes. 

The user provided this response to "Overall, how was the client's day today?": "{response}"

Please analyze this response and determine:
1. Is it detailed enough (more than just one word like "good", "excellent", "fine", "okay", "bad", "terrible")?
2. If not detailed, provide a CONCRETE EXAMPLE of a detailed response.

IMPORTANT: You MUST respond with valid JSON only. All fields must be strings, never null or undefined.

Respond in this exact JSON format:
{{
    "is_detailed": true/false,
    "suggestion": "A concrete example of a detailed response - NOT instructions",
    "message": "Brief explanation of why more detail is needed"
}}

CRITICAL RULES FOR SUGGESTION:
- Provide a CONCRETE EXAMPLE, not instructions
- Write as if you are describing the client's actual day
- Include specific activities, mood, or interactions
- Do NOT use phrases like "provide details" or "include information"
- Write a complete sentence describing a realistic day

Examples of GOOD suggestions (concrete examples):
"The client had a calm day, participating in morning exercises and showing enthusiasm during art therapy. They expressed satisfaction with their meals and completed tasks independently."
"The client appeared energetic today, actively engaging in group activities and completing their personal care routine with minimal assistance."

Examples of BAD suggestions (instructions - DO NOT DO THIS):
"Provide more details about the client's day"
"Include information about activities and mood"
"Describe specific activities they participated in"

CRITICAL: Return only valid JSON. Do not include any text before or after the JSON object.
"""
            
            # Get validation from Azure OpenAI
            validation_response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful healthcare assistant that validates response quality and provides constructive feedback."
                    },
                    {
                        "role": "user",
                        "content": validation_prompt
                    }
                ],
                max_tokens=500,
                temperature=0.3,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            
            # Parse the response
            response_content = validation_response.choices[0].message.content.strip()
            logger.info(f"AI Response Content: {response_content}")
            
            # Try to parse JSON response
            try:
                import json
                import re
                
                # Clean up the response content - remove any text before/after JSON
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(0)
                
                validation_result = json.loads(response_content)
                
                # Ensure all fields are properly typed and not None
                suggestion = validation_result.get("suggestion")
                message = validation_result.get("message")
                
                # Handle None values and ensure strings
                if suggestion is None:
                    suggestion = "Please provide more details about the client's day, including specific activities, mood, interactions, or any notable events."
                elif not isinstance(suggestion, str):
                    suggestion = str(suggestion)
                
                if message is None:
                    message = "Your response should include specific details about the client's day rather than just a single word."
                elif not isinstance(message, str):
                    message = str(message)
                
                final_result = {
                    "is_detailed": validation_result.get("is_detailed", False),
                    "suggestion": suggestion,
                    "message": message
                }
                logger.info(f"Final validation result: {final_result}")
                return final_result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "is_detailed": len(response.split()) > 3,  # Simple word count check
                    "suggestion": "Please provide more details about the client's day, including specific activities, mood, interactions, or any notable events.",
                    "message": "Your response should include specific details about the client's day rather than just a single word."
                }
            
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            # Fallback validation
            return {
                "is_detailed": len(response.split()) > 3,
                "suggestion": "Please provide more details about the client's day, including specific activities, mood, interactions, or any notable events.",
                "message": "Unable to validate response automatically. Please provide more detailed information."
            }

    def validate_service_relevance(self, response: str, selected_services: list[str], question_type: str = "client_response_outcomes") -> Dict[str, Any]:
        """Validate if a response is relevant to the selected services"""
        try:
            if not self._initialized:
                self._initialize_client()

            # Create a validation prompt for service relevance based on question type
            services_text = ", ".join(selected_services)
            
            # Determine the question context and appropriate validation criteria
            if question_type == "goals_addressed_today":
                question_context = "Goals Addressed Today"
                validation_focus = "goals that were worked on related to the specific services provided"
                example_focus = "specific goals related to the services"
            elif question_type == "support_actions_taken":
                question_context = "Support Actions Taken"
                validation_focus = "specific actions taken related to the services provided"
                example_focus = "specific actions taken for the services"
            elif question_type == "client_response_outcomes":
                question_context = "Client Response/Outcomes"
                validation_focus = "how the client responded to or benefited from the services"
                example_focus = "client responses to the services"
            elif question_type == "issues_or_concerns":
                question_context = "Issues or Concerns"
                validation_focus = "challenges or concerns related to the services provided"
                example_focus = "issues related to the services"
            else:
                question_context = "the question"
                validation_focus = "content relevant to the services provided"
                example_focus = "content related to the services"
            
            validation_prompt = f"""
You are a healthcare assistant helping to ensure progress notes are relevant to the services provided.

The user provided this response to "{question_context}": "{response}"

The selected services provided to the client were: {services_text}

Please analyze this response and determine:
1. Is the response relevant to the specific services that were provided?
2. Does it describe {validation_focus}?
3. If not relevant, provide a CONCRETE EXAMPLE of a relevant response.

IMPORTANT: You MUST respond with valid JSON only. All fields must be strings, never null or undefined.

Respond in this exact JSON format:
{{
    "is_relevant": true/false,
    "suggestion": "A concrete example of a response relevant to the provided services - NOT instructions",
    "message": "Brief explanation of why the response should relate to the services"
}}

CRITICAL RULES FOR SUGGESTION:
- Provide a CONCRETE EXAMPLE, not instructions
- Write as if you are describing {example_focus}
- Include specific details related to the services provided
- Do NOT use phrases like "provide details" or "include information"
- Write a complete sentence describing realistic {example_focus}

Examples of GOOD suggestions (concrete examples):
"The client showed improved mobility after physical therapy, walking more confidently with the walker. They expressed satisfaction with the medication management support and completed their personal hygiene tasks independently."
"The client responded positively to the home cleaning assistance, organizing their belongings and expressing gratitude. They were cooperative during bathing assistance and showed increased comfort with the mobility support."

Examples of BAD suggestions (instructions - DO NOT DO THIS):
"Provide details about how the client responded to the services"
"Include information about client outcomes related to the services"
"Describe the client's response to the provided services"

CRITICAL: Return only valid JSON. Do not include any text before or after the JSON object.
"""

            # Get validation from Azure OpenAI
            validation_response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful healthcare assistant that validates response relevance to services and provides constructive feedback."
                    },
                    {
                        "role": "user",
                        "content": validation_prompt
                    }
                ],
                max_tokens=500,
                temperature=0.3,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )

            # Parse the response
            response_content = validation_response.choices[0].message.content.strip()
            logger.info(f"AI Service Validation Response Content: {response_content}")

            # Try to parse JSON response
            try:
                import json
                import re

                # Clean up the response content - remove any text before/after JSON
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(0)

                validation_result = json.loads(response_content)

                # Ensure all fields are properly typed and not None
                suggestion = validation_result.get("suggestion")
                message = validation_result.get("message")

                # Handle None values and ensure strings
                if suggestion is None:
                    suggestion = f"The client responded positively to the {services_text} services, showing improved outcomes and expressing satisfaction with the assistance provided."
                elif not isinstance(suggestion, str):
                    suggestion = str(suggestion)

                if message is None:
                    message = f"Your response should describe how the client responded to the specific services: {services_text}"
                elif not isinstance(message, str):
                    message = str(message)

                final_result = {
                    "is_relevant": validation_result.get("is_relevant", False),
                    "suggestion": suggestion,
                    "message": message
                }
                logger.info(f"Final service validation result: {final_result}")
                return final_result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "is_relevant": len(response.split()) > 5,  # Simple word count check
                    "suggestion": f"The client responded positively to the {services_text} services, showing improved outcomes and expressing satisfaction with the assistance provided.",
                    "message": f"Unable to validate response automatically. Please describe how the client responded to the services: {services_text}"
                }

        except Exception as e:
            logger.error(f"Error validating service relevance: {str(e)}")
            # Fallback validation
            return {
                "is_relevant": len(response.split()) > 5,
                "suggestion": f"The client responded positively to the {services_text} services, showing improved outcomes and expressing satisfaction with the assistance provided.",
                "message": f"Unable to validate response automatically. Please describe how the client responded to the services: {services_text}"
            }

    def ask_chatbot(self, message: str, context: dict) -> str:
        """Handle chatbot questions with context-aware responses"""
        try:
            if not self._initialized:
                self._initialize_client()

            # Extract context information
            current_step = context.get("currentStep", 0)
            current_note_step = context.get("currentNoteStep", 0)
            selected_services = context.get("selectedServices", [])
            current_section = context.get("currentSection", "Progress Log")
            
            # Create context-aware prompt
            services_text = ", ".join(selected_services) if selected_services else "general services"
            
            chatbot_prompt = f"""
You are a helpful AI assistant for healthcare workers filling out progress logs. You provide step-by-step guidance and answer questions about documentation.

CONTEXT:
- Current Section: {current_section}
- Current Step: {current_step}
- Current Note Step: {current_note_step}
- Selected Services: {services_text}

USER QUESTION: "{message}"

Please provide a helpful, detailed response that:
1. Directly answers the user's question
2. Provides step-by-step guidance when appropriate
3. Includes specific examples related to the selected services
4. Is relevant to the current section they're working on
5. Uses professional healthcare terminology
6. Keeps responses concise but comprehensive

IMPORTANT: 
- Be specific and actionable
- Include examples related to their selected services: {services_text}
- Provide clear step-by-step instructions when needed
- Use a helpful, professional tone
- Focus on practical guidance for healthcare documentation

Respond with helpful guidance only, no additional formatting or explanations.
"""

            # Get response from Azure OpenAI
            chatbot_response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful healthcare documentation assistant that provides clear, actionable guidance for progress log completion."
                    },
                    {
                        "role": "user",
                        "content": chatbot_prompt
                    }
                ],
                max_tokens=800,
                temperature=0.3,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )

            response_content = chatbot_response.choices[0].message.content.strip()
            logger.info(f"Chatbot Response: {response_content}")
            
            return response_content

        except Exception as e:
            logger.error(f"Error in chatbot: {str(e)}")
            # Return fallback response
            fallback_section = context.get("currentSection", "Progress Log")
            fallback_services = ", ".join(context.get("selectedServices", [])) if context.get("selectedServices") else "general services"
            return f"I'm here to help with {fallback_section}. I can provide guidance on:\n\n• How to fill out this section properly\n• Examples of good responses\n• Understanding requirements\n• Best practices for documentation\n\nYour selected services are: {fallback_services}\n\nPlease ask me a specific question, and I'll provide detailed guidance!"
    
    def prescreener_strict_validation(self, message: str, client_id: str, context: dict) -> Dict[str, Any]:
        """
        Strict validation for AI Prescreener - follows rigid approval rules
        
        Rules:
        1. Only approve messages that are already high-quality and detailed
        2. Detect safety incidents (falls, fever, medication issues, etc.)
        3. Provide example messages for rejected content (users must retype)
        4. Never auto-replace user's message
        """
        try:
            if not self._initialized:
                self._initialize_client()
            
            # Analyze message quality and safety
            message_lower = message.lower().strip()
            word_count = len(message.split())
            
            # Detect safety incidents
            detected_issues = []
            critical_keywords = {
                "fall": ["fell", "fall", "falling", "fallen", "slipped", "tripped"],
                "fever": ["fever", "temperature", "hot", "burning up", "feverish"],
                "injury": ["injured", "hurt", "bleeding", "bruise", "cut", "wound"],
                "medication_error": ["wrong medication", "wrong dose", "missed medication"],
                "emergency": ["emergency", "911", "ambulance", "urgent care", "hospital"]
            }
            
            for issue_type, keywords in critical_keywords.items():
                for keyword in keywords:
                    if keyword in message_lower:
                        detected_issues.append(f"🚨 {issue_type.replace('_', ' ').title()} detected: '{keyword}'")
                        break
            
            # STRICT Quality checks - must meet AI example standards
            # AI examples are 50-80 words, so we require similar length and detail
            is_too_short = word_count < 40  # Increased from 15 to 40
            is_vague = any(word in message_lower for word in ["good", "fine", "okay", "nice", "great", "well"]) and word_count < 50
            lacks_detail = word_count < 50 and len(detected_issues) == 0  # Increased from 25 to 50
            lacks_specifics = not any(word in message_lower for word in ["client", "participated", "completed", "assisted", "engaged", "showed", "demonstrated", "performed"])
            
            # Determine if message should be approved
            is_approved = False
            status = "rejected"
            feedback = ""
            example_message = ""
            required_actions = []
            
            # Critical safety events are always rejected for review
            if detected_issues:
                status = "rejected"
                is_approved = False
                feedback = f"❌ MESSAGE REJECTED: Critical safety event detected. This requires detailed documentation and supervisor review."
                required_actions = [
                    "Document exact time and location",
                    "Describe what happened in detail",
                    "List all actions taken",
                    "Contact supervisor immediately"
                ]
                
                # Generate AI example for safety incident
                example_prompt = f"""
You are a healthcare documentation expert. The user reported a safety incident but needs a proper example.

USER'S MESSAGE: "{message}"
DETECTED ISSUES: {", ".join(detected_issues)}

Generate a COMPLETE, DETAILED example of how this incident should be documented. Include:
- Exact time and location
- What happened (specific details)
- Client's condition
- Actions taken immediately
- Whether emergency services were contacted
- Current status

Write as if you are the healthcare worker documenting the actual incident. Be specific and professional.

Return ONLY the example message, no other text.
"""
                
                example_response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": "You are a healthcare documentation expert. Provide detailed example messages."},
                        {"role": "user", "content": example_prompt}
                    ],
                    max_tokens=400,
                    temperature=0.7
                )
                example_message = example_response.choices[0].message.content.strip()
                
            elif is_too_short or is_vague or lacks_detail or lacks_specifics:
                status = "rejected"
                is_approved = False
                feedback = f"❌ MESSAGE REJECTED: Message is too brief or lacks sufficient detail (only {word_count} words).\n\nRequired: 50+ words with specific details about client interactions. Review the EXAMPLE below - your message must be SIMILAR in length and detail."
                required_actions = [
                    "Add specific details about activities performed",
                    "Describe client's mood, behavior, and engagement",
                    "Include observable facts and client responses",
                    "Write AT LEAST 50-60 words with healthcare terminology",
                    "Match the detail level shown in the example"
                ]
                
                # Generate AI example for brief message
                selected_services = context.get("selectedServices", ["general care"])
                services_text = ", ".join(selected_services[:3])
                
                example_prompt = f"""
Generate a detailed, professional progress note EXAMPLE for a healthcare shift.

SERVICES PROVIDED: {services_text}
CONTEXT: {context.get("currentSection", "Overall day observations")}

Create a realistic EXAMPLE that the user CANNOT submit directly - they must write their own version:
- 50-70 words minimum
- Describes the client's mood, engagement, and specific behaviors
- Mentions specific activities completed with detail
- Notes any challenges or successes observed
- Uses professional healthcare language
- Includes specific, observable facts (not vague terms like "good" or "fine")

Return ONLY the example message (no labels, no extra text).
"""
                
                example_response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": "You are a healthcare documentation expert. Provide detailed example messages."},
                        {"role": "user", "content": example_prompt}
                    ],
                    max_tokens=250,
                    temperature=0.7
                )
                example_message = example_response.choices[0].message.content.strip()
                
            else:
                # Message is detailed enough - APPROVE
                # Only approved if: 50+ words, specific details, professional language
                status = "approved"
                is_approved = True
                feedback = f"✅ MESSAGE APPROVED: Your message meets quality standards ({word_count} words with sufficient detail). You may submit YOUR OWN message."
                required_actions = ["Review your message one final time", "Click 'Submit Approved' to submit YOUR message"]
                example_message = ""  # No example needed for approved messages
            
            return {
                "isApproved": is_approved,
                "status": status,
                "userMessage": message,
                "exampleMessage": example_message,
                "feedback": feedback,
                "detectedIssues": detected_issues,
                "requiredActions": required_actions,
                "canSubmit": is_approved
            }
            
        except Exception as e:
            logger.error(f"Error in prescreener validation: {str(e)}")
            # Fail safe - reject on error
            return {
                "isApproved": False,
                "status": "rejected",
                "userMessage": message,
                "exampleMessage": "Client had a positive day, actively participating in all care activities. They showed good engagement during personal care routines and maintained positive mood throughout the shift. No concerns noted.",
                "feedback": "⚠️ System error during validation. Please review your message and try again.",
                "detectedIssues": ["System validation error"],
                "requiredActions": ["Review your message", "Try analyzing again"],
                "canSubmit": False
            }

# Initialize service
azure_service = AzureOpenAIService()

# Initialize AI Prescreener system components
ai_prescreener = None
alert_system = None
mobile_service = None

def initialize_ai_prescreener():
    """Initialize AI Prescreener components"""
    global ai_prescreener, alert_system, mobile_service
    
    try:
        from ai_prescreener import AIPrescreenerCore
        from database_schema import db_manager
        from alert_system import RealTimeAlertSystem
        from mobile_integration import MobileAIPrescreenerService
        
        ai_prescreener = AIPrescreenerCore(azure_service)
        alert_system = RealTimeAlertSystem(db_manager)
        mobile_service = MobileAIPrescreenerService()
        
        print("✅ AI Prescreener system initialized successfully")
    except Exception as e:
        print(f"⚠️ AI Prescreener initialization failed: {str(e)}")
        print("⚠️ Some features may not be available")

# Initialize on startup
initialize_ai_prescreener()

# Existing endpoints
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# Azure OpenAI endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Azure OpenAI Chat API is running",
        service="Azure OpenAI Integration"
    )

@app.post("/threads/create", response_model=ThreadResponse)
async def create_thread():
    """Create a new conversation thread"""
    try:
        thread_id = str(uuid.uuid4())
        azure_service.conversations[thread_id] = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information."
            }
        ]
        
        return ThreadResponse(
            success=True,
            thread_id=thread_id,
            message="Thread created successfully"
        )
    except Exception as e:
        logger.error(f"Error creating thread: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating thread: {str(e)}")

@app.post("/messages/send", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """Send a message to Azure OpenAI"""
    try:
        response = azure_service.send_message(request.thread_id, request.message)
        
        return MessageResponse(
            success=True,
            message="Message sent successfully",
            response=response,
            thread_id=request.thread_id
        )
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending message: {str(e)}")

@app.get("/threads/{thread_id}/messages", response_model=ThreadMessagesResponse)
async def get_thread_messages(thread_id: str):
    """Get all messages for a thread"""
    try:
        messages = azure_service.get_thread_messages(thread_id)
        
        return ThreadMessagesResponse(
            success=True,
            thread_id=thread_id,
            messages=messages
        )
    except Exception as e:
        logger.error(f"Error getting thread messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting thread messages: {str(e)}")

@app.post("/validate-response", response_model=ResponseValidationResponse)
async def validate_response(request: ResponseValidationRequest):
    """Validate if a response is detailed enough"""
    try:
        validation_result = azure_service.validate_response_detail(
            request.response, 
            request.question_type
        )
        
        return ResponseValidationResponse(
            is_detailed=validation_result["is_detailed"],
            suggestion=validation_result["suggestion"],
            message=validation_result["message"]
        )
    except Exception as e:
        logger.error(f"Error validating response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error validating response: {str(e)}")

@app.post("/validate-service-relevance", response_model=ServiceValidationResponse)
async def validate_service_relevance(request: ServiceValidationRequest):
    """Validate if a response is relevant to the selected services"""
    try:
        validation_result = azure_service.validate_service_relevance(
            request.response, 
            request.selected_services,
            request.question_type
        )
        
        return ServiceValidationResponse(
            is_relevant=validation_result["is_relevant"],
            suggestion=validation_result["suggestion"],
            message=validation_result["message"]
        )
    except Exception as e:
        logger.error(f"Error validating service relevance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error validating service relevance: {str(e)}")

@app.post("/chatbot/ask", response_model=ChatbotResponse)
async def ask_chatbot(request: ChatbotRequest):
    """Handle chatbot questions with context-aware responses"""
    try:
        response = azure_service.ask_chatbot(request.message, request.context)
        return ChatbotResponse(response=response)
    except Exception as e:
        logger.error(f"Error in chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in chatbot: {str(e)}")

@app.post("/prescreener/validate-strict", response_model=PrescreenerResponse)
async def prescreener_validate_strict(request: PrescreenerRequest):
    """
    Strict AI Prescreener validation with voice support
    
    Rules:
    - Only approves high-quality, detailed messages
    - Detects safety incidents (falls, fever, injuries, etc.)
    - Provides example messages for rejected content (users must retype)
    - Never auto-replaces user's message
    - Supports voice input/output
    """
    try:
        logger.info(f"🔍 PRESCREENER - Validating message for client: {request.clientId}")
        logger.info(f"📝 Message: {request.message}")
        
        validation_result = azure_service.prescreener_strict_validation(
            request.message,
            request.clientId,
            request.context
        )
        
        logger.info(f"✅ PRESCREENER - Status: {validation_result['status']}")
        logger.info(f"📊 Can Submit: {validation_result['canSubmit']}")
        
        return PrescreenerResponse(
            isApproved=validation_result["isApproved"],
            status=validation_result["status"],
            userMessage=validation_result["userMessage"],
            exampleMessage=validation_result["exampleMessage"],
            feedback=validation_result["feedback"],
            detectedIssues=validation_result["detectedIssues"],
            requiredActions=validation_result["requiredActions"],
            canSubmit=validation_result["canSubmit"]
        )
    except Exception as e:
        logger.error(f"Error in prescreener validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in prescreener validation: {str(e)}")

@app.post("/speech-to-text")
async def speech_to_text(request: dict):
    """
    Convert speech audio to text using OpenAI Whisper API
    
    Accepts base64 encoded audio and returns transcribed text
    """
    import base64
    import tempfile
    import os
    
    try:
        audio_data = request.get("audio_data")
        audio_format = request.get("format", "m4a")
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        logger.info(f"🎤 SPEECH-TO-TEXT - Received audio ({len(audio_data)} bytes base64)")
        logger.info(f"🎤 Audio format: {audio_format}")
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)
        logger.info(f"🎤 Decoded audio: {len(audio_bytes)} bytes")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
            logger.info(f"🎤 Saved to temp file: {temp_audio_path}")
        
        try:
            # Use Azure OpenAI for Whisper (same credentials as GPT-4)
            # Azure OpenAI supports Whisper via audio.transcriptions API
            
            logger.info("🔄 Using Azure Cognitive Services for Whisper transcription...")
            
            # Get Whisper configuration from environment
            whisper_deployment = os.getenv("AZURE_WHISPER_DEPLOYMENT", "whisper")
            whisper_endpoint = os.getenv("AZURE_WHISPER_ENDPOINT", "https://hakeem-4411-resource.cognitiveservices.azure.com/openai/deployments/whisper/audio/transcriptions")
            whisper_key = os.getenv("AZURE_WHISPER_KEY", os.getenv("AZURE_OPENAI_KEY"))
            whisper_api_version = os.getenv("AZURE_WHISPER_API_VERSION", "2024-06-01")
            
            logger.info(f"🎤 Whisper endpoint: {whisper_endpoint}")
            logger.info(f"🎤 API version: {whisper_api_version}")
            
            # Create Azure client for Whisper (Cognitive Services endpoint)
            whisper_client = AzureOpenAI(
                api_key=whisper_key,
                api_version=whisper_api_version,
                azure_endpoint="https://hakeem-4411-resource.cognitiveservices.azure.com/"
            )
            
            # Use Azure Cognitive Services Whisper
            with open(temp_audio_path, "rb") as audio_file:
                transcription = whisper_client.audio.transcriptions.create(
                    model=whisper_deployment,
                    file=audio_file,
                    language="en"
                )
            
            logger.info("✅ Used Azure OpenAI Whisper successfully")
            
            transcribed_text = transcription.text
            logger.info(f"✅ SPEECH-TO-TEXT - Transcription: {transcribed_text}")
            
            # Clean up temp file
            os.unlink(temp_audio_path)
            
            return {
                "success": True,
                "text": transcribed_text,
                "language": "en"
            }
            
        except Exception as whisper_error:
            logger.error(f"❌ Whisper API error: {str(whisper_error)}")
            logger.error(f"❌ Error type: {type(whisper_error)}")
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
            # Fallback: Return a message indicating transcription is unavailable
            return {
                "success": False,
                "text": "",
                "error": "Speech-to-text service temporarily unavailable. Please type your message."
            }
            
    except Exception as e:
        logger.error(f"❌ SPEECH-TO-TEXT ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in speech-to-text: {str(e)}")

@app.post("/ai-prescreener/analyze-shift")
async def analyze_shift_with_ai_prescreener(request: dict):
    """Analyze shift data using AI Prescreener system"""
    try:
        if ai_prescreener is None:
            raise HTTPException(status_code=503, detail="AI Prescreener system not initialized")
        
        # Convert request to ShiftData format
        from ai_prescreener import ShiftData
        from datetime import datetime
        
        shift_data = ShiftData(
            shift_id=request.get("shift_id", f"mobile_shift_{datetime.now().timestamp()}"),
            client_id=request.get("client_id"),
            worker_id=request.get("worker_id"),
            shift_date=datetime.fromisoformat(request.get("shift_date", datetime.now().isoformat())),
            shift_duration_hours=request.get("shift_duration_hours", 8.0),
            is_overnight_shift=request.get("is_overnight_shift", False),
            worker_notes=request.get("worker_notes", ""),
            completed_tasks=request.get("completed_tasks", []),
            services_provided=request.get("services_provided", []),
            additional_context=request.get("additional_context", {})
        )
        
        # Perform AI Prescreener analysis
        result = ai_prescreener.analyze_shift(shift_data)
        
        # Process flagged events for alerts if alert system is available
        if alert_system is not None:
            for event in result.flagged_events:
                await alert_system.process_flagged_event(event)
        
        return {
            "success": True,
            "analysis_id": result.analysis_id,
            "flagged_events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "severity": event.severity.value,
                    "description": event.description,
                    "requires_escalation": event.requires_escalation
                }
                for event in result.flagged_events
            ],
            "compliance_violations": result.compliance_violations,
            "generated_narrative": result.generated_narrative,
            "requires_human_review": result.requires_human_review,
            "confidence_score": result.confidence_score
        }
        
    except Exception as e:
        logger.error(f"Error in AI Prescreener analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing shift: {str(e)}")

@app.get("/ai-prescreener/client-safety-summary/{client_id}")
async def get_client_safety_summary(client_id: str, days: int = 7):
    """Get safety summary for a client"""
    try:
        if mobile_service is None:
            raise HTTPException(status_code=503, detail="AI Prescreener system not initialized")
        
        summary = await mobile_service.get_client_safety_summary(client_id, days)
        return summary
    except Exception as e:
        logger.error(f"Error getting safety summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting safety summary: {str(e)}")

@app.post("/ai-prescreener/resolve-event/{event_id}")
async def resolve_flagged_event(event_id: str, resolved_by: str):
    """Resolve a flagged event"""
    try:
        if mobile_service is None:
            raise HTTPException(status_code=503, detail="AI Prescreener system not initialized")
        
        result = await mobile_service.integration.resolve_flagged_event(event_id, resolved_by)
        return result
    except Exception as e:
        logger.error(f"Error resolving event: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resolving event: {str(e)}")

@app.get("/ai-prescreener/pending-alerts")
async def get_pending_alerts(client_id: str = None):
    """Get pending alerts"""
    try:
        if mobile_service is None:
            raise HTTPException(status_code=503, detail="AI Prescreener system not initialized")
        
        alerts = await mobile_service.integration.get_pending_alerts(client_id)
        return {"alerts": alerts}
    except Exception as e:
        logger.error(f"Error getting pending alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting pending alerts: {str(e)}")

@app.get("/ai-prescreener/system-status")
async def get_ai_prescreener_status():
    """Get AI Prescreener system status"""
    try:
        if mobile_service is None:
            return {
                "system_status": "unhealthy",
                "ai_prescreener_status": "inactive",
                "alert_system_status": "inactive",
                "database_status": "unhealthy",
                "error": "AI Prescreener system not initialized"
            }
        
        # Simple status check without complex integration
        return {
            "system_status": "healthy",
            "ai_prescreener_status": "active" if ai_prescreener else "inactive",
            "alert_system_status": "active" if alert_system else "inactive",
            "database_status": "healthy",
            "total_clients": 1,
            "total_flagged_events": 0,
            "pending_alerts": 0,
            "last_analysis": None
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return {
            "system_status": "unhealthy",
            "ai_prescreener_status": "inactive",
            "alert_system_status": "inactive",
            "database_status": "unhealthy",
            "error": str(e)
        }

# AI Prescreener endpoints
class ConflictDetectionRequest(BaseModel):
    concepts: List[str]
    clientId: str
    context: dict

class ConflictDetectionResponse(BaseModel):
    hasConflicts: bool
    conflicts: List[dict] = []
    resolvedConcepts: List[str] = []

class NarrativeRequest(BaseModel):
    message: str
    context: dict

class NarrativeResponse(BaseModel):
    narrative: str

@app.post("/prescreen-message", response_model=PrescreenerResponse)
async def prescreen_message(request: PrescreenerRequest):
    """AI Prescreener - Check message against client-specific guardrails"""
    try:
        # Get client-specific guardrails (in real implementation, this would come from database)
        client_guardrails = get_client_guardrails(request.clientId)
        
        # Check for restricted topics
        restrictions_applied = []
        message_lower = request.message.lower()
        
        for topic in client_guardrails.get("restrictedTopics", []):
            if topic.lower() in message_lower:
                restrictions_applied.append(f"Topic '{topic}' is not allowed for this client")
        
        # Extract concepts from message
        concepts = extract_concepts_from_message(request.message)
        
        # Detect conflicts
        conflict_check = await detect_concept_conflicts(concepts, request.clientId, request.context)
        
        # Generate narrative
        narrative = await generate_shift_narrative(request.message, request.context, client_guardrails)
        
        # Check for challenges
        challenges_noted = detect_challenges_in_message(request.message)
        challenges_description = extract_challenges_description(request.message) if challenges_noted else "No challenges noted"
        
        # Generate suggestions
        context_with_message = {**request.context, "message": request.message}
        suggestions = generate_prescreener_suggestions(restrictions_applied, context_with_message, client_guardrails, conflict_check)
        
        return PrescreenerResponse(
            isAllowed=len(restrictions_applied) == 0 and not conflict_check["hasConflicts"],
            filteredMessage=filter_restricted_content(request.message, client_guardrails) if restrictions_applied else request.message,
            restrictionsApplied=restrictions_applied,
            suggestions=suggestions,
            narrative=narrative,
            challengesNoted=challenges_noted,
            challengesDescription=challenges_description
        )
        
    except Exception as e:
        logger.error(f"Error in prescreener: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in prescreener: {str(e)}")

@app.post("/detect-conflicts", response_model=ConflictDetectionResponse)
async def detect_conflicts(request: ConflictDetectionRequest):
    """Detect conflicts between concepts"""
    try:
        if not azure_service._initialized:
            azure_service._initialize_client()
        
        # Create conflict detection prompt
        concepts_text = ", ".join(request.concepts)
        conflict_prompt = f"""
You are a healthcare AI assistant that detects conflicts between care concepts.

Analyze these concepts for potential conflicts: {concepts_text}

Client ID: {request.clientId}
Context: {request.context}

Please identify any conflicts between these concepts and provide resolutions.

Respond in this exact JSON format:
{{
    "hasConflicts": true/false,
    "conflicts": [
        {{
            "concept1": "first concept",
            "concept2": "second concept", 
            "conflictType": "type of conflict",
            "resolution": "how to resolve the conflict"
        }}
    ],
    "resolvedConcepts": ["list of concepts after resolution"]
}}

CRITICAL: Return only valid JSON. Do not include any text before or after the JSON object.
"""

        response = azure_service.client.chat.completions.create(
            model=azure_service.deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a healthcare assistant that identifies and resolves conflicts in care concepts."
                },
                {
                    "role": "user",
                    "content": conflict_prompt
                }
            ],
            max_tokens=500,
            temperature=0.3,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        
        response_content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        import re
        
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            response_content = json_match.group(0)
        
        conflict_result = json.loads(response_content)
        
        return ConflictDetectionResponse(
            hasConflicts=conflict_result.get("hasConflicts", False),
            conflicts=conflict_result.get("conflicts", []),
            resolvedConcepts=conflict_result.get("resolvedConcepts", request.concepts)
        )
        
    except Exception as e:
        logger.error(f"Error detecting conflicts: {str(e)}")
        # Fallback response
        return ConflictDetectionResponse(
            hasConflicts=False,
            conflicts=[],
            resolvedConcepts=request.concepts
        )

@app.post("/generate-narrative", response_model=NarrativeResponse)
async def generate_narrative(request: NarrativeRequest):
    """Generate dynamic narrative based on shift information"""
    try:
        if not azure_service._initialized:
            azure_service._initialize_client()
        
        # Extract narrative requirements from context
        narrative_reqs = request.context.get("narrativeRequirements", {})
        shift_duration = request.context.get("shiftDuration", 8)
        is_overnight = request.context.get("isOvernightShift", False)
        selected_services = request.context.get("selectedServices", [])
        
        # Create narrative generation prompt
        narrative_prompt = f"""
You are a healthcare AI assistant that generates detailed shift narratives.

Original Message: "{request.message}"

Shift Information:
- Duration: {shift_duration} hours
- Is Overnight Shift: {is_overnight}
- Selected Services: {', '.join(selected_services) if selected_services else 'General care'}

Narrative Requirements:
- Minimum Length: {narrative_reqs.get('minLength', 50)} characters
- Maximum Length: {narrative_reqs.get('maxLength', 500)} characters
- Overnight Format: {narrative_reqs.get('overnightShiftFormat', False)}

Generate a comprehensive narrative that:
1. Describes what the day/shift looked like
2. Is proportional to shift length (longer shift = longer narrative)
3. Uses shorter format for overnight shifts
4. Always explicitly states "no challenges noted" when no issues occur
5. Includes specific details about activities and client responses

IMPORTANT: 
- Make the narrative engaging and detailed
- Include specific activities and outcomes
- Always mention challenges explicitly (even if none occurred)
- Keep within the length requirements

Respond with just the narrative text, no additional formatting.
"""

        response = azure_service.client.chat.completions.create(
            model=azure_service.deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a healthcare documentation assistant that creates detailed, engaging shift narratives."
                },
                {
                    "role": "user",
                    "content": narrative_prompt
                }
            ],
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        
        narrative = response.choices[0].message.content.strip()
        
        return NarrativeResponse(narrative=narrative)
        
    except Exception as e:
        logger.error(f"Error generating narrative: {str(e)}")
        # Fallback narrative
        shift_duration = request.context.get("shiftDuration", 8)
        is_overnight = request.context.get("isOvernightShift", False)
        
        if is_overnight:
            fallback_narrative = f"Overnight shift completed successfully. Client was monitored throughout the night with regular check-ins. No challenges noted."
        else:
            fallback_narrative = f"Shift completed successfully. Duration: {shift_duration} hours. {request.message[:200]}... No challenges noted."
        
        return NarrativeResponse(narrative=fallback_narrative)

# Helper functions for AI Prescreener
def get_client_guardrails(client_id: str) -> dict:
    """Get client-specific guardrails (in production, this would come from database)"""
    # Default guardrails template
    default_guardrails = {
        "restrictedTopics": ["medication", "meal prep", "medical advice", "diagnosis"],
        "allowedServices": ["personal care", "companionship", "light housekeeping"],
        "specialInstructions": [
            "Always explicitly state 'no challenges noted' when no issues occur",
            "Generate narratives proportional to shift length",
            "Avoid discussing restricted topics"
        ],
        "narrativeRequirements": {
            "minLength": 50,
            "maxLength": 500,
            "overnightShiftFormat": False
        }
    }
    
    # In production, you would fetch this from a database
    # For now, return default guardrails
    return default_guardrails

async def detect_concept_conflicts(concepts: List[str], client_id: str, context: dict) -> dict:
    """Detect conflicts between concepts"""
    # Simple conflict detection logic
    conflicts = []
    
    # Check for common conflicts
    conflict_pairs = [
        ("medication", "meal prep"),
        ("bathing", "mobility"),
        ("exercise", "rest")
    ]
    
    for concept1, concept2 in conflict_pairs:
        if concept1 in concepts and concept2 in concepts:
            conflicts.append({
                "concept1": concept1,
                "concept2": concept2,
                "conflictType": "scheduling conflict",
                "resolution": f"Schedule {concept1} and {concept2} at different times"
            })
    
    return {
        "hasConflicts": len(conflicts) > 0,
        "conflicts": conflicts,
        "resolvedConcepts": concepts
    }

async def generate_shift_narrative(message: str, context: dict, guardrails: dict) -> str:
    """Generate shift narrative"""
    # This would typically call the AI service
    # For now, return a simple narrative
    shift_duration = context.get("shiftDuration", 8)
    is_overnight = context.get("isOvernightShift", False)
    
    if is_overnight:
        return f"Overnight shift completed. Client was monitored throughout the night with regular check-ins. {message[:100]}... No challenges noted."
    else:
        return f"Shift completed successfully. Duration: {shift_duration} hours. Activities included: {message[:200]}... No challenges noted."

def detect_challenges_in_message(message: str) -> bool:
    """Detect if message contains challenges"""
    challenge_keywords = [
        "difficulty", "problem", "issue", "concern", "challenge", "struggle",
        "resistance", "refusal", "incident", "emergency", "fall", "injury"
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in challenge_keywords)

def extract_challenges_description(message: str) -> str:
    """Extract challenges description from message"""
    challenge_keywords = [
        "difficulty", "problem", "issue", "concern", "challenge", "struggle",
        "resistance", "refusal", "incident", "emergency", "fall", "injury"
    ]
    
    message_lower = message.lower()
    found_challenges = [keyword for keyword in challenge_keywords if keyword in message_lower]
    
    if found_challenges:
        return f"Challenges noted: {', '.join(found_challenges)}"
    
    return "No challenges noted"

def generate_prescreener_suggestions(restrictions_applied: List[str], context: dict, guardrails: dict, conflict_check: dict) -> List[str]:
    """Generate suggestions for the prescreener"""
    suggestions = []
    
    if restrictions_applied:
        suggestions.append("Remove references to restricted topics")
        suggestions.append(f"Focus on allowed services: {', '.join(guardrails.get('allowedServices', []))}")
    
    if conflict_check.get("hasConflicts", False):
        suggestions.append("Resolve conflicting concepts before proceeding")
        for conflict in conflict_check.get("conflicts", []):
            suggestions.append(f"Resolve conflict between {conflict['concept1']} and {conflict['concept2']}: {conflict['resolution']}")
    
    # Check if challenges are explicitly mentioned
    message = context.get("message", "")
    challenges_noted = detect_challenges_in_message(message)
    if not challenges_noted:
        suggestions.append("Explicitly state 'no challenges noted' if no issues occurred")
    
    if context.get("shiftDuration", 0) > 8:
        suggestions.append("Provide more detailed narrative for longer shifts")
    
    return suggestions

def filter_restricted_content(message: str, guardrails: dict) -> str:
    """Filter restricted content from message"""
    filtered_message = message
    
    for topic in guardrails.get("restrictedTopics", []):
        import re
        regex = re.compile(re.escape(topic), re.IGNORECASE)
        filtered_message = regex.sub("[TOPIC RESTRICTED]", filtered_message)
    
    return filtered_message

def extract_concepts_from_message(message: str) -> List[str]:
    """Extract concepts from message"""
    concepts = []
    message_lower = message.lower()
    
    concept_keywords = [
        "medication", "meal", "prep", "cooking", "bathing", "mobility",
        "exercise", "therapy", "appointment", "transportation", "cleaning",
        "personal care", "companionship", "housekeeping", "assistance"
    ]
    
    for keyword in concept_keywords:
        if keyword in message_lower:
            concepts.append(keyword)
    
    return concepts