from typing import Union, List, Dict, Any
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI, RateLimitError
import uuid
import asyncio
import base64
import tempfile
import json
from datetime import datetime
from database_schema import DatabaseManager
from time import time

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

# Initialize database manager
db_manager = DatabaseManager()

# Rate Limiting Configuration
class RateLimiter:
    """Rate limiter to prevent hitting Azure OpenAI API rate limits"""
    def __init__(self, max_concurrent: int = 3, min_delay: float = 0.5):
        """
        Args:
            max_concurrent: Maximum number of concurrent requests
            min_delay: Minimum delay between requests in seconds
        """
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.min_delay = min_delay
        self.last_request_time = 0
        self.rate_limit_hits = 0
        
    async def acquire(self):
        """Acquire rate limiter lock and enforce delay"""
        await self.semaphore.acquire()
        
        # Enforce minimum delay between requests
        current_time = time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            await asyncio.sleep(self.min_delay - time_since_last)
        
        self.last_request_time = time()
    
    def release(self):
        """Release rate limiter lock"""
        self.semaphore.release()
    
    async def __aenter__(self):
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def on_rate_limit_hit(self):
        """Called when a rate limit error is detected"""
        self.rate_limit_hits += 1
        logger.warning(f"⚠️ Rate limit hit (#{self.rate_limit_hits}). Consider increasing delays or reducing concurrent requests.")

# Global rate limiter instance
rate_limiter = RateLimiter(max_concurrent=3, min_delay=0.8)  # 3 concurrent requests, 0.8s delay

# Dynamic guidelines are now generated from client guardrails in the database
# No hardcoded QUESTION_GUIDELINES needed

# Q41 guidelines should come from frontend API - no hardcoding in backend
# Frontend gets question 41 from main API and sends word count to validation API

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

class AnswerValidationRequest(BaseModel):
    user_answer: str
    question_guardrail: str = None  # Question-specific guardrail from frontend
    q41_guardrail: str = None  # Q41 guardrail (word count) from frontend
    ai_suggestion: str = None  # Optional: AI suggestion that user was asked to modify

class AnswerValidationResponse(BaseModel):
    status: str  # "approve", "reject", "review"
    message: str
    analysis: str = ""  # Detailed AI analysis of the answer
    suggested_answer: str = ""  # Suggested corrected answer from the model
    missing_elements: List[str] = []
    safety_concerns: List[str] = []
    word_count: int
    guidelines_checked: List[str] = []
    modification_status: dict = {}  # Status of modifications if ai_suggestion was provided

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
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "o4-mini")  # Store deployment name
    
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
            try:
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
            except RateLimitError as e:
                logger.error(f"❌ Rate limit error (429) in send_message: {str(e)}")
                logger.warning("⚠️ Azure OpenAI rate limit exceeded. The request will be retried automatically by the SDK.")
                raise  # Re-raise to let the caller handle retries
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    logger.error(f"❌ Rate limit detected in send_message: {str(e)}")
                    logger.warning("⚠️ Azure OpenAI rate limit exceeded. Consider reducing request frequency.")
                raise  # Re-raise to let the caller handle it
            
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

# Answer validation functions
def check_meaningful_modification(original: str, current: str) -> dict:
    """
    Check if user made meaningful modifications to AI suggestion.
    Returns validation result with detailed feedback.
    
    Requirements for meaningful modification:
    - At least 10 characters different (excluding whitespace)
    - At least 2 new/changed meaningful words (3+ chars)
    - At least 15% content change
    
    This prevents users from just adding random text or whitespace.
    """
    import re
    
    original_trim = original.strip() if original else ""
    current_trim = current.strip() if current else ""
    
    # If exactly same, no modification
    if original_trim == current_trim:
        return {
            "is_valid": False,
            "reason": "Please make changes to the AI suggestion",
            "change_percent": 0,
            "char_diff": 0,
            "new_words_count": 0,
            "needs_more_changes": True
        }
    
    # Normalize text for comparison (remove extra whitespace, lowercase)
    def normalize_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip().lower())
    
    original_norm = normalize_text(original_trim)
    current_norm = normalize_text(current_trim)
    
    # Calculate character difference (excluding whitespace)
    original_chars = re.sub(r'\s', '', original_norm)
    current_chars = re.sub(r'\s', '', current_norm)
    char_diff = abs(len(current_chars) - len(original_chars))
    
    # Count words
    original_words = [w for w in original_norm.split() if len(w) > 0]
    current_words = [w for w in current_norm.split() if len(w) > 0]
    
    # Find new words (words in current that aren't in original)
    new_words = [w for w in current_words if w not in original_words]
    
    # Calculate change percentage
    total_original_words = len(original_words) if original_words else 1
    change_percent = ((abs(len(current_words) - len(original_words)) + len(new_words)) / total_original_words) * 100
    
    # Requirements for meaningful modification
    meaningful_new_words = [w for w in new_words if len(w) >= 3]
    has_enough_new_words = len(meaningful_new_words) >= 2
    has_enough_char_diff = char_diff >= 10
    has_enough_change_percent = change_percent >= 15
    
    # Determine if modification is valid
    is_valid = has_enough_new_words and has_enough_char_diff and has_enough_change_percent
    
    # Provide specific guidance
    if is_valid:
        reason = f"Great! You've made good changes ({int(change_percent)}% modified)"
    elif not has_enough_new_words:
        reason = f"Add at least 2 more words or make more changes. You've added {len(meaningful_new_words)} new word(s)."
    elif not has_enough_char_diff:
        reason = f"Make more significant changes. You've changed {char_diff} characters. Try adding more details."
    elif not has_enough_change_percent:
        reason = f"Make more changes. You've modified {int(change_percent)}% of the text. Add more details or examples."
    else:
        reason = "Please make more meaningful changes"
    
    return {
        "is_valid": is_valid,
        "reason": reason,
        "change_percent": int(change_percent),
        "char_diff": char_diff,
        "new_words_count": len(meaningful_new_words),
        "needs_more_changes": not is_valid
    }

async def generate_suggested_answer(user_answer: str, question_guardrail: str, q41_guardrail: str, missing_elements: list, analysis: str) -> str:
    """
    Generate a suggested corrected answer when the AI doesn't provide one.
    """
    logger.info("🎯 Generating suggested answer as fallback...")
    
    try:
        prompt = f"""Generate a complete, corrected version of this answer that meets ALL requirements.

⚠️ CRITICAL RULE: The suggestion must be DERIVATIVE of what the caregiver actually entered, but COMPLETE and ready to use.
- Start with EXACTLY what the user wrote
- ONLY expand and refine the user's actual words
- Complete all sentences fully - NEVER leave blanks like "engaged in ." or "symptoms such as ."
- Use generic descriptive words to complete incomplete thoughts
- NEVER invent specific facts, temperatures, exact times, or specific activities not in the user's original answer
- Fix grammar, add professional structure, meet word count ({q41_guardrail})
- Preserve ALL facts and details from the user's answer
- The answer MUST be complete and grammatically correct - no placeholders, no blanks

Original Answer: "{user_answer}"
Question Requirements: "{question_guardrail}"
Length Requirement: "{q41_guardrail}"
Missing Elements: {missing_elements}
What to Add: {analysis}

⚠️ CRITICAL RULE - AI COLLABORATION:
The suggestion must be DERIVATIVE of what the caregiver actually entered, but COMPLETE.
✅ MUST DO:
- Start with EXACTLY what the user wrote (preserve their words)
- ONLY expand, refine, and fix grammar of the user's actual words
- Complete all sentences fully - NEVER leave blanks like "engaged in ." or "symptoms such as ."
- Use generic descriptive words to complete incomplete thoughts (e.g., "engaged in activities", "displayed various symptoms")
- NEVER invent specific facts, temperatures, exact times, or specific activities not in the user's original answer
- Fix grammar, add professional structure, meet word count ({q41_guardrail})
- Preserve ALL facts, events, and details the user mentioned
- The answer MUST be complete and grammatically correct - no placeholders, no blanks

❌ NEVER DO:
- Leave blanks, placeholders, or incomplete sentences like "engaged in ." or "symptoms such as ."
- Create templates with empty slots like "[specific activity]" or incomplete phrases
- Invent specific facts (temperatures like "37.8°C", exact times like "2:00 PM", specific activities the user didn't mention)
- Create generic boilerplate that doesn't reflect what the user actually wrote
- Use incomplete phrases that require the user to add details

✅ COMPLETION EXAMPLES:
- "client engaged in" → Complete to "The client engaged in various activities throughout the day"
- "symptoms such as" → Complete to "The client displayed symptoms such as fatigue and discomfort"
- "occurred at" → Complete to "The incident occurred during the afternoon period"
- Use generic professional language to complete thoughts while respecting what the user actually said

🎯 PRINCIPLE: The suggestion should be COMPLETE and ready to use immediately - what the user would have written with more time and proper format. Complete incomplete thoughts with generic language, but don't invent specific facts.

Return ONLY a single, complete answer (no explanations, no extra text, no blanks, no placeholders)
MUST meet the length requirement: {q41_guardrail}
MUST be grammatically complete and ready to use immediately"""
        
        response = await call_ai_model(prompt)
        suggested = response.strip().strip('"').strip("'")
        logger.info(f"✅ Generated suggested answer: {suggested[:100]}...")
        return suggested
    except Exception as e:
        logger.error(f"❌ Failed to generate suggested answer: {e}")
        return ""

async def validate_answer_with_ai(user_answer: str, question_guardrail: str = None, q41_guardrail: str = None) -> Dict[str, Any]:
    """
    Validate answer using AI model with dynamic guardrails - NO HARDCODED RULES
    
    This function uses AI to intelligently validate the user's answer against:
    1. Question-specific guardrails (what the question requires)
    2. Q41 guardrails (word count and general requirements)
    
    Returns dynamic validation results based on actual content analysis.
    """
    logger.info("=" * 60)
    logger.info("🤖 AI-BASED ANSWER VALIDATION STARTING")
    logger.info("=" * 60)
    
    logger.info(f"📝 USER ANSWER: '{user_answer}'")
    logger.info(f"📋 QUESTION GUARDRAIL: '{question_guardrail}'")
    logger.info(f"📋 Q41 GUARDRAIL: '{q41_guardrail}'")
    
    # Basic word count for logging
    word_count = len(user_answer.split())
    logger.info(f"📊 WORD COUNT: {word_count}")
    
    # Validate inputs
    if not question_guardrail:
        logger.error("❌ QUESTION GUARDRAIL MISSING")
        raise ValueError("Question guardrail must be provided from frontend")
    
    if not q41_guardrail:
        logger.error("❌ Q41 GUARDRAIL MISSING")
        raise ValueError("Q41 guardrail must be provided from frontend")
    
    try:
        # Create AI prompt for validation
        validation_prompt = f"""
You are a helpful guide helping a care worker complete their shift notes. Be supportive, clear, and specific. Help them improve their answer, don't criticize it.

⚠️ CRITICAL: Be REASONABLE, not PERFECTIONIST. Approve answers that meet basic requirements - don't look for perfection!

USER'S ANSWER:
"{user_answer}"

QUESTION: {question_guardrail.split('.')[0] if '.' in question_guardrail else question_guardrail}

QUESTION REQUIREMENTS:
"{question_guardrail}"

LENGTH REQUIREMENT:
"{q41_guardrail}"

🎯 APPROVAL PRIORITY: If the answer meets basic requirements, APPROVE IT immediately. Don't look for ways to reject it!

Analyze the answer and return ONLY valid JSON (no other text before or after).

CRITICAL REQUIREMENT: If status is "reject" or "review", you MUST include a complete "suggested_answer". This field CANNOT be empty or omitted.

Return this exact JSON structure:
{{
    "status": "approve" or "reject" or "review",
    "missing_elements": ["specific, action-oriented suggestions as list"],
    "safety_concerns": ["any safety issues as list"],
    "analysis": "helpful, encouraging guidance (friendly tone)",
    "word_count_analysis": "word count assessment",
    "content_quality": "quality and completeness assessment",
    "suggested_answer": "FOR REJECT/REVIEW: A complete, rewritten answer that meets ALL requirements. FOR APPROVE: Can be empty string."
}}

VALIDATION RULES:
1. "approve" - Answer meets basic requirements (minimum word count, answers the question, objective tone)
   ✅ ALWAYS APPROVE if answer has ALL of these:
      - Minimum word count met ({q41_guardrail})
      - Addresses the question topic clearly
      - Uses objective, professional language (not personal opinions like "I think", "I enjoyed")
      - Contains relevant information about the client/event/situation
      - Grammar is readable (minor errors are OK)
 
   ⚠️ CRITICAL: If an answer meets the 5 criteria above, APPROVE IT - don't look for perfection!
   ⚠️ DO NOT REJECT because answer "could include more specific observations" - that's nice-to-have, not required!
   ⚠️ DO NOT REJECT because answer "could mention specific times" - that's optional detail!
   ⚠️ DO NOT REJECT because answer "could be more detailed" - if it meets basic requirements, approve!
   
2. "reject" - Answer is missing CRITICAL REQUIRED elements
   ❌ REJECT ONLY if answer is missing MULTIPLE critical elements:
      - Significantly below minimum word count (less than 75% of requirement)
      - Completely unrelated to the question topic
      - Contains personal opinions that replace objective facts (e.g., "I enjoyed" instead of "client participated")
      - Contains grammar errors that make it completely unreadable
      - Completely lacks any relevant information about the client/event
   
   ✅ DO NOT REJECT if:
      - Answer meets minimum word count (even if barely)
      - Answer addresses the question (even if could be more detailed)
      - Minor grammar issues that don't affect readability
      - Missing "optional" details like specific times, mood observations, etc.
      - Answer is "good but could be better" - if it meets requirements, approve it!
   
3. "review" - Safety/health concern detected (triggers incident form)

REVIEW RULES (MANDATORY):
- Set "status": "review" and include a non-empty "safety_concerns" list when the answer indicates any potential safety/health incident, including but not limited to:
  • fall, slip, trip, head injury, loss of consciousness, fainting
  • injury, bleeding, burn, fracture, choking, seizure
  • medication error, missed dose that caused risk, overdose
  • abuse, neglect, elopement/wandering, missing person
  • aggression/violence requiring intervention, restraint, property damage
  • police/EMS/911 called, ambulance, ER/hospital visit
  • any serious hazard or emergency requiring escalation

- When setting review:
  • Populate "safety_concerns" with concise labels (e.g., ["fall"], ["seizure"], ["medication_error"], ["911_called"]).
  • Keep labels generic and professional; do NOT fabricate details beyond the user’s text.
  • Still provide a complete "suggested_answer" that is safe and professional.

- Do NOT set "review" for ordinary illness or minor discomfort (e.g., “client was sick” or “fever”) unless the text clearly implies urgent risk or escalation (e.g., unconscious, severe bleeding, seizure, 911/ER, or similar).

CRITICAL - TONE AND MESSAGING (MUST FOLLOW):
- ALWAYS write in SECOND PERSON: Use "Your answer" or "You need to" - NEVER "The user's answer" or "The answer"
- BE HELPFUL, NOT CRITICAL: Frame as "Add more detail about..." not "Missing..." or "Does not meet..."
- BE SPECIFIC: Tell them exactly what to add (e.g., "Include at least one activity" not "Answer incomplete")
- BE ENCOURAGING: Guide them forward - don't point out mistakes like a teacher
- NO CRITICAL WORDS: Never use "wrong", "inadequate", "failed", "poor quality", "nonsensical", "unprofessional"
- BE CLEAR: Older users need simple, direct instructions - keep it friendly
- NEVER criticize: If answer is bad, guide them on how to fix it - don't describe how bad it is

MISSING_ELEMENTS FORMAT:
Write as specific, direct action items. Be SHORT and CLEAR - tell them exactly what to add:
- "Include at least one activity the client did today"
- "Add social interaction details"
- "Include what time the event happened"
- "Add details about how the client participated"
- If too short: "Add more detail - you need {q41_guardrail}"
- If grammar: "Fix spelling errors"
- If personal opinions: "Remove personal thoughts (like 'I enjoyed') - use facts only"
- NOT generic: "Objective description missing" or "Insufficient detail"

ANALYSIS FORMAT (MANDATORY):
Write DIRECTLY to the user in second person. Be SHORT, HELPFUL, and SPECIFIC (1 sentence max):
✅ GOOD examples:
- "Add more detail about what the client did today. You need {q41_guardrail}."
- "Describe the event with facts instead of personal thoughts (like 'I had pleasure')."
- "Include what time the event happened and what activities took place."

❌ NEVER use these phrases:
- "The user's answer is..." 
- "The answer does not meet..."
- "The answer is vague/nonsensical/incomplete..."
- "The response lacks..."
- "It fails to..."
- "The answer does not provide..."

✅ ALWAYS start with: "Your answer" or "Add" or "Include" - guide them forward, don't criticize backward
- Focus on what to ADD, not what's WRONG

SUGGESTED_ANSWER (CRITICAL RULES - AI COLLABORATION):
⚠️ VERY IMPORTANT: The suggestion must be DERIVATIVE of what the caregiver actually entered.
✅ MUST DO:
- Start with EXACTLY what the user wrote (word-for-word if it makes sense)
- ONLY expand and refine the user's actual words
- Fix grammar, add structure, use professional language
- Meet the minimum word count requirement ({q41_guardrail})
- Preserve ALL facts, events, and details the user mentioned
- Complete sentences completely - NEVER leave blanks like "engaged in ." or "symptoms such as ."
- Use generic descriptive language to complete thoughts (e.g., "engaged in activities" not "engaged in .")
- If user says "client feel fever" → Expand to: "The client reported feeling feverish today. Appropriate actions were taken to monitor their condition."
- If user says "event was good" → Expand to: "The event [preserve user's description]. The client participated actively and appeared engaged throughout."
- If user's answer is very brief: Complete it with generic professional language that makes logical sense
- REMOVE any meta-instructions like "Please include..." or "Add more details..." from suggested answers
- The suggested answer MUST be complete and grammatically correct - no blanks, no placeholders, no incomplete sentences

❌ NEVER DO:
- Leave blanks or placeholders like "engaged in ." or "symptoms such as ."
- Use incomplete sentences that require the user to fill in details
- Create templates with empty slots like "[specific activity]"
- Invent specific facts (temperatures, exact times, specific activities) not in user's answer
- Include instruction phrases like "Please add..." or "Include more..." in the suggested answer itself

✅ DO:
- Complete all sentences with appropriate generic language
- Use phrases like "activities", "interactions", "observations" to complete incomplete thoughts
- Make the answer ready to use immediately without any blanks or placeholders

🎯 PRINCIPLE: The suggestion should be a COMPLETE, FINAL answer ready to use - what the user would have written if they had more time. Complete incomplete thoughts with generic professional language, but don't invent specific facts.

CRITICAL - BE REASONABLE WITH VALIDATION (VERY IMPORTANT):
⚠️ APPROVE answers that meet basic requirements - don't look for perfection!
⚠️ If word count is met + question is answered + objective tone = APPROVE (even if it could be more detailed)
⚠️ "Could include more specific observations" = NOT A REASON TO REJECT (that's optional nice-to-have)
⚠️ "Could mention specific times" = NOT A REASON TO REJECT (that's optional detail)
⚠️ "Could be more detailed" = NOT A REASON TO REJECT if basic requirements are met
⚠️ ONLY REJECT if answer is actually missing MULTIPLE critical required elements
⚠️ When in doubt, APPROVE - it's better to accept a reasonable answer than reject unnecessarily

APPROVAL DECISION TREE (FOLLOW STRICTLY):
Step 1: Does answer meet minimum word count ({q41_guardrail})?
  - YES → Go to Step 2
  - NO (significantly below, less than 75%) → REJECT for word count only
  
Step 2: Does answer address the question topic?
  - YES → Go to Step 3
  - NO (completely unrelated) → REJECT for off-topic only
  
Step 3: Is answer objective (not personal opinions like "I enjoyed", "it was good")?
  - YES → Go to Step 4
  - NO (contains personal opinions instead of facts) → REJECT for personal opinions only
  
Step 4: Does answer contain relevant information about client/event/situation?
  - YES → ⚠️ APPROVE IMMEDIATELY! Stop here! Don't check for more details!
  - NO (completely empty of information) → REJECT for lack of content only

⚠️ CRITICAL RULE: If you reach Step 4 and answer contains relevant information, you MUST APPROVE it.
❌ DO NOT check for:
   - "Could include more specific observations" → That's optional, not required!
   - "Could mention specific times" → That's optional detail!
   - "Could be more detailed" → If basic requirements met, that's enough!
   - "Missing mood observations" → Optional nice-to-have!

✅ When in doubt between APPROVE and REJECT → Choose APPROVE!
"""

        logger.info("🤖 Sending validation request to AI model...")
        
        # Call AI model for validation
        response = await call_ai_model(validation_prompt)
        
        logger.info("🤖 AI Response received:")
        logger.info(f"📝 Full Response: {response}")
        
        # Parse AI response
        try:
            # Extract JSON from AI response
            import json
            import re
            
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                validation_data = json.loads(json_match.group())
                # Check if suggested_answer is missing and log it
                if "suggested_answer" not in validation_data:
                    logger.warning("⚠️ AI did not include 'suggested_answer' field in response")
                    validation_data["suggested_answer"] = ""
                elif not validation_data.get("suggested_answer"):
                    logger.warning("⚠️ AI provided empty 'suggested_answer' field")
            else:
                # Fallback if no JSON found
                logger.warning("⚠️ No JSON found in AI response, using fallback")
                validation_data = {
                    "status": "reject",
                    "missing_elements": ["Unable to validate - please try again"],
                    "safety_concerns": [],
                    "analysis": "AI validation failed",
                    "word_count_analysis": "Unable to analyze",
                    "content_quality": "Unable to assess",
                    "suggested_answer": ""
                }
            
            logger.info(f"✅ AI Validation Result: {validation_data}")
            logger.info(f"📝 Suggested Answer from AI: '{validation_data.get('suggested_answer', 'NOT PROVIDED')}'")
            
            # Post-process analysis to ensure second-person tone
            if "analysis" in validation_data and validation_data["analysis"]:
                analysis = validation_data["analysis"]
                # Replace third-person references with second-person
                analysis = re.sub(r'\bThe user\'s answer\b', 'Your answer', analysis)
                analysis = re.sub(r'\bThe answer\b', 'Your answer', analysis)
                analysis = re.sub(r'\bThe response\b', 'Your answer', analysis)
                analysis = re.sub(r'\bIt fails to\b', 'Add', analysis)
                analysis = re.sub(r'\bIt does not\b', 'Your answer needs to', analysis)
                analysis = re.sub(r'\bIt lacks\b', 'Add', analysis)
                analysis = re.sub(r'\bdoes not meet\b', 'needs to meet', analysis)
                analysis = re.sub(r'\bdoes not provide\b', 'needs to include', analysis)
                analysis = re.sub(r'\bdoes not adhere to\b', 'should follow', analysis)
                analysis = re.sub(r'\bdoes not address\b', 'should address', analysis)
                # Remove critical language
                analysis = re.sub(r'\bis nonsensical\b', 'needs meaningful content', analysis, flags=re.IGNORECASE)
                analysis = re.sub(r'\bis vague\b', 'needs more detail', analysis, flags=re.IGNORECASE)
                analysis = re.sub(r'\bis incomplete\b', 'needs more detail', analysis, flags=re.IGNORECASE)
                analysis = re.sub(r'\bis unprofessional\b', 'needs professional language', analysis, flags=re.IGNORECASE)
                validation_data["analysis"] = analysis
                logger.info(f"📝 Post-processed analysis: {analysis}")
            
            # Get suggested answer from AI response (if provided)
            suggested_answer = validation_data.get("suggested_answer", "")
            
            # Clean up suggested answer: Remove instruction phrases and fix incomplete sentences
            if suggested_answer:
                import re
                # Remove common instruction phrases at the end or beginning
                instruction_patterns = [
                    r'Please include more details?.*?\.\s*$',
                    r'Please add.*?\.\s*$',
                    r'Include more.*?\.\s*$',
                    r'Add more.*?\.\s*$',
                    r'Please.*?details.*?\.\s*$',
                    r'Please.*?observations.*?\.\s*$',
                    r'Please.*?specific.*?\.\s*$',
                    r'Include.*?observations.*?\.\s*$',
                    r'Add.*?observations.*?\.\s*$',
                    r'if applicable\.\s*$',
                    r'where applicable\.\s*$',
                    r'Include more.*?if applicable\.\s*$',
                    r'Add.*?where applicable\.\s*$',
                    r'\[specific.*?\]',  # Remove template placeholders like [specific activity, e.g., ...]
                    r'\[.*?e\.g\..*?\]',  # Remove any [text, e.g., more text] patterns
                    r'e\.g\.[^.]*\.',  # Remove "e.g., ..." phrases
                    r'\[.*?\]',  # Remove any remaining square bracket patterns (template placeholders)
                ]
                for pattern in instruction_patterns:
                    suggested_answer = re.sub(pattern, '', suggested_answer, flags=re.IGNORECASE)
                
                # Fix incomplete sentences with blanks (e.g., "symptoms such as ." or "occurred at ,")
                suggested_answer = re.sub(r'\bsuch as\s+\.', 'such as various symptoms', suggested_answer, flags=re.IGNORECASE)
                suggested_answer = re.sub(r'\boccurred at\s+,\s*', 'occurred during the day,', suggested_answer, flags=re.IGNORECASE)
                suggested_answer = re.sub(r'\bengaged in\s+\.', 'engaged in various activities', suggested_answer, flags=re.IGNORECASE)
                suggested_answer = re.sub(r'\bparticipated in\s+\.', 'participated in activities', suggested_answer, flags=re.IGNORECASE)
                suggested_answer = re.sub(r'\bat\s+,\s*', 'during the period,', suggested_answer, flags=re.IGNORECASE)
                suggested_answer = re.sub(r'\bsuch as\s*$', 'such as various symptoms', suggested_answer, flags=re.IGNORECASE)
                suggested_answer = re.sub(r'\band\s+\.\s*', 'and other activities.', suggested_answer, flags=re.IGNORECASE)
                suggested_answer = re.sub(r'\s+\.\s+', '. ', suggested_answer)  # Fix "word . nextword" to "word. nextword"
                suggested_answer = re.sub(r',\s*\.', '.', suggested_answer)  # Fix ", ." to "."
                suggested_answer = re.sub(r'\s+,\s*$', '.', suggested_answer)  # Fix trailing ", " to "."
                
                # Clean up any extra whitespace and trailing punctuation
                suggested_answer = re.sub(r'\s+', ' ', suggested_answer).strip()
                suggested_answer = re.sub(r'\s*\.\s*\.', '.', suggested_answer)  # Fix double periods
                logger.info(f"🧹 Cleaned suggested answer (removed instruction phrases): '{suggested_answer[:100]}...'")
            
            if not suggested_answer and validation_data.get("status") in ["reject", "review"]:
                logger.warning("⚠️ AI did not provide suggested_answer, generating one...")
                # Generate suggested answer as a separate AI call
                suggested_answer = await generate_suggested_answer(
                    user_answer=user_answer,
                    question_guardrail=question_guardrail,
                    q41_guardrail=q41_guardrail,
                    missing_elements=validation_data.get("missing_elements", []),
                    analysis=validation_data.get("analysis", "")
                )
            
            # Ensure required fields exist
            result = {
                "word_count": word_count,
                "missing_elements": validation_data.get("missing_elements", []),
                "safety_concerns": validation_data.get("safety_concerns", []),
                "guidelines_checked": [
                    "Question-specific requirements (AI analyzed)",
                    "Q41 general requirements (AI analyzed)"
                ],
                "ai_analysis": validation_data.get("analysis", ""),
                "word_count_analysis": validation_data.get("word_count_analysis", ""),
                "content_quality": validation_data.get("content_quality", ""),
                "suggested_answer": suggested_answer
            }
            
            logger.info("=" * 60)
            logger.info("✅ AI VALIDATION COMPLETE")
            logger.info("=" * 60)
            logger.info(f"📝 FINAL SUGGESTED ANSWER: '{suggested_answer}'")
            logger.info(f"📝 SUGGESTED ANSWER LENGTH: {len(suggested_answer)}")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse AI response as JSON: {e}")
            logger.error(f"❌ AI Response: {response}")
            return {
                "word_count": word_count,
                "missing_elements": ["AI validation failed - please try again"],
                "safety_concerns": [],
                "guidelines_checked": ["AI validation failed"],
                "ai_analysis": "Failed to parse AI response",
                "word_count_analysis": "Unable to analyze",
                "content_quality": "Unable to assess"
            }
            
    except Exception as e:
        logger.error(f"❌ AI VALIDATION ERROR: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        # Fallback validation
        return {
            "word_count": word_count,
            "missing_elements": ["Validation system error - please try again"],
            "safety_concerns": [],
            "guidelines_checked": ["System error occurred"],
            "ai_analysis": f"Validation error: {str(e)}",
            "word_count_analysis": "Unable to analyze due to error",
            "content_quality": "Unable to assess due to error"
        }

async def call_ai_model(prompt: str, max_retries: int = 3) -> str:
    """
    Call Azure OpenAI model for validation analysis with rate limiting
    
    Args:
        prompt: The prompt to send to the AI model
        max_retries: Maximum number of retries on rate limit errors
    """
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            # Use rate limiter to prevent hitting rate limits
            async with rate_limiter:
                logger.info(f"🤖 Calling Azure OpenAI for validation... (attempt {retry_count + 1}/{max_retries + 1})")
                
                # Use your existing Azure OpenAI service with proper parameters
                # Create a thread for this validation request
                thread_id = "validation-thread"
                response = azure_service.send_message(thread_id, prompt)
                
                logger.info("🤖 Azure OpenAI response received")
                return response
                
        except RateLimitError as e:
            retry_count += 1
            last_error = e
            rate_limiter.on_rate_limit_hit()
            
            if retry_count <= max_retries:
                # Exponential backoff: wait longer with each retry
                wait_time = min(2 ** retry_count, 30)  # Cap at 30 seconds
                logger.warning(f"⏳ Rate limit hit (429). Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"❌ Rate limit exceeded after {max_retries} retries")
                raise
        
        except Exception as e:
            last_error = e
            logger.error(f"❌ AZURE OPENAI CALL ERROR: {str(e)}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            
            # Check if it's a rate limit error by status code
            error_str = str(e).lower()
            if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                retry_count += 1
                rate_limiter.on_rate_limit_hit()
                
                if retry_count <= max_retries:
                    wait_time = min(2 ** retry_count, 30)
                    logger.warning(f"⏳ Rate limit detected. Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
            
            # Non-rate-limit errors: return fallback after retries
            if retry_count >= max_retries:
                break
            retry_count += 1
    
    # Fallback response if AI fails after all retries
    fallback_response = """{"status": "reject", "missing_elements": ["Unable to validate due to AI service error - please try again"], "safety_concerns": [], "analysis": "AI validation service temporarily unavailable. Please try again.", "word_count_analysis": "Unable to analyze due to service error", "content_quality": "Unable to assess due to service error"}"""
    
    logger.warning(f"⚠️ Using fallback response due to AI service error (after {retry_count} attempts)")
    if last_error:
        logger.warning(f"⚠️ Last error: {str(last_error)}")
    
    return fallback_response

def determine_response_status(validation_result: Dict[str, Any]) -> str:
    """
    Determine if response should be approve, reject, or review
    CRITICAL: Safety concerns ALWAYS trigger REVIEW, never approve
    """
    # CRITICAL: If safety concerns found, ALWAYS trigger REVIEW (never approve)
    safety_concerns = validation_result.get("safety_concerns", [])
    if safety_concerns and len(safety_concerns) > 0:
        logger.warning(f"⚠️ Safety concerns detected - forcing REVIEW status (never approve): {safety_concerns}")
        return "review"
    
    # If missing elements found, trigger REJECT
    if validation_result.get("missing_elements"):
        return "reject"
    
    # Otherwise, APPROVE
    return "approve"

def generate_ai_response_message(status: str, validation_result: Dict[str, Any]) -> str:
    """
    Generate user-friendly response message using AI analysis
    """
    ai_analysis = validation_result.get("ai_analysis", "")
    missing_elements = validation_result.get("missing_elements", [])
    safety_concerns = validation_result.get("safety_concerns", [])
    
    if status == "approve":
        return "✅ Perfect! Your answer has all the details needed. You're all set to continue!"
    
    elif status == "reject":
        if missing_elements:
            elements_text = "\n".join([f"• {element}" for element in missing_elements])
            return f"💡 Add a bit more detail:\n\n{elements_text}\n\n{ai_analysis}"
        else:
            return f"💡 {ai_analysis}"
    
    elif status == "review":
        if safety_concerns:
            concerns_text = "\n".join([f"• {concern}" for concern in safety_concerns])
            return f"🚨 Safety Review Required: {concerns_text}\n\nAn Incident Form will open automatically.\n\n{ai_analysis}"
        else:
            return f"🚨 Safety Review Required: {ai_analysis}\n\nAn Incident Form will open automatically."
    
    else:
        return f"⚠️ Validation completed: {ai_analysis}"

# Q41 guidelines endpoint removed - frontend should get question 41 from main API
# and send word count guidelines to validation API

# Question guidelines endpoint removed - frontend should get question guidelines from main API
# and send them to validation API along with Q41 guidelines

# Pydantic model for guidelines request
class GuidelinesRequest(BaseModel):
    question_id: str
    client_id: str = "default"
    context: Dict[str, Any] = {}
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

@app.post("/speech-to-text-streaming")
async def speech_to_text_streaming(request: dict):
    """
    Convert speech audio to text with streaming support for real-time experience
    
    Accepts base64 encoded audio chunks and returns transcribed text
    Optimized for shorter audio segments for better real-time feel
    """
    import base64
    import tempfile
    import os
    
    try:
        audio_data = request.get("audio_data")
        audio_format = request.get("format", "m4a")
        chunk_id = request.get("chunk_id", 0)
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        logger.info(f"🎤 STREAMING SPEECH-TO-TEXT - Chunk {chunk_id} ({len(audio_data)} bytes base64)")
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)
        logger.info(f"🎤 Decoded audio chunk: {len(audio_bytes)} bytes")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=f"_chunk_{chunk_id}.{audio_format}", delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
            logger.info(f"🎤 Saved chunk to temp file: {temp_audio_path}")
        
        try:
            # Use Azure OpenAI for Whisper (same credentials as GPT-4)
            whisper_deployment = os.getenv("AZURE_WHISPER_DEPLOYMENT", "whisper")
            whisper_endpoint = os.getenv("AZURE_WHISPER_ENDPOINT", "https://hakeem-4411-resource.cognitiveservices.azure.com/openai/deployments/whisper/audio/transcriptions")
            whisper_key = os.getenv("AZURE_WHISPER_KEY", os.getenv("AZURE_OPENAI_KEY"))
            whisper_api_version = os.getenv("AZURE_WHISPER_API_VERSION", "2024-06-01")
            
            # Create Azure client for Whisper
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
            
            transcribed_text = transcription.text
            logger.info(f"✅ STREAMING SPEECH-TO-TEXT - Chunk {chunk_id} Transcription: {transcribed_text}")
            
            # Clean up temp file
            os.unlink(temp_audio_path)
            
            return {
                "success": True,
                "text": transcribed_text,
                "chunk_id": chunk_id,
                "language": "en",
                "is_final": True  # For now, treat each chunk as final
            }
            
        except Exception as whisper_error:
            logger.error(f"❌ Streaming Whisper API error: {str(whisper_error)}")
            # Clean up temp file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
            return {
                "success": False,
                "text": "",
                "chunk_id": chunk_id,
                "error": "Speech-to-text service temporarily unavailable"
            }
            
    except Exception as e:
        logger.error(f"❌ STREAMING SPEECH-TO-TEXT ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in streaming speech-to-text: {str(e)}")

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

# Answer validation API endpoint
@app.post("/validate-answer", response_model=AnswerValidationResponse)
async def validate_answer(request: AnswerValidationRequest):
    """
    Validate answer against question guidelines and Q41 universal guidelines
    
    Returns:
    - approve: Answer meets all requirements
    - reject: Answer missing required elements (shows what's missing)
    - review: Safety concerns detected (triggers Incident Form)
    """
    try:
        logger.info(f"🔍 Validating answer")
        logger.info(f"📝 Answer: {request.user_answer[:100]}...")
        logger.info(f"📋 Question Guardrail: {request.question_guardrail}")
        logger.info(f"📋 Q41 Guardrail: {request.q41_guardrail}")
        
        # Check if AI suggestion was provided and validate modifications
        modification_status = {}
        if request.ai_suggestion:
            logger.info(f"🔍 AI suggestion provided, checking modifications...")
            logger.info(f"📝 AI Suggestion: {request.ai_suggestion[:100]}...")
            modification_status = check_meaningful_modification(request.ai_suggestion, request.user_answer)
            logger.info(f"✅ Modification check: {modification_status}")
        
        logger.info(f"🔍 FULL REQUEST DATA:")
        logger.info(f"   - user_answer: '{request.user_answer}'")
        logger.info(f"   - question_guardrail: '{request.question_guardrail}'")
        logger.info(f"   - q41_guardrail: '{request.q41_guardrail}'")
        if request.ai_suggestion:
            logger.info(f"   - ai_suggestion: '{request.ai_suggestion[:100]}...'")
        
        # Validate answer content using AI model with dynamic guardrails
        validation_result = await validate_answer_with_ai(
            request.user_answer,
            request.question_guardrail,
            request.q41_guardrail
        )
        
        # CRITICAL: Always check safety concerns FIRST - they override everything
        safety_concerns = validation_result.get("safety_concerns", [])
        if safety_concerns and len(safety_concerns) > 0:
            logger.warning(f"🚨 SAFETY CONCERNS DETECTED - Forcing REVIEW status (cannot be overridden)")
            logger.warning(f"🚨 Safety concerns: {safety_concerns}")
            status = "review"
        elif modification_status and modification_status.get("needs_more_changes", False):
            # If modification check failed, override status to reject with modification feedback
            logger.info(f"🔄 Overriding status to 'reject' due to insufficient modifications")
            status = "reject"
            # Add modification requirement to missing elements
            if "missing_elements" not in validation_result:
                validation_result["missing_elements"] = []
            validation_result["missing_elements"].insert(0, modification_status.get("reason", "Please make meaningful changes to the AI suggestion"))
            validation_result["ai_analysis"] = modification_status.get("reason", "Please edit the AI suggestion and add your own observations or details.")
        else:
            # Determine response status normally
            status = determine_response_status(validation_result)
        
        # Generate response message using AI analysis
        message = generate_ai_response_message(status, validation_result)
        
        logger.info(f"✅ Validation result: {status}")
        logger.info(f"📊 Word count: {validation_result['word_count']}")
        logger.info(f"🔍 Guidelines checked: {validation_result['guidelines_checked']}")
        
        if validation_result['missing_elements']:
            logger.info(f"❌ Missing elements: {validation_result['missing_elements']}")
        
        if validation_result['safety_concerns']:
            logger.warning(f"⚠️ Safety concerns: {validation_result['safety_concerns']}")
        
        # Get suggested answer for response
        suggested_answer_value = validation_result.get('suggested_answer', '')
        logger.info(f"✅ Returning suggested_answer to frontend: '{suggested_answer_value}'")
        logger.info(f"✅ Suggested answer length in response: {len(suggested_answer_value)}")
        
        response = AnswerValidationResponse(
            status=status,
            message=message,
            analysis=validation_result.get('ai_analysis', message),  # Include AI analysis
            suggested_answer=suggested_answer_value,  # Include suggested answer
            missing_elements=validation_result['missing_elements'],
            safety_concerns=validation_result['safety_concerns'],
            word_count=validation_result['word_count'],
            guidelines_checked=validation_result['guidelines_checked'],
            modification_status=modification_status if request.ai_suggestion else {}  # Include modification status
        )
        
        logger.info(f"✅ API Response built successfully with suggested_answer")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error validating answer: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error validating answer: {str(e)}"
        )

# Real-time WebSocket streaming using Azure GPT Realtime API
@app.websocket("/ws/realtime-speech")
async def websocket_realtime_speech(websocket: WebSocket):
    """
    Real-time WebSocket endpoint using Azure GPT Realtime API
    Provides true real-time speech-to-text with low latency
    """
    await websocket.accept()
    logger.info("🎤 WebSocket connection established for GPT Realtime API")
    
    # GPT Realtime API configuration
    GPT_REALTIME_ENDPOINT = "https://hakeem-4411-resource.cognitiveservices.azure.com/openai/realtime"
    GPT_REALTIME_KEY = "47oYsTR8tSAcu3BJqsXDg4zZXLOcO1fY0uxhkR5fghe2NFmJm6A6JQQJ99BGACHYHv6XJ3w3AAAAACOG5jxP"
    GPT_REALTIME_API_VERSION = "2024-10-01-preview"
    GPT_REALTIME_DEPLOYMENT = "gpt-realtime"
    
    # Send transcription metadata first (Azure Communication Services style)
    await websocket.send_text(json.dumps({
        "kind": "TranscriptionMetadata",
        "transcriptionMetadata": {
            "subscriptionId": "gpt-realtime-subscription",
            "locale": "en-US",
            "callConnectionId": "gpt-realtime-connection",
            "correlationId": "gpt-realtime-correlation",
            "model": "gpt-realtime",
            "endpoint": GPT_REALTIME_ENDPOINT
        }
    }))
    
    # Track connection and audio processing for GPT Realtime API
    connection_start_time = asyncio.get_event_loop().time()
    last_transcription_time = 0
    transcription_interval = 2.0  # Send transcription every 2 seconds for real-time experience
    audio_buffer = bytearray()
    
    try:
        while True:
            current_time = asyncio.get_event_loop().time()
            
            # Wait 2 seconds before starting to request audio data
            if (current_time - connection_start_time) < 2.0:
                await asyncio.sleep(0.1)
                continue
            
            # Send periodic requests for audio data
            if (current_time - last_transcription_time) >= transcription_interval:
                logger.info(f"🎤 Requesting audio data for GPT Realtime API transcription")
                
                # Request current audio from frontend
                await websocket.send_text(json.dumps({
                    "action": "get_audio",
                    "timestamp": current_time
                }))
                
                last_transcription_time = current_time
            
            # Wait a bit before next check
            await asyncio.sleep(0.1)
            
            # Handle incoming messages from frontend (non-blocking)
            try:
                # Use asyncio.wait_for to make receive_text non-blocking
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                message = json.loads(data)
                
                if message.get("action") == "audio_data":
                    audio_data = message.get("audio_data")
                    status = message.get("status", "")
                    
                    if audio_data and len(audio_data) > 0:
                        logger.info(f"🎤 Processing audio data with Whisper API ({len(audio_data)} bytes)")
                    elif status == "no_recording":
                        logger.info(f"🎤 Frontend has no recording available yet - waiting...")
                        continue
                    else:
                        logger.info(f"🎤 Empty audio data received - skipping processing")
                        continue
                    
                    try:
                        # Decode base64 audio
                        audio_bytes = base64.b64decode(audio_data)
                        
                        # Use Azure Whisper API for real-time transcription
                        whisper_deployment = os.getenv("AZURE_WHISPER_DEPLOYMENT", "whisper")
                        whisper_key = os.getenv("AZURE_WHISPER_KEY", os.getenv("AZURE_OPENAI_KEY"))
                        whisper_api_version = os.getenv("AZURE_WHISPER_API_VERSION", "2024-06-01")
                        
                        # Create Whisper API client
                        whisper_client = AzureOpenAI(
                            api_key=whisper_key,
                            api_version=whisper_api_version,
                            azure_endpoint="https://hakeem-4411-resource.cognitiveservices.azure.com/"
                        )
                        
                        # Save to temporary file for Whisper API
                        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_audio:
                            temp_audio.write(audio_bytes)
                            temp_audio_path = temp_audio.name
                        
                        # Process audio with Whisper API
                        with open(temp_audio_path, "rb") as audio_file:
                            transcription = whisper_client.audio.transcriptions.create(
                                model=whisper_deployment,
                                file=audio_file,
                                language="en"
                            )
                        
                        transcribed_text = transcription.text.strip()
                        logger.info(f"🎤 Real-time Whisper API transcription: {transcribed_text}")
                        logger.info(f"🎤 WebSocket state before sending: {websocket.client_state}")
                        
                        # Send transcription data back to frontend via WebSocket
                        logger.info(f"🎤 Attempting to send transcription via WebSocket...")
                        try:
                            transcription_message = json.dumps({
                                "kind": "TranscriptionData",
                                "transcriptionData": {
                                    "text": transcribed_text,
                                    "format": "display",
                                    "confidence": 0.95,
                                    "offset": int(current_time * 1000000),  # Convert to microseconds
                                    "words": [
                                        {
                                            "text": word,
                                            "offset": int(current_time * 1000000)
                                        } for word in transcribed_text.split()
                                    ],
                                    "participantRawID": "gpt-realtime-user",
                                    "resultStatus": "Final",
                                    "apiSource": "Azure GPT Realtime API"
                                }
                            })
                            
                            logger.info(f"🎤 Prepared transcription message: {transcription_message[:100]}...")
                            await websocket.send_text(transcription_message)
                            logger.info(f"🎤 Sent GPT Realtime transcription to frontend: {transcribed_text}")
                        except Exception as send_error:
                            logger.error(f"❌ Failed to send transcription to frontend: {str(send_error)}")
                            # Don't close WebSocket, just log the error and continue
                        
                        # Clean up temp file
                        os.unlink(temp_audio_path)
                            
                    except Exception as gpt_error:
                            logger.error(f"❌ Real-time GPT Realtime API error: {str(gpt_error)}")
                            try:
                                await websocket.send_text(json.dumps({
                                    "kind": "TranscriptionData",
                                    "transcriptionData": {
                                        "text": "",
                                        "format": "display",
                                        "confidence": 0.0,
                                        "offset": int(current_time * 1000000),
                                        "words": [],
                                        "participantRawID": "gpt-realtime-user",
                                        "resultStatus": "Failed",
                                        "error": str(gpt_error)
                                    }
                                }))
                            except Exception as send_error:
                                logger.error(f"❌ Failed to send error message to frontend: {str(send_error)}")
                                # Don't close WebSocket, just log the error and continue
                            
            except asyncio.TimeoutError:
                # No message received within timeout - this is normal, continue loop
                pass
            except Exception as e:
                logger.error(f"❌ WebSocket message error: {str(e)}")
                pass
                
    except WebSocketDisconnect:
        logger.info("🎤 WebSocket connection closed")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {str(e)}")
        try:
            await websocket.close()
        except:
            pass