"""
AI Prescreener System - Complete Implementation
==============================================

This module implements the complete AI Prescreener system with four core functionalities:
1. Event-Based Flagging System (Safety & Wellness Monitoring)
2. Compliance and Restriction Monitoring (Adherence to Care Plan)
3. Dynamic Narrative Generation (Automated Shift Reporting)
4. Customizable Guardrails Framework (Client-Specific Logic)

Author: Metaxoft AI Assistant
Version: 1.0.0
Priority: HIGH - URGENT PROJECT
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels for flagged events"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class EventType(Enum):
    """Types of events that can be flagged"""
    SAFETY_INCIDENT = "safety_incident"
    MEDICAL_EVENT = "medical_event"
    COMPLIANCE_VIOLATION = "compliance_violation"
    RESTRICTION_BREACH = "restriction_breach"
    GENERAL_FLAG = "general_flag"

@dataclass
class FlaggedEvent:
    """Represents a flagged event from the AI Prescreener"""
    event_id: str
    client_id: str
    event_type: EventType
    severity: AlertSeverity
    description: str
    original_text: str
    timestamp: datetime
    requires_escalation: bool
    escalation_notes: Optional[str] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None

@dataclass
class ClientGuardrails:
    """Client-specific guardrails and restrictions"""
    client_id: str
    client_name: str
    restricted_topics: List[str]
    allowed_services: List[str]
    medication_restrictions: List[str]
    dietary_restrictions: List[str]
    activity_restrictions: List[str]
    visitor_restrictions: List[str]
    special_instructions: List[str]
    narrative_requirements: Dict[str, Any]
    custom_flags: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    is_active: bool = True

@dataclass
class ShiftData:
    """Represents shift data input from care workers"""
    shift_id: str
    client_id: str
    worker_id: str
    shift_date: datetime
    shift_duration_hours: float
    is_overnight_shift: bool
    worker_notes: str
    completed_tasks: List[str]
    services_provided: List[str]
    additional_context: Dict[str, Any]

@dataclass
class PrescreenerResult:
    """Complete result from AI Prescreener analysis"""
    analysis_id: str
    shift_id: str
    client_id: str
    flagged_events: List[FlaggedEvent]
    compliance_violations: List[Dict[str, Any]]
    generated_narrative: str
    narrative_length: int
    processing_time_ms: int
    confidence_score: float
    requires_human_review: bool
    analysis_timestamp: datetime

class AIPrescreenerCore:
    """
    Core AI Prescreener Engine
    Handles all four main functionalities of the AI Prescreener system
    """
    
    def __init__(self, azure_openai_service):
        self.azure_service = azure_openai_service
        self.client_guardrails_db = {}  # In production, this would be a real database
        self.flagged_events_db = {}     # In production, this would be a real database
        self.initialize_default_patterns()
    
    def initialize_default_patterns(self):
        """Initialize default patterns for event detection"""
        self.safety_patterns = {
            "fall": [
                r"\b(?:client|patient|resident)\s+(?:fell|fall|fallen|falling)\b",
                r"\b(?:fell|fall|fallen|falling)\s+(?:down|over|backwards|forwards)\b",
                r"\b(?:accident|incident)\s+(?:involving|with)\s+(?:fall|falling)\b"
            ],
            "fever": [
                r"\b(?:client|patient|resident)\s+(?:had|has|having)\s+(?:a\s+)?fever\b",
                r"\b(?:fever|temperature|temp)\s+(?:high|elevated|spiked)\b",
                r"\b(?:running|has)\s+(?:a\s+)?fever\b"
            ],
            "injury": [
                r"\b(?:injured|hurt|wounded|bruised|cut|scraped)\b",
                r"\b(?:injury|wound|bruise|cut|scrape)\b",
                r"\b(?:pain|ache|sore|tender)\s+(?:in|on|at)\b"
            ],
            "medical_emergency": [
                r"\b(?:emergency|urgent|critical|serious)\b",
                r"\b(?:ambulance|called|911|emergency\s+services)\b",
                r"\b(?:hospital|ER|emergency\s+room)\b"
            ]
        }
        
        self.compliance_patterns = {
            "medication": [
                r"\b(?:medication|med|pill|dose|prescription)\b",
                r"\b(?:gave|administered|provided)\s+(?:medication|med|pill)\b",
                r"\b(?:missed|skipped|forgot)\s+(?:medication|med|dose)\b"
            ],
            "meal_prep": [
                r"\b(?:meal|food|dinner|lunch|breakfast|snack)\s+(?:prep|preparation|prepared)\b",
                r"\b(?:cooked|prepared|made)\s+(?:meal|food|dinner|lunch|breakfast)\b",
                r"\b(?:dietary|diet|nutrition|nutritional)\b"
            ]
        }
    
    def analyze_shift(self, shift_data: ShiftData) -> PrescreenerResult:
        """
        Main analysis function that processes shift data through all four core functionalities
        """
        start_time = datetime.now()
        analysis_id = str(uuid.uuid4())
        
        logger.info(f"🔍 Starting AI Prescreener analysis for shift {shift_data.shift_id}")
        
        # Get client-specific guardrails
        guardrails = self.get_client_guardrails(shift_data.client_id)
        
        # 1. Event-Based Flagging System
        flagged_events = self.detect_safety_events(shift_data, guardrails)
        
        # 2. Compliance and Restriction Monitoring
        compliance_violations = self.check_compliance_violations(shift_data, guardrails)
        
        # 3. Dynamic Narrative Generation
        generated_narrative = self.generate_shift_narrative(shift_data, guardrails, flagged_events)
        
        # 4. Determine if human review is required
        requires_human_review = (
            len(flagged_events) > 0 or 
            len(compliance_violations) > 0 or
            any(event.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH] for event in flagged_events)
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = PrescreenerResult(
            analysis_id=analysis_id,
            shift_id=shift_data.shift_id,
            client_id=shift_data.client_id,
            flagged_events=flagged_events,
            compliance_violations=compliance_violations,
            generated_narrative=generated_narrative,
            narrative_length=len(generated_narrative),
            processing_time_ms=int(processing_time),
            confidence_score=self.calculate_confidence_score(flagged_events, compliance_violations),
            requires_human_review=requires_human_review,
            analysis_timestamp=datetime.now()
        )
        
        # Store results for tracking
        self.store_analysis_result(result)
        
        logger.info(f"✅ AI Prescreener analysis completed in {processing_time:.2f}ms")
        logger.info(f"📊 Results: {len(flagged_events)} events flagged, {len(compliance_violations)} violations detected")
        
        return result
    
    def detect_safety_events(self, shift_data: ShiftData, guardrails: ClientGuardrails) -> List[FlaggedEvent]:
        """
        3.1. Event-Based Flagging System (Safety & Wellness Monitoring)
        Detects predefined critical incidents and generates immediate alerts
        """
        flagged_events = []
        text_lower = shift_data.worker_notes.lower()
        
        logger.info("🚨 Analyzing for safety events...")
        
        # Check for predefined safety patterns
        for event_type, patterns in self.safety_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    event = FlaggedEvent(
                        event_id=str(uuid.uuid4()),
                        client_id=shift_data.client_id,
                        event_type=EventType.SAFETY_INCIDENT,
                        severity=self.determine_severity(event_type),
                        description=f"Detected {event_type}: {match.group()}",
                        original_text=match.group(),
                        timestamp=datetime.now(),
                        requires_escalation=True,
                        escalation_notes=f"Automatic detection of {event_type} in shift notes"
                    )
                    flagged_events.append(event)
                    logger.warning(f"🚨 FLAGGED EVENT: {event_type.upper()} - {event.description}")
        
        # Check custom client-specific flags
        for custom_flag in guardrails.custom_flags:
            if self.check_custom_flag(text_lower, custom_flag):
                event = FlaggedEvent(
                    event_id=str(uuid.uuid4()),
                    client_id=shift_data.client_id,
                    event_type=EventType.GENERAL_FLAG,
                    severity=AlertSeverity(custom_flag.get("severity", "medium")),
                    description=f"Custom flag triggered: {custom_flag.get('description', 'Unknown')}",
                    original_text=text_lower,
                    timestamp=datetime.now(),
                    requires_escalation=custom_flag.get("requires_escalation", False),
                    escalation_notes=custom_flag.get("escalation_notes", "")
                )
                flagged_events.append(event)
                logger.warning(f"🚨 CUSTOM FLAG TRIGGERED: {custom_flag.get('name', 'Unknown')}")
        
        return flagged_events
    
    def check_compliance_violations(self, shift_data: ShiftData, guardrails: ClientGuardrails) -> List[Dict[str, Any]]:
        """
        3.2. Compliance and Restriction Monitoring (Adherence to Care Plan)
        Validates worker actions against client-specific care plan and restrictions
        """
        violations = []
        text_lower = shift_data.worker_notes.lower()
        
        logger.info("📋 Checking compliance violations...")
        
        # Check medication restrictions
        if any(med in text_lower for med in guardrails.medication_restrictions):
            violations.append({
                "type": "medication_violation",
                "description": "Medication restriction violation detected",
                "severity": "high",
                "details": f"Detected restricted medication activity",
                "recommended_action": "Review medication administration with supervisor"
            })
        
        # Check dietary restrictions
        for restriction in guardrails.dietary_restrictions:
            if restriction.lower() in text_lower:
                violations.append({
                    "type": "dietary_violation",
                    "description": f"Dietary restriction violation: {restriction}",
                    "severity": "medium",
                    "details": f"Detected restricted food item: {restriction}",
                    "recommended_action": "Verify dietary compliance with care plan"
                })
        
        # Check activity restrictions
        for restriction in guardrails.activity_restrictions:
            if restriction.lower() in text_lower:
                violations.append({
                    "type": "activity_violation",
                    "description": f"Activity restriction violation: {restriction}",
                    "severity": "medium",
                    "details": f"Detected restricted activity: {restriction}",
                    "recommended_action": "Review activity compliance with care plan"
                })
        
        # Check visitor restrictions
        for restriction in guardrails.visitor_restrictions:
            if restriction.lower() in text_lower:
                violations.append({
                    "type": "visitor_violation",
                    "description": f"Visitor restriction violation: {restriction}",
                    "severity": "low",
                    "details": f"Detected restricted visitor: {restriction}",
                    "recommended_action": "Verify visitor policy compliance"
                })
        
        logger.info(f"📋 Found {len(violations)} compliance violations")
        return violations
    
    def generate_shift_narrative(self, shift_data: ShiftData, guardrails: ClientGuardrails, flagged_events: List[FlaggedEvent]) -> str:
        """
        3.3. Dynamic Narrative Generation (Automated Shift Reporting)
        Generates coherent, human-readable narrative summary proportional to shift duration
        """
        logger.info("📝 Generating dynamic narrative...")
        
        try:
            # Prepare context for AI narrative generation
            narrative_context = {
                "shift_duration": shift_data.shift_duration_hours,
                "is_overnight": shift_data.is_overnight_shift,
                "worker_notes": shift_data.worker_notes,
                "completed_tasks": shift_data.completed_tasks,
                "services_provided": shift_data.services_provided,
                "flagged_events": [event.description for event in flagged_events],
                "client_name": guardrails.client_name,
                "narrative_requirements": guardrails.narrative_requirements
            }
            
            # Calculate narrative length based on shift duration
            if shift_data.is_overnight_shift:
                # Special format for overnight shifts
                narrative_prompt = self.create_overnight_narrative_prompt(narrative_context)
            else:
                # Proportional length for regular shifts
                narrative_prompt = self.create_proportional_narrative_prompt(narrative_context)
            
            # Generate narrative using Azure OpenAI
            narrative_response = self.azure_service.client.chat.completions.create(
                model=self.azure_service.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a healthcare documentation assistant that creates detailed, professional shift narratives for care workers."
                    },
                    {
                        "role": "user",
                        "content": narrative_prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            
            narrative = narrative_response.choices[0].message.content.strip()
            logger.info(f"📝 Generated narrative: {len(narrative)} characters")
            
            return narrative
            
        except Exception as e:
            logger.error(f"❌ Error generating narrative: {str(e)}")
            # Fallback narrative
            return self.generate_fallback_narrative(shift_data, flagged_events)
    
    def create_proportional_narrative_prompt(self, context: Dict[str, Any]) -> str:
        """Create narrative prompt with proportional length logic"""
        duration = context["shift_duration"]
        
        # Calculate target length based on shift duration
        # Rule: longer the day, longer the narrative
        base_length = 100
        length_per_hour = 50
        target_length = base_length + (duration * length_per_hour)
        
        # Cap the maximum length
        max_length = context["narrative_requirements"].get("maxLength", 500)
        target_length = min(target_length, max_length)
        
        prompt = f"""
Generate a comprehensive shift narrative for {context["client_name"]}.

SHIFT DETAILS:
- Duration: {duration} hours
- Services Provided: {', '.join(context["services_provided"])}
- Completed Tasks: {', '.join(context["completed_tasks"])}
- Worker Notes: {context["worker_notes"]}

FLAGGED EVENTS: {', '.join(context["flagged_events"]) if context["flagged_events"] else "None"}

REQUIREMENTS:
- Target Length: {target_length} characters
- Professional healthcare documentation style
- Include specific activities and client responses
- Mention any challenges or incidents
- Always end with "No challenges noted" if no issues occurred
- Make narrative proportional to shift length (longer shift = more detail)

Generate a detailed, engaging narrative that captures the essence of this {duration}-hour shift.
"""
        return prompt
    
    def create_overnight_narrative_prompt(self, context: Dict[str, Any]) -> str:
        """Create narrative prompt for overnight shifts"""
        prompt = f"""
Generate an overnight shift narrative for {context["client_name"]}.

OVERNIGHT SHIFT DETAILS:
- Duration: {context["shift_duration"]} hours (overnight)
- Services Provided: {', '.join(context["services_provided"])}
- Worker Notes: {context["worker_notes"]}
- Flagged Events: {', '.join(context["flagged_events"]) if context["flagged_events"] else "None"}

OVERNIGHT FORMAT REQUIREMENTS:
- Shorter, more concise format
- Focus on monitoring and safety checks
- Include sleep patterns and rest quality
- Mention any nighttime incidents or concerns
- Professional but brief documentation style
- Always end with "No challenges noted" if no issues occurred

Generate a concise overnight shift narrative.
"""
        return prompt
    
    def generate_fallback_narrative(self, shift_data: ShiftData, flagged_events: List[FlaggedEvent]) -> str:
        """Generate fallback narrative when AI generation fails"""
        duration = shift_data.shift_duration_hours
        services = ', '.join(shift_data.services_provided)
        
        if shift_data.is_overnight_shift:
            narrative = f"Overnight shift completed successfully. Duration: {duration} hours. Services provided: {services}. Client was monitored throughout the night with regular check-ins."
        else:
            narrative = f"Shift completed successfully. Duration: {duration} hours. Services provided: {services}. Activities included: {', '.join(shift_data.completed_tasks)}."
        
        if flagged_events:
            narrative += f" Note: {len(flagged_events)} events were flagged for review."
        else:
            narrative += " No challenges noted."
        
        return narrative
    
    def get_client_guardrails(self, client_id: str) -> ClientGuardrails:
        """
        3.4. Customizable Guardrails Framework (Client-Specific Logic)
        Retrieves client-specific guardrails and restrictions
        """
        if client_id in self.client_guardrails_db:
            return self.client_guardrails_db[client_id]
        
        # Return default guardrails if client-specific ones don't exist
        return self.create_default_guardrails(client_id)
    
    def create_default_guardrails(self, client_id: str) -> ClientGuardrails:
        """Create default guardrails for a client"""
        return ClientGuardrails(
            client_id=client_id,
            client_name=f"Client {client_id}",
            restricted_topics=["medication", "meal prep", "medical advice"],
            allowed_services=["personal care", "companionship", "light housekeeping"],
            medication_restrictions=["unauthorized medication", "overdose", "wrong medication"],
            dietary_restrictions=["nuts", "dairy", "gluten"],
            activity_restrictions=["heavy lifting", "unsupervised outings"],
            visitor_restrictions=["unauthorized visitors", "after hours visits"],
            special_instructions=[
                "Always explicitly state 'no challenges noted' when no issues occur",
                "Generate narratives proportional to shift length",
                "Avoid discussing restricted topics"
            ],
            narrative_requirements={
                "minLength": 50,
                "maxLength": 500,
                "overnightShiftFormat": True
            },
            custom_flags=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def add_client_guardrails(self, guardrails: ClientGuardrails):
        """Add or update client-specific guardrails"""
        guardrails.updated_at = datetime.now()
        self.client_guardrails_db[guardrails.client_id] = guardrails
        logger.info(f"✅ Updated guardrails for client: {guardrails.client_id}")
    
    def check_custom_flag(self, text: str, custom_flag: Dict[str, Any]) -> bool:
        """Check if a custom flag condition is met"""
        flag_type = custom_flag.get("type", "text_match")
        
        if flag_type == "text_match":
            pattern = custom_flag.get("pattern", "")
            return bool(re.search(pattern, text, re.IGNORECASE))
        elif flag_type == "keyword_list":
            keywords = custom_flag.get("keywords", [])
            return any(keyword.lower() in text for keyword in keywords)
        elif flag_type == "regex":
            pattern = custom_flag.get("pattern", "")
            return bool(re.search(pattern, text, re.IGNORECASE))
        
        return False
    
    def determine_severity(self, event_type: str) -> AlertSeverity:
        """Determine severity level for different event types"""
        severity_mapping = {
            "fall": AlertSeverity.CRITICAL,
            "fever": AlertSeverity.HIGH,
            "injury": AlertSeverity.HIGH,
            "medical_emergency": AlertSeverity.CRITICAL
        }
        return severity_mapping.get(event_type, AlertSeverity.MEDIUM)
    
    def calculate_confidence_score(self, flagged_events: List[FlaggedEvent], compliance_violations: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the analysis"""
        total_checks = len(self.safety_patterns) + len(self.compliance_patterns)
        successful_detections = len(flagged_events) + len(compliance_violations)
        
        # Higher confidence when we detect issues (more specific)
        if successful_detections > 0:
            return min(0.95, 0.7 + (successful_detections * 0.1))
        else:
            return 0.8  # High confidence in "no issues" detection
    
    def store_analysis_result(self, result: PrescreenerResult):
        """Store analysis result for tracking and reporting"""
        self.flagged_events_db[result.analysis_id] = result
        logger.info(f"📊 Stored analysis result: {result.analysis_id}")
    
    def get_analysis_history(self, client_id: str, days: int = 30) -> List[PrescreenerResult]:
        """Get analysis history for a client"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        results = []
        for result in self.flagged_events_db.values():
            if (result.client_id == client_id and 
                result.analysis_timestamp >= cutoff_date):
                results.append(result)
        
        return sorted(results, key=lambda x: x.analysis_timestamp, reverse=True)
    
    def generate_escalation_alert(self, flagged_event: FlaggedEvent) -> Dict[str, Any]:
        """Generate escalation alert for flagged events"""
        return {
            "alert_id": str(uuid.uuid4()),
            "event_id": flagged_event.event_id,
            "client_id": flagged_event.client_id,
            "severity": flagged_event.severity.value,
            "description": flagged_event.description,
            "timestamp": flagged_event.timestamp.isoformat(),
            "escalation_notes": flagged_event.escalation_notes,
            "priority": "HIGH" if flagged_event.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH] else "MEDIUM",
            "action_required": "Immediate supervisor review required",
            "escalation_path": "Supervisor → Manager → Director"
        }

# Export the main class
__all__ = ['AIPrescreenerCore', 'PrescreenerResult', 'FlaggedEvent', 'ClientGuardrails', 'ShiftData']
