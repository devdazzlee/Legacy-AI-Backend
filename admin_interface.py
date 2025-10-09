"""
Admin Interface for AI Prescreener System
=========================================

This module provides FastAPI endpoints for managing client-specific guardrails,
viewing flagged events, and administering the AI Prescreener system.

Author: Metaxoft AI Assistant
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import uuid
import logging

from ai_prescreener import AIPrescreenerCore, ClientGuardrails, ShiftData, FlaggedEvent
from database_schema import db_manager, ClientGuardrailsDB, FlaggedEventDB, PrescreenerResultDB
from alert_system import RealTimeAlertSystem, AlertRecipient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Prescreener Admin Interface",
    description="Administrative interface for managing AI Prescreener system",
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

# Initialize AI Prescreener and Alert System
ai_prescreener = None
alert_system = None

# Pydantic models for API requests/responses
class ClientGuardrailsRequest(BaseModel):
    client_id: str
    client_name: str
    restricted_topics: List[str] = []
    allowed_services: List[str] = []
    medication_restrictions: List[str] = []
    dietary_restrictions: List[str] = []
    activity_restrictions: List[str] = []
    visitor_restrictions: List[str] = []
    special_instructions: List[str] = []
    narrative_requirements: Dict[str, Any] = {
        "minLength": 50,
        "maxLength": 500,
        "overnightShiftFormat": True
    }
    custom_flags: List[Dict[str, Any]] = []
    is_active: bool = True

class ClientGuardrailsResponse(BaseModel):
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
    is_active: bool
    created_at: datetime
    updated_at: datetime

class ShiftAnalysisRequest(BaseModel):
    shift_id: str
    client_id: str
    worker_id: str
    shift_date: datetime
    shift_duration_hours: float
    is_overnight_shift: bool = False
    worker_notes: str
    completed_tasks: List[str] = []
    services_provided: List[str] = []
    additional_context: Dict[str, Any] = {}

class ShiftAnalysisResponse(BaseModel):
    analysis_id: str
    shift_id: str
    client_id: str
    flagged_events: List[Dict[str, Any]]
    compliance_violations: List[Dict[str, Any]]
    generated_narrative: str
    narrative_length: int
    processing_time_ms: int
    confidence_score: float
    requires_human_review: bool
    analysis_timestamp: datetime

class FlaggedEventResponse(BaseModel):
    event_id: str
    client_id: str
    event_type: str
    severity: str
    description: str
    original_text: str
    timestamp: datetime
    requires_escalation: bool
    escalation_notes: Optional[str] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None

class AlertRecipientRequest(BaseModel):
    user_id: str
    name: str
    email: str
    phone: Optional[str] = None
    role: str = "supervisor"
    escalation_level: int = 1
    is_active: bool = True

class SystemStatusResponse(BaseModel):
    system_status: str
    ai_prescreener_status: str
    alert_system_status: str
    database_status: str
    total_clients: int
    total_flagged_events: int
    pending_alerts: int
    last_analysis: Optional[datetime] = None

# Dependency to get AI Prescreener instance
def get_ai_prescreener():
    global ai_prescreener
    if ai_prescreener is None:
        # Initialize with Azure OpenAI service
        from main import azure_service
        ai_prescreener = AIPrescreenerCore(azure_service)
    return ai_prescreener

# Dependency to get Alert System instance
def get_alert_system():
    global alert_system
    if alert_system is None:
        alert_system = RealTimeAlertSystem(db_manager)
    return alert_system

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "AI Prescreener Admin Interface is running",
        "timestamp": datetime.utcnow().isoformat()
    }

# System status endpoint
@app.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Check database status
        db_status = "healthy"
        try:
            db_manager.get_session().close()
        except Exception:
            db_status = "unhealthy"
        
        # Get system statistics
        total_clients = len(db_manager.get_session().query(ClientGuardrailsDB).all())
        total_flagged_events = len(db_manager.get_session().query(FlaggedEventDB).all())
        pending_alerts = len(db_manager.get_pending_alerts())
        
        # Get last analysis
        last_analysis = None
        try:
            last_result = db_manager.get_session().query(PrescreenerResultDB).order_by(
                PrescreenerResultDB.analysis_timestamp.desc()
            ).first()
            if last_result:
                last_analysis = last_result.analysis_timestamp
        except Exception:
            pass
        
        return SystemStatusResponse(
            system_status="healthy",
            ai_prescreener_status="active" if ai_prescreener else "inactive",
            alert_system_status="active" if alert_system else "inactive",
            database_status=db_status,
            total_clients=total_clients,
            total_flagged_events=total_flagged_events,
            pending_alerts=pending_alerts,
            last_analysis=last_analysis
        )
    except Exception as e:
        logger.error(f"❌ Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

# Client Guardrails Management Endpoints
@app.post("/admin/guardrails", response_model=ClientGuardrailsResponse)
async def create_client_guardrails(
    request: ClientGuardrailsRequest,
    prescreener: AIPrescreenerCore = Depends(get_ai_prescreener)
):
    """Create or update client-specific guardrails"""
    try:
        logger.info(f"📝 Creating guardrails for client: {request.client_id}")
        
        # Convert request to database format
        guardrails_data = {
            "client_id": request.client_id,
            "client_name": request.client_name,
            "restricted_topics": request.restricted_topics,
            "allowed_services": request.allowed_services,
            "medication_restrictions": request.medication_restrictions,
            "dietary_restrictions": request.dietary_restrictions,
            "activity_restrictions": request.activity_restrictions,
            "visitor_restrictions": request.visitor_restrictions,
            "special_instructions": request.special_instructions,
            "narrative_requirements": request.narrative_requirements,
            "custom_flags": request.custom_flags,
            "is_active": request.is_active,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Add to database
        db_guardrails = db_manager.add_client_guardrails(guardrails_data)
        
        # Add to AI Prescreener
        client_guardrails = ClientGuardrails(
            client_id=request.client_id,
            client_name=request.client_name,
            restricted_topics=request.restricted_topics,
            allowed_services=request.allowed_services,
            medication_restrictions=request.medication_restrictions,
            dietary_restrictions=request.dietary_restrictions,
            activity_restrictions=request.activity_restrictions,
            visitor_restrictions=request.visitor_restrictions,
            special_instructions=request.special_instructions,
            narrative_requirements=request.narrative_requirements,
            custom_flags=request.custom_flags,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_active=request.is_active
        )
        
        prescreener.add_client_guardrails(client_guardrails)
        
        logger.info(f"✅ Successfully created guardrails for client: {request.client_id}")
        
        return ClientGuardrailsResponse(
            client_id=db_guardrails.client_id,
            client_name=db_guardrails.client_name,
            restricted_topics=db_guardrails.restricted_topics,
            allowed_services=db_guardrails.allowed_services,
            medication_restrictions=db_guardrails.medication_restrictions,
            dietary_restrictions=db_guardrails.dietary_restrictions,
            activity_restrictions=db_guardrails.activity_restrictions,
            visitor_restrictions=db_guardrails.visitor_restrictions,
            special_instructions=db_guardrails.special_instructions,
            narrative_requirements=db_guardrails.narrative_requirements,
            custom_flags=db_guardrails.custom_flags,
            is_active=db_guardrails.is_active,
            created_at=db_guardrails.created_at,
            updated_at=db_guardrails.updated_at
        )
        
    except Exception as e:
        logger.error(f"❌ Error creating guardrails: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating guardrails: {str(e)}")

@app.get("/admin/guardrails/{client_id}", response_model=ClientGuardrailsResponse)
async def get_client_guardrails(client_id: str):
    """Get client-specific guardrails"""
    try:
        guardrails = db_manager.get_client_guardrails(client_id)
        
        if not guardrails:
            raise HTTPException(status_code=404, detail=f"Guardrails not found for client: {client_id}")
        
        return ClientGuardrailsResponse(
            client_id=guardrails.client_id,
            client_name=guardrails.client_name,
            restricted_topics=guardrails.restricted_topics,
            allowed_services=guardrails.allowed_services,
            medication_restrictions=guardrails.medication_restrictions,
            dietary_restrictions=guardrails.dietary_restrictions,
            activity_restrictions=guardrails.activity_restrictions,
            visitor_restrictions=guardrails.visitor_restrictions,
            special_instructions=guardrails.special_instructions,
            narrative_requirements=guardrails.narrative_requirements,
            custom_flags=guardrails.custom_flags,
            is_active=guardrails.is_active,
            created_at=guardrails.created_at,
            updated_at=guardrails.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting guardrails: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting guardrails: {str(e)}")

@app.get("/admin/guardrails", response_model=List[ClientGuardrailsResponse])
async def list_all_guardrails():
    """List all client guardrails"""
    try:
        session = db_manager.get_session()
        guardrails_list = session.query(ClientGuardrailsDB).all()
        session.close()
        
        return [
            ClientGuardrailsResponse(
                client_id=g.client_id,
                client_name=g.client_name,
                restricted_topics=g.restricted_topics,
                allowed_services=g.allowed_services,
                medication_restrictions=g.medication_restrictions,
                dietary_restrictions=g.dietary_restrictions,
                activity_restrictions=g.activity_restrictions,
                visitor_restrictions=g.visitor_restrictions,
                special_instructions=g.special_instructions,
                narrative_requirements=g.narrative_requirements,
                custom_flags=g.custom_flags,
                is_active=g.is_active,
                created_at=g.created_at,
                updated_at=g.updated_at
            )
            for g in guardrails_list
        ]
        
    except Exception as e:
        logger.error(f"❌ Error listing guardrails: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing guardrails: {str(e)}")

# Shift Analysis Endpoints
@app.post("/analysis/shift", response_model=ShiftAnalysisResponse)
async def analyze_shift(
    request: ShiftAnalysisRequest,
    background_tasks: BackgroundTasks,
    prescreener: AIPrescreenerCore = Depends(get_ai_prescreener),
    alert_sys: RealTimeAlertSystem = Depends(get_alert_system)
):
    """Analyze a shift using AI Prescreener"""
    try:
        logger.info(f"🔍 Analyzing shift: {request.shift_id}")
        
        # Convert request to ShiftData
        shift_data = ShiftData(
            shift_id=request.shift_id,
            client_id=request.client_id,
            worker_id=request.worker_id,
            shift_date=request.shift_date,
            shift_duration_hours=request.shift_duration_hours,
            is_overnight_shift=request.is_overnight_shift,
            worker_notes=request.worker_notes,
            completed_tasks=request.completed_tasks,
            services_provided=request.services_provided,
            additional_context=request.additional_context
        )
        
        # Perform analysis
        result = prescreener.analyze_shift(shift_data)
        
        # Process flagged events in background
        if result.flagged_events:
            background_tasks.add_task(process_flagged_events_background, result.flagged_events, alert_sys)
        
        # Convert result to response format
        flagged_events_response = [
            {
                "event_id": event.event_id,
                "client_id": event.client_id,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "description": event.description,
                "original_text": event.original_text,
                "timestamp": event.timestamp,
                "requires_escalation": event.requires_escalation,
                "escalation_notes": event.escalation_notes,
                "resolved": event.resolved,
                "resolved_by": event.resolved_by,
                "resolved_at": event.resolved_at
            }
            for event in result.flagged_events
        ]
        
        # Store analysis result in database
        analysis_data = {
            "analysis_id": result.analysis_id,
            "shift_id": result.shift_id,
            "client_id": result.client_id,
            "worker_id": request.worker_id,
            "shift_date": request.shift_date,
            "shift_duration_hours": request.shift_duration_hours,
            "is_overnight_shift": request.is_overnight_shift,
            "flagged_events_count": len(result.flagged_events),
            "compliance_violations_count": len(result.compliance_violations),
            "generated_narrative": result.generated_narrative,
            "narrative_length": result.narrative_length,
            "processing_time_ms": result.processing_time_ms,
            "confidence_score": result.confidence_score,
            "requires_human_review": result.requires_human_review,
            "analysis_timestamp": result.analysis_timestamp
        }
        
        db_manager.add_analysis_result(analysis_data)
        
        logger.info(f"✅ Successfully analyzed shift: {request.shift_id}")
        
        return ShiftAnalysisResponse(
            analysis_id=result.analysis_id,
            shift_id=result.shift_id,
            client_id=result.client_id,
            flagged_events=flagged_events_response,
            compliance_violations=result.compliance_violations,
            generated_narrative=result.generated_narrative,
            narrative_length=result.narrative_length,
            processing_time_ms=result.processing_time_ms,
            confidence_score=result.confidence_score,
            requires_human_review=result.requires_human_review,
            analysis_timestamp=result.analysis_timestamp
        )
        
    except Exception as e:
        logger.error(f"❌ Error analyzing shift: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing shift: {str(e)}")

# Flagged Events Management Endpoints
@app.get("/events/flagged", response_model=List[FlaggedEventResponse])
async def get_flagged_events(client_id: Optional[str] = None, days: int = 30):
    """Get flagged events with optional filtering"""
    try:
        session = db_manager.get_session()
        
        query = session.query(FlaggedEventDB)
        
        if client_id:
            query = query.filter(FlaggedEventDB.client_id == client_id)
        
        # Filter by date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = query.filter(FlaggedEventDB.timestamp >= cutoff_date)
        
        events = query.order_by(FlaggedEventDB.timestamp.desc()).all()
        session.close()
        
        return [
            FlaggedEventResponse(
                event_id=event.event_id,
                client_id=event.client_id,
                event_type=event.event_type,
                severity=event.severity,
                description=event.description,
                original_text=event.original_text,
                timestamp=event.timestamp,
                requires_escalation=event.requires_escalation,
                escalation_notes=event.escalation_notes,
                resolved=event.resolved,
                resolved_by=event.resolved_by,
                resolved_at=event.resolved_at
            )
            for event in events
        ]
        
    except Exception as e:
        logger.error(f"❌ Error getting flagged events: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting flagged events: {str(e)}")

@app.post("/events/{event_id}/resolve")
async def resolve_flagged_event(event_id: str, resolved_by: str):
    """Mark a flagged event as resolved"""
    try:
        session = db_manager.get_session()
        event = session.query(FlaggedEventDB).filter(FlaggedEventDB.event_id == event_id).first()
        
        if not event:
            raise HTTPException(status_code=404, detail=f"Flagged event not found: {event_id}")
        
        event.resolved = True
        event.resolved_by = resolved_by
        event.resolved_at = datetime.utcnow()
        
        session.commit()
        session.close()
        
        logger.info(f"✅ Resolved flagged event: {event_id}")
        return {"success": True, "message": f"Event {event_id} resolved by {resolved_by}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error resolving event: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resolving event: {str(e)}")

# Alert Management Endpoints
@app.get("/alerts/pending")
async def get_pending_alerts(client_id: Optional[str] = None):
    """Get pending escalation alerts"""
    try:
        alerts = db_manager.get_pending_alerts(client_id)
        
        return [
            {
                "alert_id": alert.alert_id,
                "event_id": alert.event_id,
                "client_id": alert.client_id,
                "severity": alert.severity,
                "description": alert.description,
                "timestamp": alert.timestamp,
                "priority": alert.priority,
                "action_required": alert.action_required,
                "escalation_path": alert.escalation_path,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved
            }
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"❌ Error getting pending alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting pending alerts: {str(e)}")

@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, resolved_by: str):
    """Resolve an escalation alert"""
    try:
        success = db_manager.resolve_alert(alert_id, resolved_by)
        
        if success:
            return {"success": True, "message": f"Alert {alert_id} resolved by {resolved_by}"}
        else:
            raise HTTPException(status_code=404, detail=f"Alert not found: {alert_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error resolving alert: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resolving alert: {str(e)}")

# Alert Recipients Management
@app.post("/admin/recipients")
async def add_alert_recipient(request: AlertRecipientRequest):
    """Add alert recipient"""
    try:
        recipient = AlertRecipient(
            user_id=request.user_id,
            name=request.name,
            email=request.email,
            phone=request.phone,
            role=request.role,
            escalation_level=request.escalation_level,
            is_active=request.is_active
        )
        
        alert_system = get_alert_system()
        alert_system.alert_recipients[request.user_id] = recipient
        
        logger.info(f"✅ Added alert recipient: {request.user_id}")
        return {"success": True, "message": f"Recipient {request.user_id} added successfully"}
        
    except Exception as e:
        logger.error(f"❌ Error adding recipient: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding recipient: {str(e)}")

# Background task functions
async def process_flagged_events_background(flagged_events: List[FlaggedEvent], alert_system: RealTimeAlertSystem):
    """Process flagged events in background"""
    try:
        for event in flagged_events:
            await alert_system.process_flagged_event(event)
        logger.info(f"✅ Processed {len(flagged_events)} flagged events in background")
    except Exception as e:
        logger.error(f"❌ Error processing flagged events in background: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize systems on startup"""
    logger.info("🚀 Starting AI Prescreener Admin Interface")
    
    # Initialize AI Prescreener
    global ai_prescreener
    from main import azure_service
    ai_prescreener = AIPrescreenerCore(azure_service)
    
    # Initialize Alert System
    global alert_system
    alert_system = RealTimeAlertSystem(db_manager)
    
    # Start alert system
    await alert_system.start_alert_system()
    
    logger.info("✅ AI Prescreener Admin Interface started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down AI Prescreener Admin Interface")
    
    if alert_system:
        await alert_system.stop_alert_system()
    
    logger.info("✅ AI Prescreener Admin Interface shut down successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
