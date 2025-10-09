"""
AI Prescreener System - Integration with Existing Mobile App
===========================================================

This module provides integration between the AI Prescreener system and the existing
TimesheetsMobile application, enabling real-time analysis of care worker shift data.

Author: Metaxoft AI Assistant
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIPrescreenerMobileIntegration:
    """
    Integration layer between AI Prescreener and TimesheetsMobile app
    Provides seamless integration for real-time shift analysis
    """
    
    def __init__(self, base_url: str = "http://192.168.18.64:8001"):
        self.base_url = base_url
        self.session = None
        
    async def initialize(self):
        """Initialize the integration"""
        try:
            import aiohttp
            self.session = aiohttp.ClientSession()
            logger.info("✅ AI Prescreener Mobile Integration initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize integration: {str(e)}")
            raise
    
    async def close(self):
        """Close the integration"""
        if self.session:
            await self.session.close()
    
    async def analyze_shift_data(self, shift_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze shift data using AI Prescreener
        
        Args:
            shift_data: Dictionary containing shift information
            
        Returns:
            Analysis result with flagged events, compliance violations, and narrative
        """
        try:
            logger.info(f"🔍 Analyzing shift data for client: {shift_data.get('client_id')}")
            
            # Prepare analysis request
            analysis_request = {
                "shift_id": shift_data.get("shift_id", f"mobile_shift_{datetime.now().timestamp()}"),
                "client_id": shift_data.get("client_id"),
                "worker_id": shift_data.get("worker_id"),
                "shift_date": shift_data.get("shift_date", datetime.now().isoformat()),
                "shift_duration_hours": shift_data.get("shift_duration_hours", 8.0),
                "is_overnight_shift": shift_data.get("is_overnight_shift", False),
                "worker_notes": shift_data.get("worker_notes", ""),
                "completed_tasks": shift_data.get("completed_tasks", []),
                "services_provided": shift_data.get("services_provided", []),
                "additional_context": shift_data.get("additional_context", {})
            }
            
            # Send request to AI Prescreener
            async with self.session.post(
                f"{self.base_url}/analysis/shift",
                json=analysis_request,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Shift analysis completed: {result.get('analysis_id')}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Analysis failed: {response.status} - {error_text}")
                    return {"error": f"Analysis failed: {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Error analyzing shift data: {str(e)}")
            return {"error": str(e)}
    
    async def get_client_guardrails(self, client_id: str) -> Dict[str, Any]:
        """
        Get client-specific guardrails
        
        Args:
            client_id: Client identifier
            
        Returns:
            Client guardrails configuration
        """
        try:
            async with self.session.get(f"{self.base_url}/admin/guardrails/{client_id}") as response:
                if response.status == 200:
                    guardrails = await response.json()
                    logger.info(f"✅ Retrieved guardrails for client: {client_id}")
                    return guardrails
                else:
                    logger.warning(f"⚠️ No guardrails found for client: {client_id}")
                    return {}
                    
        except Exception as e:
            logger.error(f"❌ Error getting guardrails: {str(e)}")
            return {}
    
    async def create_client_guardrails(self, guardrails_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create or update client guardrails
        
        Args:
            guardrails_data: Guardrails configuration
            
        Returns:
            Created guardrails data
        """
        try:
            async with self.session.post(
                f"{self.base_url}/admin/guardrails",
                json=guardrails_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Created guardrails for client: {guardrails_data.get('client_id')}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Failed to create guardrails: {response.status} - {error_text}")
                    return {"error": f"Failed to create guardrails: {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Error creating guardrails: {str(e)}")
            return {"error": str(e)}
    
    async def get_flagged_events(self, client_id: Optional[str] = None, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get flagged events for a client
        
        Args:
            client_id: Optional client filter
            days: Number of days to look back
            
        Returns:
            List of flagged events
        """
        try:
            url = f"{self.base_url}/events/flagged?days={days}"
            if client_id:
                url += f"&client_id={client_id}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    events = await response.json()
                    logger.info(f"✅ Retrieved {len(events)} flagged events")
                    return events
                else:
                    logger.error(f"❌ Failed to get flagged events: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error getting flagged events: {str(e)}")
            return []
    
    async def resolve_flagged_event(self, event_id: str, resolved_by: str) -> Dict[str, Any]:
        """
        Resolve a flagged event
        
        Args:
            event_id: Event identifier
            resolved_by: User who resolved the event
            
        Returns:
            Resolution result
        """
        try:
            async with self.session.post(
                f"{self.base_url}/events/{event_id}/resolve",
                json={"resolved_by": resolved_by},
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Resolved flagged event: {event_id}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Failed to resolve event: {response.status} - {error_text}")
                    return {"error": f"Failed to resolve event: {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Error resolving event: {str(e)}")
            return {"error": str(e)}
    
    async def get_pending_alerts(self, client_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get pending escalation alerts
        
        Args:
            client_id: Optional client filter
            
        Returns:
            List of pending alerts
        """
        try:
            url = f"{self.base_url}/alerts/pending"
            if client_id:
                url += f"?client_id={client_id}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    alerts = await response.json()
                    logger.info(f"✅ Retrieved {len(alerts)} pending alerts")
                    return alerts
                else:
                    logger.error(f"❌ Failed to get pending alerts: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error getting pending alerts: {str(e)}")
            return []
    
    async def resolve_alert(self, alert_id: str, resolved_by: str) -> Dict[str, Any]:
        """
        Resolve an escalation alert
        
        Args:
            alert_id: Alert identifier
            resolved_by: User who resolved the alert
            
        Returns:
            Resolution result
        """
        try:
            async with self.session.post(
                f"{self.base_url}/alerts/{alert_id}/resolve",
                json={"resolved_by": resolved_by},
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Resolved alert: {alert_id}")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Failed to resolve alert: {response.status} - {error_text}")
                    return {"error": f"Failed to resolve alert: {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Error resolving alert: {str(e)}")
            return {"error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get AI Prescreener system status
        
        Returns:
            System status information
        """
        try:
            async with self.session.get(f"{self.base_url}/system/status") as response:
                if response.status == 200:
                    status = await response.json()
                    logger.info("✅ Retrieved system status")
                    return status
                else:
                    logger.error(f"❌ Failed to get system status: {response.status}")
                    return {"error": f"Failed to get system status: {response.status}"}
                    
        except Exception as e:
            logger.error(f"❌ Error getting system status: {str(e)}")
            return {"error": str(e)}

# Mobile App Integration Service
class MobileAIPrescreenerService:
    """
    Service class for integrating AI Prescreener with TimesheetsMobile app
    Provides simplified interface for mobile app usage
    """
    
    def __init__(self, base_url: str = "http://192.168.18.64:8001"):
        self.integration = AIPrescreenerMobileIntegration(base_url)
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the service"""
        if not self.is_initialized:
            await self.integration.initialize()
            self.is_initialized = True
    
    async def close(self):
        """Close the service"""
        if self.is_initialized:
            await self.integration.close()
            self.is_initialized = False
    
    async def analyze_progress_log(self, progress_log_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze progress log data from mobile app
        
        Args:
            progress_log_data: Progress log data from mobile app
            
        Returns:
            Analysis result with suggestions and validation
        """
        try:
            await self.initialize()
            
            # Convert progress log data to shift data format
            shift_data = {
                "client_id": progress_log_data.get("client_id"),
                "worker_id": progress_log_data.get("worker_id"),
                "shift_date": progress_log_data.get("date", datetime.now().isoformat()),
                "shift_duration_hours": progress_log_data.get("duration_hours", 8.0),
                "is_overnight_shift": progress_log_data.get("is_overnight", False),
                "worker_notes": progress_log_data.get("notes", ""),
                "completed_tasks": progress_log_data.get("completed_tasks", []),
                "services_provided": progress_log_data.get("services", []),
                "additional_context": {
                    "section": progress_log_data.get("section", "Progress Log"),
                    "current_step": progress_log_data.get("current_step", 0)
                }
            }
            
            # Perform analysis
            result = await self.integration.analyze_shift_data(shift_data)
            
            if "error" in result:
                return result
            
            # Format result for mobile app
            mobile_result = {
                "success": True,
                "analysis_id": result.get("analysis_id"),
                "flagged_events": result.get("flagged_events", []),
                "compliance_violations": result.get("compliance_violations", []),
                "generated_narrative": result.get("generated_narrative", ""),
                "requires_human_review": result.get("requires_human_review", False),
                "confidence_score": result.get("confidence_score", 0.0),
                "suggestions": self._generate_suggestions(result),
                "validation_status": self._get_validation_status(result)
            }
            
            return mobile_result
            
        except Exception as e:
            logger.error(f"❌ Error analyzing progress log: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _generate_suggestions(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate suggestions based on analysis result"""
        suggestions = []
        
        # Check for flagged events
        flagged_events = analysis_result.get("flagged_events", [])
        if flagged_events:
            suggestions.append(f"⚠️ {len(flagged_events)} safety events detected - review required")
        
        # Check for compliance violations
        violations = analysis_result.get("compliance_violations", [])
        if violations:
            suggestions.append(f"📋 {len(violations)} compliance issues found - verify care plan adherence")
        
        # Check narrative quality
        narrative_length = analysis_result.get("narrative_length", 0)
        if narrative_length < 50:
            suggestions.append("📝 Consider adding more detail to your notes")
        
        # Check confidence score
        confidence = analysis_result.get("confidence_score", 0.0)
        if confidence < 0.7:
            suggestions.append("🤔 Analysis confidence is low - consider reviewing your input")
        
        return suggestions
    
    def _get_validation_status(self, analysis_result: Dict[str, Any]) -> str:
        """Get validation status based on analysis result"""
        flagged_events = analysis_result.get("flagged_events", [])
        violations = analysis_result.get("compliance_violations", [])
        
        if flagged_events or violations:
            return "requires_review"
        elif analysis_result.get("confidence_score", 0.0) < 0.7:
            return "needs_improvement"
        else:
            return "approved"
    
    async def get_client_safety_summary(self, client_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Get safety summary for a client
        
        Args:
            client_id: Client identifier
            days: Number of days to analyze
            
        Returns:
            Safety summary with statistics and trends
        """
        try:
            await self.initialize()
            
            # Get flagged events
            events = await self.integration.get_flagged_events(client_id, days)
            
            # Get pending alerts
            alerts = await self.integration.get_pending_alerts(client_id)
            
            # Calculate statistics
            total_events = len(events)
            critical_events = len([e for e in events if e.get("severity") == "critical"])
            high_events = len([e for e in events if e.get("severity") == "high"])
            resolved_events = len([e for e in events if e.get("resolved", False)])
            pending_alerts = len(alerts)
            
            summary = {
                "client_id": client_id,
                "period_days": days,
                "total_events": total_events,
                "critical_events": critical_events,
                "high_events": high_events,
                "resolved_events": resolved_events,
                "pending_alerts": pending_alerts,
                "safety_score": self._calculate_safety_score(total_events, critical_events, days),
                "recent_events": events[:5],  # Last 5 events
                "trend": self._calculate_trend(events)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting safety summary: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_safety_score(self, total_events: int, critical_events: int, days: int) -> float:
        """Calculate safety score (0-100)"""
        if total_events == 0:
            return 100.0
        
        # Base score
        base_score = 100.0
        
        # Deduct points for events
        event_penalty = (total_events * 5) + (critical_events * 15)
        
        # Normalize by days
        daily_penalty = event_penalty / days
        
        score = max(0.0, base_score - daily_penalty)
        return round(score, 1)
    
    def _calculate_trend(self, events: List[Dict[str, Any]]) -> str:
        """Calculate trend based on recent events"""
        if len(events) < 2:
            return "insufficient_data"
        
        # Simple trend calculation based on recent events
        recent_events = events[:3]
        older_events = events[3:6] if len(events) > 3 else []
        
        if len(recent_events) > len(older_events):
            return "increasing"
        elif len(recent_events) < len(older_events):
            return "decreasing"
        else:
            return "stable"

# Export classes
__all__ = ['AIPrescreenerMobileIntegration', 'MobileAIPrescreenerService']

