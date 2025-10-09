"""
Real-Time Alert System for AI Prescreener
==========================================

This module implements the real-time alert system for flagged events,
including escalation management and notification delivery.

Author: Metaxoft AI Assistant
Version: 1.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    PUSH_NOTIFICATION = "push_notification"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"

class AlertStatus(Enum):
    """Alert status tracking"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FAILED = "failed"

@dataclass
class AlertRecipient:
    """Alert recipient information"""
    user_id: str
    name: str
    email: str
    phone: Optional[str] = None
    role: str = "supervisor"
    escalation_level: int = 1
    is_active: bool = True

@dataclass
class AlertNotification:
    """Alert notification data"""
    notification_id: str
    alert_id: str
    recipient_id: str
    channel: AlertChannel
    message: str
    subject: str
    priority: str
    status: AlertStatus
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    error_message: Optional[str] = None

class RealTimeAlertSystem:
    """
    Real-Time Alert System for AI Prescreener
    Handles immediate escalation and notification delivery for flagged events
    """
    
    def __init__(self, db_manager, config: Dict[str, Any] = None):
        self.db_manager = db_manager
        self.config = config or self.get_default_config()
        self.alert_recipients = {}  # In production, this would be loaded from database
        self.escalation_rules = self.initialize_escalation_rules()
        self.notification_queue = asyncio.Queue()
        self.is_running = False
        
        # Initialize notification handlers
        self.notification_handlers = {
            AlertChannel.EMAIL: self.send_email_notification,
            AlertChannel.SMS: self.send_sms_notification,
            AlertChannel.PUSH_NOTIFICATION: self.send_push_notification,
            AlertChannel.WEBHOOK: self.send_webhook_notification,
            AlertChannel.DASHBOARD: self.send_dashboard_notification
        }
        
        self.initialize_default_recipients()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for alert system"""
        return {
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "alerts@company.com",
                "password": "your_password",
                "from_address": "AI Prescreener <alerts@company.com>"
            },
            "sms": {
                "provider": "twilio",
                "account_sid": "your_account_sid",
                "auth_token": "your_auth_token",
                "from_number": "+1234567890"
            },
            "webhook": {
                "url": "https://your-webhook-url.com/alerts",
                "timeout": 30,
                "retry_attempts": 3
            },
            "escalation": {
                "max_escalation_levels": 3,
                "escalation_timeout_minutes": 15,
                "retry_interval_minutes": 5
            }
        }
    
    def initialize_escalation_rules(self) -> Dict[str, Any]:
        """Initialize escalation rules based on severity"""
        return {
            "critical": {
                "immediate_escalation": True,
                "escalation_levels": [1, 2, 3],
                "timeout_minutes": 5,
                "channels": [AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.PUSH_NOTIFICATION]
            },
            "high": {
                "immediate_escalation": True,
                "escalation_levels": [1, 2],
                "timeout_minutes": 10,
                "channels": [AlertChannel.EMAIL, AlertChannel.PUSH_NOTIFICATION]
            },
            "medium": {
                "immediate_escalation": False,
                "escalation_levels": [1],
                "timeout_minutes": 30,
                "channels": [AlertChannel.EMAIL, AlertChannel.DASHBOARD]
            },
            "low": {
                "immediate_escalation": False,
                "escalation_levels": [1],
                "timeout_minutes": 60,
                "channels": [AlertChannel.DASHBOARD]
            }
        }
    
    def initialize_default_recipients(self):
        """Initialize default alert recipients"""
        default_recipients = [
            AlertRecipient(
                user_id="supervisor_001",
                name="John Supervisor",
                email="supervisor@company.com",
                phone="+1234567890",
                role="supervisor",
                escalation_level=1
            ),
            AlertRecipient(
                user_id="manager_001",
                name="Jane Manager",
                email="manager@company.com",
                phone="+1234567891",
                role="manager",
                escalation_level=2
            ),
            AlertRecipient(
                user_id="director_001",
                name="Bob Director",
                email="director@company.com",
                phone="+1234567892",
                role="director",
                escalation_level=3
            )
        ]
        
        for recipient in default_recipients:
            self.alert_recipients[recipient.user_id] = recipient
    
    async def process_flagged_event(self, flagged_event) -> Dict[str, Any]:
        """
        Process a flagged event and generate appropriate alerts
        This is the main entry point for the alert system
        """
        logger.info(f"🚨 Processing flagged event: {flagged_event.event_id}")
        
        try:
            # Determine escalation strategy based on severity
            escalation_rules = self.escalation_rules.get(flagged_event.severity.value, {})
            
            if not escalation_rules:
                logger.warning(f"⚠️ No escalation rules found for severity: {flagged_event.severity.value}")
                return {"success": False, "error": "No escalation rules found"}
            
            # Generate escalation alert
            escalation_alert = self.generate_escalation_alert(flagged_event, escalation_rules)
            
            # Store alert in database
            alert_data = {
                "alert_id": escalation_alert["alert_id"],
                "event_id": flagged_event.event_id,
                "client_id": flagged_event.client_id,
                "severity": flagged_event.severity.value,
                "description": flagged_event.description,
                "timestamp": flagged_event.timestamp,
                "escalation_notes": flagged_event.escalation_notes,
                "priority": escalation_alert["priority"],
                "action_required": escalation_alert["action_required"],
                "escalation_path": escalation_alert["escalation_path"],
                "acknowledged": False,
                "resolved": False
            }
            
            self.db_manager.add_escalation_alert(alert_data)
            
            # Process immediate escalation if required
            if escalation_rules.get("immediate_escalation", False):
                await self.process_immediate_escalation(escalation_alert, escalation_rules)
            
            # Schedule delayed escalation if needed
            if escalation_rules.get("escalation_levels", []):
                await self.schedule_escalation(escalation_alert, escalation_rules)
            
            logger.info(f"✅ Successfully processed flagged event: {flagged_event.event_id}")
            return {"success": True, "alert_id": escalation_alert["alert_id"]}
            
        except Exception as e:
            logger.error(f"❌ Error processing flagged event: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def generate_escalation_alert(self, flagged_event, escalation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Generate escalation alert data"""
        priority = "HIGH" if flagged_event.severity.value in ["critical", "high"] else "MEDIUM"
        
        return {
            "alert_id": str(uuid.uuid4()),
            "event_id": flagged_event.event_id,
            "client_id": flagged_event.client_id,
            "severity": flagged_event.severity.value,
            "description": flagged_event.description,
            "timestamp": flagged_event.timestamp.isoformat(),
            "escalation_notes": flagged_event.escalation_notes,
            "priority": priority,
            "action_required": "Immediate supervisor review required",
            "escalation_path": "Supervisor → Manager → Director"
        }
    
    async def process_immediate_escalation(self, escalation_alert: Dict[str, Any], escalation_rules: Dict[str, Any]):
        """Process immediate escalation for critical alerts"""
        logger.info(f"🚨 Processing immediate escalation for alert: {escalation_alert['alert_id']}")
        
        # Get recipients for immediate escalation
        immediate_recipients = self.get_recipients_for_level(1)
        
        # Send notifications through all configured channels
        channels = escalation_rules.get("channels", [AlertChannel.EMAIL])
        
        for recipient in immediate_recipients:
            for channel in channels:
                await self.send_notification(escalation_alert, recipient, channel)
    
    async def schedule_escalation(self, escalation_alert: Dict[str, Any], escalation_rules: Dict[str, Any]):
        """Schedule delayed escalation for multi-level alerts"""
        escalation_levels = escalation_rules.get("escalation_levels", [])
        timeout_minutes = escalation_rules.get("timeout_minutes", 15)
        
        for level in escalation_levels[1:]:  # Skip level 1 (immediate)
            delay_seconds = timeout_minutes * 60 * (level - 1)
            
            # Schedule escalation task
            asyncio.create_task(
                self.delayed_escalation(escalation_alert, level, delay_seconds)
            )
            
            logger.info(f"⏰ Scheduled escalation level {level} for alert {escalation_alert['alert_id']} in {delay_seconds} seconds")
    
    async def delayed_escalation(self, escalation_alert: Dict[str, Any], level: int, delay_seconds: int):
        """Execute delayed escalation after timeout"""
        await asyncio.sleep(delay_seconds)
        
        # Check if alert is already resolved
        if self.is_alert_resolved(escalation_alert["alert_id"]):
            logger.info(f"✅ Alert {escalation_alert['alert_id']} already resolved, skipping escalation level {level}")
            return
        
        logger.info(f"🚨 Executing delayed escalation level {level} for alert: {escalation_alert['alert_id']}")
        
        # Get recipients for this escalation level
        recipients = self.get_recipients_for_level(level)
        
        # Send notifications
        for recipient in recipients:
            await self.send_notification(escalation_alert, recipient, AlertChannel.EMAIL)
    
    def get_recipients_for_level(self, escalation_level: int) -> List[AlertRecipient]:
        """Get recipients for a specific escalation level"""
        recipients = []
        
        for recipient in self.alert_recipients.values():
            if recipient.escalation_level == escalation_level and recipient.is_active:
                recipients.append(recipient)
        
        return recipients
    
    async def send_notification(self, escalation_alert: Dict[str, Any], recipient: AlertRecipient, channel: AlertChannel):
        """Send notification through specified channel"""
        try:
            # Create notification message
            message = self.create_notification_message(escalation_alert, recipient)
            subject = self.create_notification_subject(escalation_alert)
            
            # Create notification record
            notification = AlertNotification(
                notification_id=str(uuid.uuid4()),
                alert_id=escalation_alert["alert_id"],
                recipient_id=recipient.user_id,
                channel=channel,
                message=message,
                subject=subject,
                priority=escalation_alert["priority"],
                status=AlertStatus.PENDING
            )
            
            # Send notification
            handler = self.notification_handlers.get(channel)
            if handler:
                success = await handler(notification, recipient)
                
                if success:
                    notification.status = AlertStatus.SENT
                    notification.sent_at = datetime.utcnow()
                    logger.info(f"✅ Notification sent via {channel.value} to {recipient.name}")
                else:
                    notification.status = AlertStatus.FAILED
                    logger.error(f"❌ Failed to send notification via {channel.value} to {recipient.name}")
            else:
                logger.error(f"❌ No handler found for channel: {channel.value}")
                
        except Exception as e:
            logger.error(f"❌ Error sending notification: {str(e)}")
    
    def create_notification_message(self, escalation_alert: Dict[str, Any], recipient: AlertRecipient) -> str:
        """Create notification message content"""
        return f"""
🚨 AI PRESCREENER ALERT 🚨

Priority: {escalation_alert['priority']}
Severity: {escalation_alert['severity'].upper()}
Client ID: {escalation_alert['client_id']}
Event ID: {escalation_alert['event_id']}

Description: {escalation_alert['description']}

Action Required: {escalation_alert['action_required']}
Escalation Path: {escalation_alert['escalation_path']}

Timestamp: {escalation_alert['timestamp']}

This alert was automatically generated by the AI Prescreener system.
Please review and take appropriate action immediately.

---
AI Prescreener System
Automated Healthcare Monitoring
"""
    
    def create_notification_subject(self, escalation_alert: Dict[str, Any]) -> str:
        """Create notification subject line"""
        severity = escalation_alert['severity'].upper()
        priority = escalation_alert['priority']
        client_id = escalation_alert['client_id']
        
        return f"🚨 {priority} ALERT - {severity} - Client {client_id} - AI Prescreener"
    
    async def send_email_notification(self, notification: AlertNotification, recipient: AlertRecipient) -> bool:
        """Send email notification"""
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['from_address']
            msg['To'] = recipient.email
            msg['Subject'] = notification.subject
            
            # Add body
            msg.attach(MIMEText(notification.message, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config['email']['smtp_server'], self.config['email']['smtp_port'])
            server.starttls()
            server.login(self.config['email']['username'], self.config['email']['password'])
            
            text = msg.as_string()
            server.sendmail(self.config['email']['from_address'], recipient.email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Email notification failed: {str(e)}")
            return False
    
    async def send_sms_notification(self, notification: AlertNotification, recipient: AlertRecipient) -> bool:
        """Send SMS notification (placeholder - implement with actual SMS provider)"""
        try:
            # Placeholder for SMS implementation
            # In production, integrate with Twilio, AWS SNS, or similar service
            logger.info(f"📱 SMS notification sent to {recipient.phone}: {notification.message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ SMS notification failed: {str(e)}")
            return False
    
    async def send_push_notification(self, notification: AlertNotification, recipient: AlertRecipient) -> bool:
        """Send push notification (placeholder - implement with actual push service)"""
        try:
            # Placeholder for push notification implementation
            # In production, integrate with Firebase, OneSignal, or similar service
            logger.info(f"📲 Push notification sent to {recipient.name}: {notification.message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Push notification failed: {str(e)}")
            return False
    
    async def send_webhook_notification(self, notification: AlertNotification, recipient: AlertRecipient) -> bool:
        """Send webhook notification"""
        try:
            import aiohttp
            
            webhook_data = {
                "alert_id": notification.alert_id,
                "recipient_id": recipient.user_id,
                "message": notification.message,
                "subject": notification.subject,
                "priority": notification.priority,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config['webhook']['url'],
                    json=webhook_data,
                    timeout=self.config['webhook']['timeout']
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.error(f"❌ Webhook notification failed with status: {response.status}")
                        return False
            
        except Exception as e:
            logger.error(f"❌ Webhook notification failed: {str(e)}")
            return False
    
    async def send_dashboard_notification(self, notification: AlertNotification, recipient: AlertRecipient) -> bool:
        """Send dashboard notification (store in database for dashboard display)"""
        try:
            # Store notification in database for dashboard display
            # This would typically involve updating a notifications table
            logger.info(f"📊 Dashboard notification stored for {recipient.name}: {notification.message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dashboard notification failed: {str(e)}")
            return False
    
    def is_alert_resolved(self, alert_id: str) -> bool:
        """Check if alert is already resolved"""
        # In production, this would query the database
        # For now, return False (not resolved)
        return False
    
    async def start_alert_system(self):
        """Start the alert system background tasks"""
        if self.is_running:
            logger.warning("⚠️ Alert system is already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting AI Prescreener Alert System")
        
        # Start background tasks
        asyncio.create_task(self.process_notification_queue())
        asyncio.create_task(self.monitor_pending_alerts())
    
    async def stop_alert_system(self):
        """Stop the alert system"""
        self.is_running = False
        logger.info("🛑 Stopping AI Prescreener Alert System")
    
    async def process_notification_queue(self):
        """Process notification queue in background"""
        while self.is_running:
            try:
                # Process queued notifications
                await asyncio.sleep(1)  # Check every second
            except Exception as e:
                logger.error(f"❌ Error processing notification queue: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def monitor_pending_alerts(self):
        """Monitor pending alerts for escalation"""
        while self.is_running:
            try:
                # Check for pending alerts that need escalation
                pending_alerts = self.db_manager.get_pending_alerts()
                
                for alert in pending_alerts:
                    # Check if alert needs escalation
                    if self.should_escalate_alert(alert):
                        await self.escalate_alert(alert)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"❌ Error monitoring pending alerts: {str(e)}")
                await asyncio.sleep(60)
    
    def should_escalate_alert(self, alert) -> bool:
        """Determine if alert should be escalated"""
        # Check if alert is past escalation timeout
        escalation_timeout = timedelta(minutes=self.config['escalation']['escalation_timeout_minutes'])
        return datetime.utcnow() - alert.timestamp > escalation_timeout
    
    async def escalate_alert(self, alert):
        """Escalate an alert to the next level"""
        logger.info(f"🚨 Escalating alert: {alert.alert_id}")
        # Implementation for escalating alerts
        pass

# Export the main class
__all__ = ['RealTimeAlertSystem', 'AlertRecipient', 'AlertNotification', 'AlertChannel', 'AlertStatus']
