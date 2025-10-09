"""
Database Schema for AI Prescreener System
=========================================

This module defines the database schema for storing client guardrails,
incident tracking, and analysis results.

Author: Metaxoft AI Assistant
Version: 1.0.0
"""

from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class ClientGuardrailsDB(Base):
    """Database model for client-specific guardrails"""
    __tablename__ = 'client_guardrails'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(50), unique=True, nullable=False, index=True)
    client_name = Column(String(100), nullable=False)
    restricted_topics = Column(JSON, nullable=False, default=list)
    allowed_services = Column(JSON, nullable=False, default=list)
    medication_restrictions = Column(JSON, nullable=False, default=list)
    dietary_restrictions = Column(JSON, nullable=False, default=list)
    activity_restrictions = Column(JSON, nullable=False, default=list)
    visitor_restrictions = Column(JSON, nullable=False, default=list)
    special_instructions = Column(JSON, nullable=False, default=list)
    narrative_requirements = Column(JSON, nullable=False, default=dict)
    custom_flags = Column(JSON, nullable=False, default=list)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    flagged_events = relationship("FlaggedEventDB", back_populates="client_guardrails")
    analysis_results = relationship("PrescreenerResultDB", back_populates="client_guardrails")

class FlaggedEventDB(Base):
    """Database model for flagged events"""
    __tablename__ = 'flagged_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(50), unique=True, nullable=False, index=True)
    client_id = Column(String(50), ForeignKey('client_guardrails.client_id'), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    description = Column(Text, nullable=False)
    original_text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    requires_escalation = Column(Boolean, default=True, nullable=False)
    escalation_notes = Column(Text, nullable=True)
    resolved = Column(Boolean, default=False, nullable=False)
    resolved_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Relationships
    client_guardrails = relationship("ClientGuardrailsDB", back_populates="flagged_events")

class PrescreenerResultDB(Base):
    """Database model for AI Prescreener analysis results"""
    __tablename__ = 'prescreener_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(50), unique=True, nullable=False, index=True)
    shift_id = Column(String(50), nullable=False, index=True)
    client_id = Column(String(50), ForeignKey('client_guardrails.client_id'), nullable=False, index=True)
    worker_id = Column(String(50), nullable=False, index=True)
    shift_date = Column(DateTime, nullable=False, index=True)
    shift_duration_hours = Column(Float, nullable=False)
    is_overnight_shift = Column(Boolean, default=False, nullable=False)
    flagged_events_count = Column(Integer, default=0, nullable=False)
    compliance_violations_count = Column(Integer, default=0, nullable=False)
    generated_narrative = Column(Text, nullable=False)
    narrative_length = Column(Integer, nullable=False)
    processing_time_ms = Column(Integer, nullable=False)
    confidence_score = Column(Float, nullable=False)
    requires_human_review = Column(Boolean, default=False, nullable=False)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    client_guardrails = relationship("ClientGuardrailsDB", back_populates="analysis_results")

class EscalationAlertDB(Base):
    """Database model for escalation alerts"""
    __tablename__ = 'escalation_alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(50), unique=True, nullable=False, index=True)
    event_id = Column(String(50), ForeignKey('flagged_events.event_id'), nullable=False, index=True)
    client_id = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False)
    description = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    escalation_notes = Column(Text, nullable=True)
    priority = Column(String(20), nullable=False)
    action_required = Column(Text, nullable=False)
    escalation_path = Column(Text, nullable=False)
    acknowledged = Column(Boolean, default=False, nullable=False)
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved = Column(Boolean, default=False, nullable=False)
    resolved_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    
    # Relationships
    flagged_event = relationship("FlaggedEventDB")

class DatabaseManager:
    """Database manager for AI Prescreener system"""
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            # Use PostgreSQL database from environment or default
            import os
            database_url = os.getenv(
                "DATABASE_URL", 
                "postgresql://neondb_owner:npg_vCLPn9zdK2uY@ep-old-art-adro7qy2-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
            )
        
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        print("✅ Database tables created successfully")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def add_client_guardrails(self, guardrails_data: dict):
        """Add client guardrails to database"""
        session = self.get_session()
        try:
            guardrails = ClientGuardrailsDB(**guardrails_data)
            session.add(guardrails)
            session.commit()
            print(f"✅ Added guardrails for client: {guardrails_data['client_id']}")
            return guardrails
        except Exception as e:
            session.rollback()
            print(f"❌ Error adding guardrails: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_client_guardrails(self, client_id: str):
        """Get client guardrails from database"""
        session = self.get_session()
        try:
            guardrails = session.query(ClientGuardrailsDB).filter(
                ClientGuardrailsDB.client_id == client_id
            ).first()
            return guardrails
        except Exception as e:
            print(f"❌ Error getting guardrails: {str(e)}")
            return None
        finally:
            session.close()
    
    def add_flagged_event(self, event_data: dict):
        """Add flagged event to database"""
        session = self.get_session()
        try:
            event = FlaggedEventDB(**event_data)
            session.add(event)
            session.commit()
            print(f"✅ Added flagged event: {event_data['event_id']}")
            return event
        except Exception as e:
            session.rollback()
            print(f"❌ Error adding flagged event: {str(e)}")
            raise
        finally:
            session.close()
    
    def add_analysis_result(self, result_data: dict):
        """Add analysis result to database"""
        session = self.get_session()
        try:
            result = PrescreenerResultDB(**result_data)
            session.add(result)
            session.commit()
            print(f"✅ Added analysis result: {result_data['analysis_id']}")
            return result
        except Exception as e:
            session.rollback()
            print(f"❌ Error adding analysis result: {str(e)}")
            raise
        finally:
            session.close()
    
    def add_escalation_alert(self, alert_data: dict):
        """Add escalation alert to database"""
        session = self.get_session()
        try:
            alert = EscalationAlertDB(**alert_data)
            session.add(alert)
            session.commit()
            print(f"✅ Added escalation alert: {alert_data['alert_id']}")
            return alert
        except Exception as e:
            session.rollback()
            print(f"❌ Error adding escalation alert: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_pending_alerts(self, client_id: str = None):
        """Get pending escalation alerts"""
        session = self.get_session()
        try:
            query = session.query(EscalationAlertDB).filter(
                EscalationAlertDB.resolved == False
            )
            
            if client_id:
                query = query.filter(EscalationAlertDB.client_id == client_id)
            
            alerts = query.order_by(EscalationAlertDB.timestamp.desc()).all()
            return alerts
        except Exception as e:
            print(f"❌ Error getting pending alerts: {str(e)}")
            return []
        finally:
            session.close()
    
    def get_analysis_history(self, client_id: str, days: int = 30):
        """Get analysis history for a client"""
        session = self.get_session()
        try:
            from datetime import datetime, timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            results = session.query(PrescreenerResultDB).filter(
                PrescreenerResultDB.client_id == client_id,
                PrescreenerResultDB.analysis_timestamp >= cutoff_date
            ).order_by(PrescreenerResultDB.analysis_timestamp.desc()).all()
            
            return results
        except Exception as e:
            print(f"❌ Error getting analysis history: {str(e)}")
            return []
        finally:
            session.close()
    
    def resolve_alert(self, alert_id: str, resolved_by: str):
        """Mark an alert as resolved"""
        session = self.get_session()
        try:
            alert = session.query(EscalationAlertDB).filter(
                EscalationAlertDB.alert_id == alert_id
            ).first()
            
            if alert:
                alert.resolved = True
                alert.resolved_by = resolved_by
                alert.resolved_at = datetime.utcnow()
                session.commit()
                print(f"✅ Resolved alert: {alert_id}")
                return True
            else:
                print(f"❌ Alert not found: {alert_id}")
                return False
        except Exception as e:
            session.rollback()
            print(f"❌ Error resolving alert: {str(e)}")
            return False
        finally:
            session.close()

# Initialize database manager
db_manager = DatabaseManager()

# Export classes and manager
__all__ = [
    'ClientGuardrailsDB', 
    'FlaggedEventDB', 
    'PrescreenerResultDB', 
    'EscalationAlertDB',
    'DatabaseManager',
    'db_manager'
]
