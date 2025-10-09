#!/usr/bin/env python3
"""
AI Prescreener Database Setup Script
====================================

This script initializes the PostgreSQL database for the AI Prescreener system.

Author: Metaxoft AI Assistant
Version: 1.0.0
"""

import os
import sys
from sqlalchemy import create_engine, text
from database_schema import Base, DatabaseManager

def setup_database():
    """Setup the PostgreSQL database for AI Prescreener"""
    
    # Database connection string
    database_url = os.getenv(
        "DATABASE_URL", 
        "postgresql://neondb_owner:npg_vCLPn9zdK2uY@ep-old-art-adro7qy2-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    )
    
    print("🚀 Setting up AI Prescreener Database...")
    print(f"📊 Database URL: {database_url[:50]}...")
    
    try:
        # Test database connection
        engine = create_engine(database_url, echo=False)
        
        with engine.connect() as conn:
            # Test connection
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
        
        # Create database manager and tables
        db_manager = DatabaseManager(database_url)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        
        # Add some default client guardrails
        add_default_guardrails(db_manager)
        
        print("🎉 Database setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {str(e)}")
        return False

def add_default_guardrails(db_manager):
    """Add default client guardrails"""
    
    default_guardrails = {
        "client_id": "default_client",
        "client_name": "Default Client",
        "restricted_topics": ["medication", "meal prep", "medical advice"],
        "allowed_services": ["personal care", "companionship", "light housekeeping"],
        "medication_restrictions": ["unauthorized medication", "overdose", "wrong medication"],
        "dietary_restrictions": ["nuts", "dairy", "gluten"],
        "activity_restrictions": ["heavy lifting", "unsupervised outings"],
        "visitor_restrictions": ["unauthorized visitors", "after hours visits"],
        "special_instructions": [
            "Always explicitly state 'no challenges noted' when no issues occur",
            "Generate narratives proportional to shift length",
            "Avoid discussing restricted topics"
        ],
        "narrative_requirements": {
            "minLength": 50,
            "maxLength": 500,
            "overnightShiftFormat": True
        },
        "custom_flags": [
            {
                "name": "fall_detection",
                "type": "keyword_list",
                "keywords": ["fell", "fall", "fallen", "falling"],
                "severity": "critical",
                "requires_escalation": True,
                "description": "Detects fall incidents"
            },
            {
                "name": "fever_detection",
                "type": "keyword_list",
                "keywords": ["fever", "temperature", "temp"],
                "severity": "high",
                "requires_escalation": True,
                "description": "Detects fever incidents"
            }
        ],
        "is_active": True
    }
    
    try:
        db_manager.add_client_guardrails(default_guardrails)
        print("✅ Default client guardrails added")
    except Exception as e:
        print(f"⚠️ Failed to add default guardrails: {str(e)}")

def test_database_operations():
    """Test basic database operations"""
    
    print("\n🧪 Testing database operations...")
    
    try:
        db_manager = DatabaseManager()
        
        # Test adding a flagged event
        test_event = {
            "event_id": "test_event_001",
            "client_id": "default_client",
            "event_type": "safety_incident",
            "severity": "critical",
            "description": "Test fall detection",
            "original_text": "The client fell",
            "timestamp": "2024-01-01T12:00:00Z",
            "requires_escalation": True,
            "escalation_notes": "Test escalation",
            "resolved": False
        }
        
        db_manager.add_flagged_event(test_event)
        print("✅ Test flagged event added")
        
        # Test adding analysis result
        test_analysis = {
            "analysis_id": "test_analysis_001",
            "shift_id": "test_shift_001",
            "client_id": "default_client",
            "worker_id": "worker_001",
            "shift_date": "2024-01-01T12:00:00Z",
            "shift_duration_hours": 8.0,
            "is_overnight_shift": False,
            "flagged_events_count": 1,
            "compliance_violations_count": 0,
            "generated_narrative": "Test narrative for shift analysis",
            "narrative_length": 35,
            "processing_time_ms": 1500,
            "confidence_score": 0.95,
            "requires_human_review": True,
            "analysis_timestamp": "2024-01-01T12:00:00Z"
        }
        
        db_manager.add_analysis_result(test_analysis)
        print("✅ Test analysis result added")
        
        print("🎉 Database operations test completed successfully!")
        
    except Exception as e:
        print(f"❌ Database operations test failed: {str(e)}")

if __name__ == "__main__":
    print("=" * 60)
    print("AI Prescreener Database Setup")
    print("=" * 60)
    
    # Setup database
    if setup_database():
        # Test operations
        test_database_operations()
        
        print("\n" + "=" * 60)
        print("✅ Setup completed successfully!")
        print("🚀 You can now start the AI Prescreener server:")
        print("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Setup failed!")
        print("Please check your database connection and try again.")
        print("=" * 60)
        sys.exit(1)
