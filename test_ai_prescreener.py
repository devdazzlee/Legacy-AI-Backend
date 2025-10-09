"""
Comprehensive Testing Suite for AI Prescreener System
=====================================================

This module provides comprehensive tests for all AI Prescreener functionalities:
1. Event-Based Flagging System
2. Compliance and Restriction Monitoring
3. Dynamic Narrative Generation
4. Customizable Guardrails Framework
5. Real-Time Alert System
6. Admin Interface

Author: Metaxoft AI Assistant
Version: 1.0.0
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Import the modules to test
from ai_prescreener import AIPrescreenerCore, ClientGuardrails, ShiftData, FlaggedEvent, EventType, AlertSeverity
from database_schema import DatabaseManager, ClientGuardrailsDB, FlaggedEventDB, PrescreenerResultDB
from alert_system import RealTimeAlertSystem, AlertRecipient, AlertChannel, AlertStatus
from admin_interface import app

class TestAIPrescreenerCore:
    """Test suite for AI Prescreener Core functionality"""
    
    @pytest.fixture
    def mock_azure_service(self):
        """Mock Azure OpenAI service"""
        mock_service = Mock()
        mock_service.deployment = "gpt-4o"
        mock_service.client = Mock()
        
        # Mock chat completion response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated narrative for test shift"
        mock_service.client.chat.completions.create.return_value = mock_response
        
        return mock_service
    
    @pytest.fixture
    def ai_prescreener(self, mock_azure_service):
        """Create AI Prescreener instance for testing"""
        return AIPrescreenerCore(mock_azure_service)
    
    @pytest.fixture
    def sample_shift_data(self):
        """Sample shift data for testing"""
        return ShiftData(
            shift_id="test_shift_001",
            client_id="client_001",
            worker_id="worker_001",
            shift_date=datetime.now(),
            shift_duration_hours=8.0,
            is_overnight_shift=False,
            worker_notes="The client fell during the morning routine. The client had a fever in the afternoon.",
            completed_tasks=["medication administration", "meal preparation", "personal care"],
            services_provided=["personal care", "medication management"],
            additional_context={"location": "home", "weather": "sunny"}
        )
    
    @pytest.fixture
    def sample_guardrails(self):
        """Sample client guardrails for testing"""
        return ClientGuardrails(
            client_id="client_001",
            client_name="Test Client",
            restricted_topics=["medication", "meal prep"],
            allowed_services=["personal care", "companionship"],
            medication_restrictions=["unauthorized medication"],
            dietary_restrictions=["nuts", "dairy"],
            activity_restrictions=["heavy lifting"],
            visitor_restrictions=["unauthorized visitors"],
            special_instructions=["Always state 'no challenges noted'"],
            narrative_requirements={"minLength": 50, "maxLength": 500},
            custom_flags=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def test_event_based_flagging_system(self, ai_prescreener, sample_shift_data, sample_guardrails):
        """Test 3.1: Event-Based Flagging System (Safety & Wellness Monitoring)"""
        print("🧪 Testing Event-Based Flagging System...")
        
        # Test fall detection
        flagged_events = ai_prescreener.detect_safety_events(sample_shift_data, sample_guardrails)
        
        # Should detect both "fell" and "fever" events
        assert len(flagged_events) >= 2, f"Expected at least 2 flagged events, got {len(flagged_events)}"
        
        # Check for fall event
        fall_events = [e for e in flagged_events if "fell" in e.description.lower()]
        assert len(fall_events) > 0, "Should detect fall event"
        assert fall_events[0].severity == AlertSeverity.CRITICAL, "Fall should be critical severity"
        assert fall_events[0].requires_escalation == True, "Fall should require escalation"
        
        # Check for fever event
        fever_events = [e for e in flagged_events if "fever" in e.description.lower()]
        assert len(fever_events) > 0, "Should detect fever event"
        assert fever_events[0].severity == AlertSeverity.HIGH, "Fever should be high severity"
        
        print("✅ Event-Based Flagging System tests passed")
    
    def test_compliance_monitoring(self, ai_prescreener, sample_shift_data, sample_guardrails):
        """Test 3.2: Compliance and Restriction Monitoring"""
        print("🧪 Testing Compliance and Restriction Monitoring...")
        
        # Test with medication restriction violation
        sample_shift_data.worker_notes = "Administered unauthorized medication to the client"
        violations = ai_prescreener.check_compliance_violations(sample_shift_data, sample_guardrails)
        
        assert len(violations) > 0, "Should detect medication violation"
        medication_violations = [v for v in violations if v["type"] == "medication_violation"]
        assert len(medication_violations) > 0, "Should detect medication violation"
        
        # Test with dietary restriction violation
        sample_shift_data.worker_notes = "Prepared meal with nuts for the client"
        violations = ai_prescreener.check_compliance_violations(sample_shift_data, sample_guardrails)
        
        dietary_violations = [v for v in violations if v["type"] == "dietary_violation"]
        assert len(dietary_violations) > 0, "Should detect dietary violation"
        
        print("✅ Compliance and Restriction Monitoring tests passed")
    
    def test_dynamic_narrative_generation(self, ai_prescreener, sample_shift_data, sample_guardrails):
        """Test 3.3: Dynamic Narrative Generation"""
        print("🧪 Testing Dynamic Narrative Generation...")
        
        # Test regular shift narrative
        sample_shift_data.shift_duration_hours = 8.0
        sample_shift_data.is_overnight_shift = False
        
        narrative = ai_prescreener.generate_shift_narrative(sample_shift_data, sample_guardrails, [])
        
        assert len(narrative) > 0, "Should generate narrative"
        assert len(narrative) >= sample_guardrails.narrative_requirements["minLength"], "Should meet minimum length"
        assert len(narrative) <= sample_guardrails.narrative_requirements["maxLength"], "Should not exceed maximum length"
        
        # Test overnight shift narrative
        sample_shift_data.is_overnight_shift = True
        overnight_narrative = ai_prescreener.generate_shift_narrative(sample_shift_data, sample_guardrails, [])
        
        assert len(overnight_narrative) > 0, "Should generate overnight narrative"
        assert "overnight" in overnight_narrative.lower(), "Should mention overnight shift"
        
        # Test proportional length logic
        sample_shift_data.shift_duration_hours = 12.0
        sample_shift_data.is_overnight_shift = False
        long_shift_narrative = ai_prescreener.generate_shift_narrative(sample_shift_data, sample_guardrails, [])
        
        # Longer shift should generate longer narrative (within limits)
        assert len(long_shift_narrative) >= len(narrative), "Longer shift should generate longer narrative"
        
        print("✅ Dynamic Narrative Generation tests passed")
    
    def test_customizable_guardrails_framework(self, ai_prescreener):
        """Test 3.4: Customizable Guardrails Framework"""
        print("🧪 Testing Customizable Guardrails Framework...")
        
        # Test adding custom guardrails
        custom_guardrails = ClientGuardrails(
            client_id="test_client_002",
            client_name="Custom Test Client",
            restricted_topics=["custom_topic"],
            allowed_services=["custom_service"],
            medication_restrictions=["custom_med_restriction"],
            dietary_restrictions=["custom_diet_restriction"],
            activity_restrictions=["custom_activity_restriction"],
            visitor_restrictions=["custom_visitor_restriction"],
            special_instructions=["custom_instruction"],
            narrative_requirements={"minLength": 100, "maxLength": 600},
            custom_flags=[
                {
                    "name": "custom_flag_1",
                    "type": "keyword_list",
                    "keywords": ["custom_keyword"],
                    "severity": "medium",
                    "requires_escalation": False,
                    "description": "Custom flag for testing"
                }
            ],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        ai_prescreener.add_client_guardrails(custom_guardrails)
        
        # Test retrieving custom guardrails
        retrieved_guardrails = ai_prescreener.get_client_guardrails("test_client_002")
        assert retrieved_guardrails.client_id == "test_client_002", "Should retrieve correct client guardrails"
        assert retrieved_guardrails.client_name == "Custom Test Client", "Should have correct client name"
        assert len(retrieved_guardrails.custom_flags) == 1, "Should have custom flags"
        
        # Test custom flag detection
        test_shift = ShiftData(
            shift_id="test_shift_002",
            client_id="test_client_002",
            worker_id="worker_001",
            shift_date=datetime.now(),
            shift_duration_hours=4.0,
            is_overnight_shift=False,
            worker_notes="The client mentioned custom_keyword during the shift",
            completed_tasks=[],
            services_provided=[],
            additional_context={}
        )
        
        flagged_events = ai_prescreener.detect_safety_events(test_shift, retrieved_guardrails)
        custom_flag_events = [e for e in flagged_events if "custom_flag" in e.description.lower()]
        assert len(custom_flag_events) > 0, "Should detect custom flag"
        
        print("✅ Customizable Guardrails Framework tests passed")
    
    def test_complete_analysis_workflow(self, ai_prescreener, sample_shift_data):
        """Test complete analysis workflow"""
        print("🧪 Testing Complete Analysis Workflow...")
        
        # Add guardrails for the test client
        test_guardrails = ClientGuardrails(
            client_id="client_001",
            client_name="Test Client",
            restricted_topics=["medication"],
            allowed_services=["personal care"],
            medication_restrictions=["unauthorized medication"],
            dietary_restrictions=["nuts"],
            activity_restrictions=["heavy lifting"],
            visitor_restrictions=["unauthorized visitors"],
            special_instructions=["Always state 'no challenges noted'"],
            narrative_requirements={"minLength": 50, "maxLength": 500},
            custom_flags=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        ai_prescreener.add_client_guardrails(test_guardrails)
        
        # Perform complete analysis
        result = ai_prescreener.analyze_shift(sample_shift_data)
        
        # Verify result structure
        assert result.analysis_id is not None, "Should have analysis ID"
        assert result.shift_id == sample_shift_data.shift_id, "Should have correct shift ID"
        assert result.client_id == sample_shift_data.client_id, "Should have correct client ID"
        assert len(result.flagged_events) > 0, "Should detect flagged events"
        assert len(result.generated_narrative) > 0, "Should generate narrative"
        assert result.processing_time_ms > 0, "Should have processing time"
        assert 0 <= result.confidence_score <= 1, "Should have valid confidence score"
        assert result.requires_human_review == True, "Should require human review due to flagged events"
        
        print("✅ Complete Analysis Workflow tests passed")

class TestDatabaseManager:
    """Test suite for Database Manager"""
    
    @pytest.fixture
    def db_manager(self):
        """Create database manager for testing"""
        return DatabaseManager("sqlite:///:memory:")
    
    def test_database_operations(self, db_manager):
        """Test database operations"""
        print("🧪 Testing Database Operations...")
        
        # Test adding client guardrails
        guardrails_data = {
            "client_id": "test_client_db",
            "client_name": "Test Client DB",
            "restricted_topics": ["test_topic"],
            "allowed_services": ["test_service"],
            "medication_restrictions": [],
            "dietary_restrictions": [],
            "activity_restrictions": [],
            "visitor_restrictions": [],
            "special_instructions": [],
            "narrative_requirements": {"minLength": 50, "maxLength": 500},
            "custom_flags": [],
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        db_guardrails = db_manager.add_client_guardrails(guardrails_data)
        assert db_guardrails.client_id == "test_client_db", "Should add guardrails correctly"
        
        # Test retrieving client guardrails
        retrieved = db_manager.get_client_guardrails("test_client_db")
        assert retrieved is not None, "Should retrieve guardrails"
        assert retrieved.client_name == "Test Client DB", "Should have correct client name"
        
        # Test adding flagged event
        event_data = {
            "event_id": "test_event_001",
            "client_id": "test_client_db",
            "event_type": "safety_incident",
            "severity": "critical",
            "description": "Test event",
            "original_text": "test",
            "timestamp": datetime.utcnow(),
            "requires_escalation": True,
            "escalation_notes": "Test escalation",
            "resolved": False
        }
        
        db_event = db_manager.add_flagged_event(event_data)
        assert db_event.event_id == "test_event_001", "Should add flagged event correctly"
        
        # Test adding analysis result
        analysis_data = {
            "analysis_id": "test_analysis_001",
            "shift_id": "test_shift_001",
            "client_id": "test_client_db",
            "worker_id": "worker_001",
            "shift_date": datetime.utcnow(),
            "shift_duration_hours": 8.0,
            "is_overnight_shift": False,
            "flagged_events_count": 1,
            "compliance_violations_count": 0,
            "generated_narrative": "Test narrative",
            "narrative_length": 13,
            "processing_time_ms": 100,
            "confidence_score": 0.9,
            "requires_human_review": True,
            "analysis_timestamp": datetime.utcnow()
        }
        
        db_analysis = db_manager.add_analysis_result(analysis_data)
        assert db_analysis.analysis_id == "test_analysis_001", "Should add analysis result correctly"
        
        print("✅ Database Operations tests passed")

class TestAlertSystem:
    """Test suite for Real-Time Alert System"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager"""
        return Mock()
    
    @pytest.fixture
    def alert_system(self, mock_db_manager):
        """Create alert system for testing"""
        return RealTimeAlertSystem(mock_db_manager)
    
    @pytest.fixture
    def sample_flagged_event(self):
        """Sample flagged event for testing"""
        return FlaggedEvent(
            event_id="test_event_001",
            client_id="client_001",
            event_type=EventType.SAFETY_INCIDENT,
            severity=AlertSeverity.CRITICAL,
            description="Test critical event",
            original_text="test",
            timestamp=datetime.now(),
            requires_escalation=True,
            escalation_notes="Test escalation"
        )
    
    def test_alert_system_initialization(self, alert_system):
        """Test alert system initialization"""
        print("🧪 Testing Alert System Initialization...")
        
        assert alert_system.config is not None, "Should have configuration"
        assert len(alert_system.escalation_rules) > 0, "Should have escalation rules"
        assert len(alert_system.alert_recipients) > 0, "Should have default recipients"
        assert len(alert_system.notification_handlers) > 0, "Should have notification handlers"
        
        print("✅ Alert System Initialization tests passed")
    
    @pytest.mark.asyncio
    async def test_flagged_event_processing(self, alert_system, sample_flagged_event):
        """Test flagged event processing"""
        print("🧪 Testing Flagged Event Processing...")
        
        # Mock the database operations
        alert_system.db_manager.add_escalation_alert = Mock()
        
        # Process flagged event
        result = await alert_system.process_flagged_event(sample_flagged_event)
        
        assert result["success"] == True, "Should process event successfully"
        assert "alert_id" in result, "Should return alert ID"
        
        # Verify escalation alert was added to database
        alert_system.db_manager.add_escalation_alert.assert_called_once()
        
        print("✅ Flagged Event Processing tests passed")
    
    def test_escalation_alert_generation(self, alert_system, sample_flagged_event):
        """Test escalation alert generation"""
        print("🧪 Testing Escalation Alert Generation...")
        
        escalation_rules = alert_system.escalation_rules["critical"]
        alert = alert_system.generate_escalation_alert(sample_flagged_event, escalation_rules)
        
        assert alert["alert_id"] is not None, "Should have alert ID"
        assert alert["event_id"] == sample_flagged_event.event_id, "Should have correct event ID"
        assert alert["client_id"] == sample_flagged_event.client_id, "Should have correct client ID"
        assert alert["severity"] == sample_flagged_event.severity.value, "Should have correct severity"
        assert alert["priority"] == "HIGH", "Critical events should have HIGH priority"
        assert "action_required" in alert, "Should have action required"
        assert "escalation_path" in alert, "Should have escalation path"
        
        print("✅ Escalation Alert Generation tests passed")
    
    def test_recipient_management(self, alert_system):
        """Test alert recipient management"""
        print("🧪 Testing Recipient Management...")
        
        # Test getting recipients for escalation level
        level_1_recipients = alert_system.get_recipients_for_level(1)
        assert len(level_1_recipients) > 0, "Should have level 1 recipients"
        
        # Test adding custom recipient
        custom_recipient = AlertRecipient(
            user_id="custom_001",
            name="Custom Recipient",
            email="custom@test.com",
            role="custom_role",
            escalation_level=1
        )
        
        alert_system.alert_recipients["custom_001"] = custom_recipient
        
        # Verify recipient was added
        assert "custom_001" in alert_system.alert_recipients, "Should add custom recipient"
        
        print("✅ Recipient Management tests passed")

class TestAdminInterface:
    """Test suite for Admin Interface API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing"""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        print("🧪 Testing Health Check Endpoint...")
        
        response = client.get("/health")
        assert response.status_code == 200, "Health check should return 200"
        
        data = response.json()
        assert data["status"] == "healthy", "Should return healthy status"
        assert "timestamp" in data, "Should include timestamp"
        
        print("✅ Health Check Endpoint tests passed")
    
    def test_system_status(self, client):
        """Test system status endpoint"""
        print("🧪 Testing System Status Endpoint...")
        
        response = client.get("/system/status")
        assert response.status_code == 200, "System status should return 200"
        
        data = response.json()
        assert "system_status" in data, "Should have system status"
        assert "ai_prescreener_status" in data, "Should have AI prescreener status"
        assert "alert_system_status" in data, "Should have alert system status"
        assert "database_status" in data, "Should have database status"
        
        print("✅ System Status Endpoint tests passed")
    
    def test_client_guardrails_crud(self, client):
        """Test client guardrails CRUD operations"""
        print("🧪 Testing Client Guardrails CRUD...")
        
        # Test creating guardrails
        guardrails_data = {
            "client_id": "test_api_client",
            "client_name": "Test API Client",
            "restricted_topics": ["test_topic"],
            "allowed_services": ["test_service"],
            "medication_restrictions": [],
            "dietary_restrictions": [],
            "activity_restrictions": [],
            "visitor_restrictions": [],
            "special_instructions": [],
            "narrative_requirements": {"minLength": 50, "maxLength": 500},
            "custom_flags": [],
            "is_active": True
        }
        
        response = client.post("/admin/guardrails", json=guardrails_data)
        assert response.status_code == 200, "Should create guardrails successfully"
        
        # Test retrieving guardrails
        response = client.get(f"/admin/guardrails/{guardrails_data['client_id']}")
        assert response.status_code == 200, "Should retrieve guardrails successfully"
        
        data = response.json()
        assert data["client_id"] == guardrails_data["client_id"], "Should have correct client ID"
        assert data["client_name"] == guardrails_data["client_name"], "Should have correct client name"
        
        # Test listing all guardrails
        response = client.get("/admin/guardrails")
        assert response.status_code == 200, "Should list guardrails successfully"
        
        guardrails_list = response.json()
        assert len(guardrails_list) > 0, "Should have guardrails in list"
        
        print("✅ Client Guardrails CRUD tests passed")
    
    def test_shift_analysis(self, client):
        """Test shift analysis endpoint"""
        print("🧪 Testing Shift Analysis Endpoint...")
        
        shift_data = {
            "shift_id": "test_shift_api",
            "client_id": "test_api_client",
            "worker_id": "worker_001",
            "shift_date": datetime.now().isoformat(),
            "shift_duration_hours": 8.0,
            "is_overnight_shift": False,
            "worker_notes": "The client fell during morning routine. The client had a fever.",
            "completed_tasks": ["medication", "meal prep"],
            "services_provided": ["personal care"],
            "additional_context": {}
        }
        
        response = client.post("/analysis/shift", json=shift_data)
        assert response.status_code == 200, "Should analyze shift successfully"
        
        data = response.json()
        assert data["analysis_id"] is not None, "Should have analysis ID"
        assert data["shift_id"] == shift_data["shift_id"], "Should have correct shift ID"
        assert data["client_id"] == shift_data["client_id"], "Should have correct client ID"
        assert len(data["flagged_events"]) > 0, "Should detect flagged events"
        assert len(data["generated_narrative"]) > 0, "Should generate narrative"
        assert data["requires_human_review"] == True, "Should require human review"
        
        print("✅ Shift Analysis Endpoint tests passed")

class TestIntegration:
    """Integration tests for the complete AI Prescreener system"""
    
    @pytest.fixture
    def complete_system(self):
        """Set up complete system for integration testing"""
        # Mock Azure service
        mock_azure = Mock()
        mock_azure.deployment = "gpt-4o"
        mock_azure.client = Mock()
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Integration test narrative"
        mock_azure.client.chat.completions.create.return_value = mock_response
        
        # Initialize components
        db_manager = DatabaseManager("sqlite:///:memory:")
        ai_prescreener = AIPrescreenerCore(mock_azure)
        alert_system = RealTimeAlertSystem(db_manager)
        
        return {
            "db_manager": db_manager,
            "ai_prescreener": ai_prescreener,
            "alert_system": alert_system
        }
    
    def test_end_to_end_workflow(self, complete_system):
        """Test complete end-to-end workflow"""
        print("🧪 Testing End-to-End Workflow...")
        
        db_manager = complete_system["db_manager"]
        ai_prescreener = complete_system["ai_prescreener"]
        alert_system = complete_system["alert_system"]
        
        # 1. Set up client guardrails
        guardrails_data = {
            "client_id": "integration_client",
            "client_name": "Integration Test Client",
            "restricted_topics": ["medication"],
            "allowed_services": ["personal care"],
            "medication_restrictions": ["unauthorized medication"],
            "dietary_restrictions": ["nuts"],
            "activity_restrictions": ["heavy lifting"],
            "visitor_restrictions": ["unauthorized visitors"],
            "special_instructions": ["Always state 'no challenges noted'"],
            "narrative_requirements": {"minLength": 50, "maxLength": 500},
            "custom_flags": [],
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        db_guardrails = db_manager.add_client_guardrails(guardrails_data)
        
        # 2. Create shift data with safety incidents
        shift_data = ShiftData(
            shift_id="integration_shift",
            client_id="integration_client",
            worker_id="worker_001",
            shift_date=datetime.now(),
            shift_duration_hours=8.0,
            is_overnight_shift=False,
            worker_notes="The client fell in the bathroom. The client had a fever of 101°F. Administered unauthorized medication.",
            completed_tasks=["medication", "meal prep", "personal care"],
            services_provided=["personal care", "medication management"],
            additional_context={}
        )
        
        # 3. Perform AI analysis
        result = ai_prescreener.analyze_shift(shift_data)
        
        # 4. Verify analysis results
        assert result.analysis_id is not None, "Should have analysis ID"
        assert len(result.flagged_events) >= 3, "Should detect fall, fever, and medication events"
        assert len(result.compliance_violations) > 0, "Should detect compliance violations"
        assert len(result.generated_narrative) > 0, "Should generate narrative"
        assert result.requires_human_review == True, "Should require human review"
        
        # 5. Store analysis result
        analysis_data = {
            "analysis_id": result.analysis_id,
            "shift_id": result.shift_id,
            "client_id": result.client_id,
            "worker_id": "worker_001",
            "shift_date": shift_data.shift_date,
            "shift_duration_hours": shift_data.shift_duration_hours,
            "is_overnight_shift": shift_data.is_overnight_shift,
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
        
        # 6. Process flagged events for alerts
        for event in result.flagged_events:
            event_data = {
                "event_id": event.event_id,
                "client_id": event.client_id,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "description": event.description,
                "original_text": event.original_text,
                "timestamp": event.timestamp,
                "requires_escalation": event.requires_escalation,
                "escalation_notes": event.escalation_notes,
                "resolved": False
            }
            db_manager.add_flagged_event(event_data)
        
        # 7. Verify data persistence
        stored_guardrails = db_manager.get_client_guardrails("integration_client")
        assert stored_guardrails is not None, "Should store guardrails"
        
        stored_analysis = db_manager.get_analysis_history("integration_client", 1)
        assert len(stored_analysis) > 0, "Should store analysis history"
        
        print("✅ End-to-End Workflow tests passed")
    
    def test_mandatory_examples_from_video(self, complete_system):
        """Test mandatory examples from the video requirements"""
        print("🧪 Testing Mandatory Examples from Video...")
        
        ai_prescreener = complete_system["ai_prescreener"]
        
        # Test "The client fell" detection
        fall_shift = ShiftData(
            shift_id="fall_test",
            client_id="test_client",
            worker_id="worker_001",
            shift_date=datetime.now(),
            shift_duration_hours=4.0,
            is_overnight_shift=False,
            worker_notes="The client fell in the living room.",
            completed_tasks=[],
            services_provided=[],
            additional_context={}
        )
        
        result = ai_prescreener.analyze_shift(fall_shift)
        fall_events = [e for e in result.flagged_events if "fell" in e.description.lower()]
        assert len(fall_events) > 0, "Should detect 'The client fell'"
        assert fall_events[0].severity == AlertSeverity.CRITICAL, "Fall should be critical severity"
        
        # Test "The client had a fever" detection
        fever_shift = ShiftData(
            shift_id="fever_test",
            client_id="test_client",
            worker_id="worker_001",
            shift_date=datetime.now(),
            shift_duration_hours=4.0,
            is_overnight_shift=False,
            worker_notes="The client had a fever of 102°F.",
            completed_tasks=[],
            services_provided=[],
            additional_context={}
        )
        
        result = ai_prescreener.analyze_shift(fever_shift)
        fever_events = [e for e in result.flagged_events if "fever" in e.description.lower()]
        assert len(fever_events) > 0, "Should detect 'The client had a fever'"
        assert fever_events[0].severity == AlertSeverity.HIGH, "Fever should be high severity"
        
        print("✅ Mandatory Examples from Video tests passed")

def run_all_tests():
    """Run all test suites"""
    print("🚀 Starting Comprehensive AI Prescreener Test Suite")
    print("=" * 60)
    
    # Run individual test classes
    test_classes = [
        TestAIPrescreenerCore,
        TestDatabaseManager,
        TestAlertSystem,
        TestAdminInterface,
        TestIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}...")
        
        # Create test instance
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                # Run the test method
                method = getattr(test_instance, test_method)
                
                # Handle async tests
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    method()
                
                passed_tests += 1
                print(f"  ✅ {test_method}")
                
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! AI Prescreener system is working correctly.")
    else:
        print(f"⚠️ {total_tests - passed_tests} tests failed. Please review the errors above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    if success:
        print("\n✅ AI Prescreener System is ready for deployment!")
    else:
        print("\n❌ AI Prescreener System needs fixes before deployment.")

