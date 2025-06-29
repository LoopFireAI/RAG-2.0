#!/usr/bin/env python3
"""
Comprehensive stress test for the RAG system.
Tests all nodes, workflows, and edge cases.
"""

import os
import sys
import asyncio
import traceback
from typing import Dict, Any, List
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from langchain_core.messages import HumanMessage, AIMessage
from rag_2_0.agents.rag_agent import (
    RAGState, 
    create_rag_graph,
    extract_query,
    detect_social_media_request,
    elicit_leader,
    process_leader_selection,
    detect_tone_and_leader,
    retrieve_documents,
    grade_documents,
    generate_response,
    generate_social_media_post,
    register_response_for_feedback,
    collect_feedback,
    load_tone_profile
)

class RAGSystemTester:
    def __init__(self):
        self.graph = create_rag_graph()
        self.test_results = []
        self.passed = 0
        self.failed = 0
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results."""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append(f"{status}: {test_name}")
        if details:
            self.test_results.append(f"   Details: {details}")
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            
        print(f"{status}: {test_name}")
        if details and not passed:
            print(f"   Details: {details}")
    
    def test_individual_nodes(self):
        """Test each node individually."""
        print("\n🔍 Testing Individual Nodes...")
        
        # Test extract_query
        try:
            state = {"messages": [HumanMessage(content="What is machine learning?")]}
            result = extract_query(state)
            assert "query" in result
            assert result["query"] == "What is machine learning?"
            self.log_test("extract_query node", True)
        except Exception as e:
            self.log_test("extract_query node", False, str(e))
        
        # Test detect_social_media_request
        try:
            # Test non-social media query
            state = {"query": "What is machine learning?"}
            result = detect_social_media_request(state)
            assert "is_social_media" in result
            assert result["is_social_media"] == False
            
            # Test social media query
            state = {"query": "Create a tweet about AI"}
            result = detect_social_media_request(state)
            assert result["is_social_media"] == True
            self.log_test("detect_social_media_request node", True)
        except Exception as e:
            self.log_test("detect_social_media_request node", False, str(e))
        
        # Test load_tone_profile
        try:
            janelle_profile = load_tone_profile("janelle")
            assert len(janelle_profile) > 0
            assert "Janelle" in janelle_profile
            
            default_profile = load_tone_profile("default")
            assert len(default_profile) > 0
            
            doreen_profile = load_tone_profile("doreen")
            assert len(doreen_profile) > 0
            self.log_test("load_tone_profile function", True)
        except Exception as e:
            self.log_test("load_tone_profile function", False, str(e))
        
        # Test detect_tone_and_leader
        try:
            # Test with leader already detected
            state = {"query": "test", "detected_leader": "janelle", "tone_profile": "test profile"}
            result = detect_tone_and_leader(state)
            assert result["detected_leader"] == "janelle"
            assert result["tone_profile"] == "test profile"
            
            # Test pattern matching
            state = {"query": "respond as janelle would"}
            result = detect_tone_and_leader(state)
            assert "detected_leader" in result
            assert "tone_profile" in result
            self.log_test("detect_tone_and_leader node", True)
        except Exception as e:
            self.log_test("detect_tone_and_leader node", False, str(e))
    
    def test_leader_elicitation(self):
        """Test leader elicitation scenarios."""
        print("\n👥 Testing Leader Elicitation...")
        
        # Test with explicit leader mention
        try:
            state = {"query": "Can you respond as Janelle about leadership?"}
            result = elicit_leader(state)
            
            if result.get("waiting_for_leader"):
                self.log_test("Leader elicitation - explicit mention", False, "Should have detected Janelle")
            else:
                assert result.get("detected_leader") in ["janelle", "doreen"]
                assert "tone_profile" in result
                self.log_test("Leader elicitation - explicit mention", True)
        except Exception as e:
            self.log_test("Leader elicitation - explicit mention", False, str(e))
        
        # Test without leader mention (should prompt)
        try:
            state = {"query": "What is effective leadership?"}
            result = elicit_leader(state)
            
            if result.get("waiting_for_leader"):
                assert "messages" in result
                assert "original_query" in result
                self.log_test("Leader elicitation - no mention (prompting)", True)
            else:
                self.log_test("Leader elicitation - no mention (prompting)", False, "Should have prompted for leader")
        except Exception as e:
            self.log_test("Leader elicitation - no mention (prompting)", False, str(e))
        
        # Test process_leader_selection
        try:
            state = {
                "messages": [HumanMessage(content="Janelle")],
                "original_query": "What is leadership?",
                "waiting_for_leader": True
            }
            result = process_leader_selection(state)
            assert result["detected_leader"] == "janelle"
            assert "tone_profile" in result
            assert result["waiting_for_leader"] == False
            assert result["query"] == "What is leadership?"
            self.log_test("process_leader_selection node", True)
        except Exception as e:
            self.log_test("process_leader_selection node", False, str(e))
    
    def test_document_retrieval(self):
        """Test document retrieval and grading."""
        print("\n📚 Testing Document Retrieval...")
        
        try:
            # Test retrieve_documents
            state = {"query": "machine learning"}
            result = retrieve_documents(state)
            
            assert "documents" in result
            assert "context" in result
            assert "sources" in result
            assert "retrieved_docs_metadata" in result
            assert isinstance(result["documents"], list)
            
            # Test grade_documents
            grade_state = {
                "query": "machine learning",
                "context": result["context"]
            }
            grade_result = grade_documents(grade_state)
            assert "grade" in grade_result
            assert grade_result["grade"] in ["yes", "no"]
            
            self.log_test("Document retrieval and grading", True, f"Retrieved {len(result['documents'])} docs")
        except Exception as e:
            self.log_test("Document retrieval and grading", False, str(e))
    
    def test_response_generation(self):
        """Test response generation."""
        print("\n✍️ Testing Response Generation...")
        
        # Test regular response generation
        try:
            state = {
                "query": "What is machine learning?",
                "context": "Machine learning is a subset of AI that enables computers to learn.",
                "grade": "yes",
                "sources": ["test_source.pdf"],
                "detected_leader": "janelle",
                "tone_profile": "Professional tone"
            }
            result = generate_response(state)
            
            assert "messages" in result
            assert len(result["messages"]) > 0
            assert "response_id" in result
            assert "feedback_collected" in result
            self.log_test("Regular response generation", True)
        except Exception as e:
            self.log_test("Regular response generation", False, str(e))
        
        # Test social media generation
        try:
            state = {
                "query": "Create a tweet about machine learning",
                "context": "Machine learning is transforming industries",
                "grade": "yes",
                "detected_leader": "janelle",
                "tone_profile": "Engaging social media tone",
                "is_social_media": True
            }
            result = generate_social_media_post(state)
            
            assert "messages" in result
            assert len(result["messages"]) > 0
            assert "response_id" in result
            self.log_test("Social media generation", True)
        except Exception as e:
            self.log_test("Social media generation", False, str(e))
    
    def test_feedback_system(self):
        """Test feedback collection system."""
        print("\n📝 Testing Feedback System...")
        
        try:
            # Test register_response_for_feedback
            state = {
                "query": "test query",
                "messages": [AIMessage(content="test response")],
                "response_id": "test-123",
                "retrieved_docs_metadata": [{"id": "doc1", "title": "Test Doc"}]
            }
            result = register_response_for_feedback(state)
            # This should not fail even if feedback system is not fully initialized
            assert isinstance(result, dict)
            
            # Test collect_feedback
            feedback_result = collect_feedback(state)
            assert isinstance(feedback_result, dict)
            
            self.log_test("Feedback system", True)
        except Exception as e:
            self.log_test("Feedback system", False, str(e))
    
    def test_full_workflows(self):
        """Test complete workflows end-to-end."""
        print("\n🔄 Testing Full Workflows...")
        
        # Test 1: Basic RAG workflow with explicit leader
        try:
            initial_state = {
                "messages": [HumanMessage(content="What is leadership? Respond as Janelle.")]
            }
            
            # Run the workflow (this may take a while)
            result = self.graph.invoke(initial_state)
            
            assert "messages" in result
            assert len(result["messages"]) > 0
            assert result.get("detected_leader") == "janelle"
            assert not result.get("waiting_for_leader", False)
            
            self.log_test("Full RAG workflow - explicit leader", True)
        except Exception as e:
            self.log_test("Full RAG workflow - explicit leader", False, str(e))
        
        # Test 2: Social media workflow
        try:
            initial_state = {
                "messages": [HumanMessage(content="Create a LinkedIn post about leadership as Doreen")]
            }
            
            result = self.graph.invoke(initial_state)
            
            assert "messages" in result
            assert result.get("is_social_media") == True
            assert result.get("detected_leader") == "doreen"
            
            self.log_test("Social media workflow", True)
        except Exception as e:
            self.log_test("Social media workflow", False, str(e))
        
        # Test 3: Workflow that should prompt for leader
        try:
            initial_state = {
                "messages": [HumanMessage(content="What makes a good leader?")]
            }
            
            result = self.graph.invoke(initial_state)
            
            # This should either complete with default or prompt for leader
            if result.get("waiting_for_leader"):
                assert "messages" in result
                assert "original_query" in result
                self.log_test("Workflow - leader prompting", True, "Correctly prompted for leader")
            else:
                # Should have used default
                assert result.get("detected_leader") in ["default", "janelle", "doreen"]
                self.log_test("Workflow - leader prompting", True, "Used default leader")
        except Exception as e:
            self.log_test("Workflow - leader prompting", False, str(e))
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n⚠️ Testing Edge Cases...")
        
        # Test empty query
        try:
            state = {"messages": [HumanMessage(content="")]}
            result = extract_query(state)
            assert "query" in result
            self.log_test("Empty query handling", True)
        except Exception as e:
            self.log_test("Empty query handling", False, str(e))
        
        # Test missing fields
        try:
            state = {}
            result = extract_query(state)
            assert "query" in result
            self.log_test("Missing fields handling", True)
        except Exception as e:
            self.log_test("Missing fields handling", False, str(e))
        
        # Test invalid leader selection
        try:
            state = {
                "messages": [HumanMessage(content="invalid_leader")],
                "original_query": "test"
            }
            result = process_leader_selection(state)
            assert result["detected_leader"] == "default"  # Should fallback
            self.log_test("Invalid leader selection", True)
        except Exception as e:
            self.log_test("Invalid leader selection", False, str(e))
    
    def run_all_tests(self):
        """Run all tests."""
        print("🚀 Starting RAG System Stress Test...")
        print("=" * 50)
        
        # Run all test categories
        self.test_individual_nodes()
        self.test_leader_elicitation()
        self.test_document_retrieval()
        self.test_response_generation()
        self.test_feedback_system()
        self.test_full_workflows()
        self.test_edge_cases()
        
        # Print final results
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        for result in self.test_results:
            print(result)
        
        print(f"\n✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📈 Success Rate: {(self.passed / (self.passed + self.failed) * 100):.1f}%")
        
        if self.failed == 0:
            print("\n🎉 ALL TESTS PASSED! Your RAG system is working correctly.")
        else:
            print(f"\n⚠️ {self.failed} tests failed. Check the details above.")
        
        return self.failed == 0

if __name__ == "__main__":
    tester = RAGSystemTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)