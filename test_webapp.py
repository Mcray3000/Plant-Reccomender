#!/usr/bin/env python
"""
Test script for Plant Recommender Web App
Tests both the questionnaire and item-based flows
"""

import requests
import json
import time
import subprocess
import sys
from threading import Thread

BASE_URL = "http://127.0.0.1:8000"

def start_server():
    """Start the Flask development server in the background"""
    proc = subprocess.Popen(
        ["python", "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(3)  # Wait for server to start
    return proc

def test_home_page():
    """Test that home page loads"""
    print("Testing home page...")
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200, f"Home page failed: {response.status_code}"
    assert "Plant Recommender" in response.text
    print("Home page loads successfully")

def test_questionnaire_page():
    """Test that questionnaire page loads"""
    print("\nTesting questionnaire page...")
    response = requests.get(f"{BASE_URL}/questionnaire")
    assert response.status_code == 200, f"Questionnaire page failed: {response.status_code}"
    assert "Light Availability" in response.text or "light" in response.text.lower()
    print("Questionnaire page loads successfully")

def test_item_based_page():
    """Test that item-based page loads"""
    print("\nTesting item-based page...")
    response = requests.get(f"{BASE_URL}/item-based")
    assert response.status_code == 200, f"Item-based page failed: {response.status_code}"
    assert "Your Plants" in response.text or "plants" in response.text.lower()
    print("Item-based page loads successfully")

def test_questionnaire_recommendation():
    """Test questionnaire recommendation API"""
    print("\nTesting questionnaire recommendation API...")
    payload = {
        "light": "Grow Light",
        "care": "Low",
        "room": "Living Room",
        "flowers": "Dont Care",
        "toxic": "No",
        "model": "heuristic"
    }
    
    response = requests.post(f"{BASE_URL}/api/recommend/questionnaire", json=payload)
    assert response.status_code == 200, f"API failed: {response.status_code}"
    
    result = response.json()
    assert result["success"] == True, f"API returned error: {result.get('error')}"
    assert "plants" in result, "No plants in response"
    assert len(result["plants"]) > 0, "No plants returned"
    
    # Check plant data structure
    plant = result["plants"][0]
    required_keys = ["scientific_name", "common_name", "image_url", "light_level", 
                     "water_need", "general_care", "is_toxic"]
    for key in required_keys:
        assert key in plant, f"Missing key in plant data: {key}"
    
    print(f"Questionnaire API works! Got {len(result['plants'])} recommendations")
    print(f"  Example: {plant['common_name']} ({plant['scientific_name']})")
    print(f"  Light: {plant['light_level']}, Water: {plant['water_need']}")
    print(f"  Toxic: {plant['is_toxic']}, Care: {plant['general_care'][:50] if plant['general_care'] else 'N/A'}...")

def test_item_based_recommendation():
    """Test item-based recommendation API"""
    print("\nTesting item-based recommendation API...")
    payload = {
        "plants": ["Monstera deliciosa", "Epipremnum aureum"],
        "flowers": "Dont Care",
        "toxic": "Dont Care",
        "model": "content"
    }
    
    response = requests.post(f"{BASE_URL}/api/recommend/item-based", json=payload)
    assert response.status_code == 200, f"API failed: {response.status_code}"
    
    result = response.json()
    assert result["success"] == True, f"API returned error: {result.get('error')}"
    assert "plants" in result, "No plants in response"
    assert len(result["plants"]) > 0, "No plants returned"
    
    # Check plant data structure
    plant = result["plants"][0]
    required_keys = ["scientific_name", "common_name", "image_url", "light_level", 
                     "water_need", "general_care", "is_toxic"]
    for key in required_keys:
        assert key in plant, f"Missing key in plant data: {key}"
    
    print(f"Item-based API works! Got {len(result['plants'])} recommendations")
    print(f"  Example: {plant['common_name']} ({plant['scientific_name']})")
    print(f"  Light: {plant['light_level']}, Water: {plant['water_need']}")
    print(f"  Toxic: {plant['is_toxic']}, Care: {plant['general_care'][:50] if plant['general_care'] else 'N/A'}...")

def test_plant_search():
    """Test plant search API"""
    print("\nTesting plant search API...")
    response = requests.get(f"{BASE_URL}/api/plants/search?q=monstera")
    assert response.status_code == 200, f"Search API failed: {response.status_code}"
    
    results = response.json()
    assert len(results) > 0, "No search results"
    
    result = results[0]
    assert "scientific" in result, "Missing scientific name in search result"
    assert "common" in result, "Missing common name in search result"
    assert "label" in result, "Missing label in search result"
    
    print(f"Plant search works! Found {len(results)} results")
    print(f"  Example: {result['label']}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("PLANT RECOMMENDER WEB APP TEST SUITE")
    print("=" * 60)
    
    print("\nStarting Flask development server...")
    server_proc = start_server()
    
    try:
        # Give the server time to initialize
        time.sleep(2)
        
        # Run tests
        test_home_page()
        test_questionnaire_page()
        test_item_based_page()
        test_questionnaire_recommendation()
        test_item_based_recommendation()
        test_plant_search()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print(f"\nThe web app is ready! Visit: http://127.0.0.1:8000")
        print("\nFeatures verified:")
        print("  Home page with navigation")
        print("  Questionnaire form and recommendations")
        print("  Item-based recommendations")
        print("  Plant search autocomplete")
        print("  Images and care information display")
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return 1
    finally:
        print("\nShutting down server...")
        server_proc.terminate()
        server_proc.wait()

if __name__ == "__main__":
    sys.exit(main())
