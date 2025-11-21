#!/usr/bin/env python3
"""
Simple A2A client test for SOP Agent.
Tests multi-tenant functionality with different store policies.
"""

import requests
import json
from uuid import uuid4


def get_agent_card(base_url: str = "http://localhost:9998"):
    """Fetch the agent card."""
    response = requests.get(f"{base_url}/.well-known/agent.json")
    response.raise_for_status()
    return response.json()


def send_message(
    base_url: str, message: str, tenant_id: str = "default", thread_id: str = None
):
    """Send a message to the agent with tenant_id support."""
    metadata = {"tenant_id": tenant_id}

    if thread_id:
        metadata["thread_id"] = thread_id

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
                "messageId": str(uuid4()),
            },
            "metadata": metadata,
        },
    }

    response = requests.post(
        f"{base_url}/", json=payload, headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    return response.json()


def extract_text_from_response(response_data):
    """Extract text from JSON-RPC response."""
    try:
        result = response_data.get("result", {})
        parts = result.get("parts", [])
        texts = [part.get("text", "") for part in parts if part.get("type") == "text"]
        return " ".join(texts)
    except Exception:
        return str(response_data)


def main():
    """Run simple tests."""
    base_url = "http://localhost:9998"

    print("🧪 SOP Agent - Multi-Tenant Test\n")

    # Get agent card
    print("📡 Fetching agent card...")
    try:
        card = get_agent_card(base_url)
        print(f"✅ Connected to: {card['name']}")
        print(f"   Description: {card['description']}\n")

        print("📋 Available Skills:")
        for skill in card.get("skills", []):
            print(f"  - {skill['name']}: {skill['description']}")
        print()

    except Exception as e:
        print(f"❌ Failed to fetch agent card: {e}")
        return

    # Test messages with different tenants
    # Format: (message, tenant_id, thread_id)
    test_cases = [
        # Default tenant - Standard store policies
        # ("What is the return policy?", "default", "employee_1"),
        # ("How do I handle a customer complaint?", "default", "employee_1"),
        # Tenant A - Custom policies
        # ("What is the return policy for electronics?", "tenant_a", "store_a_emp_1"),
        # ("How should I process refunds?", "tenant_a", "store_a_emp_1"),
        # Tenant B - Different policies
        ("What is the employee dress code?", "tenant_b", "store_b_emp_1"),
        ("What are the safety procedures?", "tenant_b", "store_b_emp_2"),
    ]

    for i, (message, tenant_id, thread_id) in enumerate(test_cases, 1):
        print(f"{'='*70}")
        print(f"TEST {i}: {message}")
        print(f"Tenant: {tenant_id} | Thread: {thread_id}")
        print(f"{'='*70}")

        try:
            response = send_message(
                base_url, message, tenant_id=tenant_id, thread_id=thread_id
            )
            print("Response JSON:")
            print(json.dumps(response, indent=2))
            result_text = extract_text_from_response(response)
            print(f"📥 Response: {result_text}\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    print("\n📋 SOP Agent - Multi-Tenant Test Client")
    print("Make sure your agent is running at http://localhost:9998")
    print("Start it with: python -m src.agent\n")

    try:
        main()
        print("✅ All tests completed!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
