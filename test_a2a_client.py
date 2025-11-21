#!/usr/bin/env python3
"""
Test client for the SOP Agent using A2A protocol.
"""

import asyncio
import json
import sys
import argparse

try:
    from a2a.client import A2AClient
except ImportError:
    print("Error: a2a-sdk not installed. Install with: pip install a2a-sdk[all]")
    sys.exit(1)


async def test_agent(debug: bool = False):
    """Test the SOP agent with sample queries."""

    agent_url = "http://localhost:9998"

    print(f"🔗 Connecting to SOP Agent at {agent_url}")
    print("=" * 60)

    client = A2AClient(agent_url)

    # Test queries
    test_queries = [
        "What is the return policy for electronics?",
        "How should I handle customer complaints?",
        "What are the store opening procedures?",
        "Tell me about the safety protocols",
    ]

    for idx, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {idx}: {query}")
        print("-" * 60)

        try:
            response = await client.send_message(
                text=query, thread_id=f"test_thread_{idx}"
            )

            if debug:
                print(f"\n🔍 Raw Response:\n{json.dumps(response, indent=2)}")

            # Extract text from response
            if isinstance(response, dict):
                if "parts" in response:
                    for part in response["parts"]:
                        if part.get("type") == "text":
                            print(
                                f"\n💬 Response:\n{part.get('text', 'No text found')}"
                            )
                elif "text" in response:
                    print(f"\n💬 Response:\n{response['text']}")
                else:
                    print(f"\n💬 Response:\n{response}")
            else:
                print(f"\n💬 Response:\n{response}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            if debug:
                import traceback

                traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ Test completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test the SOP Agent")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    try:
        asyncio.run(test_agent(debug=args.debug))
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
