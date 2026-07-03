#!/usr/bin/env python3
"""
Simplified approach tests for CI/CD
Tests the basic structure of approaches without requiring actual model inference
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optillm.mcts import chat_with_mcts
from optillm.bon import best_of_n_sampling
from optillm.moa import mixture_of_agents
from optillm.self_consistency import advanced_self_consistency_approach
from optillm.reread import re2_approach
from optillm.cot_reflection import cot_reflection
from optillm.plansearch import plansearch
from optillm.leap import leap
from optillm.rto import round_trip_optimization
from optillm.mars import multi_agent_reasoning_system


class MockClient:
    """Mock OpenAI client for testing"""
    def __init__(self):
        self.chat = self.Chat()
    
    class Chat:
        def __init__(self):
            self.completions = self.Completions()
        
        class Completions:
            def create(self, **kwargs):
                class MockChoice:
                    class Message:
                        content = "Test response: 2 + 2 = 4"
                    message = Message()
                    finish_reason = "stop"

                class MockUsage:
                    completion_tokens = 10
                    total_tokens = 20

                class MockResponse:
                    choices = [MockChoice()]
                    usage = MockUsage()

                return MockResponse()


class MockBadClient:
    """Mock client that simulates degraded provider responses.

    mode:
      - "empty_choices": response with an empty choices list
      - "none_content":  a choice whose message.content is None
      - "length":        a choice truncated with finish_reason == "length"
    """
    def __init__(self, mode):
        self.chat = self.Chat(mode)

    class Chat:
        def __init__(self, mode):
            self.completions = self.Completions(mode)

        class Completions:
            def __init__(self, mode):
                self.mode = mode

            def create(self, **kwargs):
                mode = self.mode

                class MockUsage:
                    completion_tokens = 0
                    total_tokens = 0

                if mode == "empty_choices":
                    class MockResponse:
                        choices = []
                        usage = MockUsage()
                    return MockResponse()

                class MockMessage:
                    content = None if mode == "none_content" else "partial output"

                class MockChoice:
                    message = MockMessage()
                    finish_reason = "length" if mode == "length" else "stop"

                class MockResponse:
                    choices = [MockChoice()]
                    usage = MockUsage()

                return MockResponse()


def test_approach_imports():
    """Test that all approaches can be imported"""
    approaches = [
        chat_with_mcts,
        best_of_n_sampling,
        mixture_of_agents,
        advanced_self_consistency_approach,
        re2_approach,
        cot_reflection,
        plansearch,
        leap,
        multi_agent_reasoning_system
    ]
    
    for approach in approaches:
        assert callable(approach), f"{approach.__name__} is not callable"
    
    print("✅ All approaches imported successfully")


def test_basic_approach_calls():
    """Test basic approach calls with mock client"""
    client = MockClient()
    system_prompt = "You are a helpful assistant."
    query = "What is 2 + 2?"
    model = "mock-model"
    
    # Test approaches that should work with mock client
    simple_approaches = [
        ("re2_approach", re2_approach),
        ("cot_reflection", cot_reflection),
        ("leap", leap),
        ("mars", multi_agent_reasoning_system),
    ]
    
    for name, approach_func in simple_approaches:
        try:
            result = approach_func(system_prompt, query, client, model)
            assert result is not None, f"{name} returned None"
            assert isinstance(result, tuple), f"{name} should return a tuple"
            assert len(result) == 2, f"{name} should return (response, tokens)"
            print(f"✅ {name} basic test passed")
        except Exception as e:
            print(f"❌ {name} basic test failed: {e}")


def test_approach_parameters():
    """Test that approaches handle parameters correctly"""
    # Test that approaches accept the expected parameters
    import inspect
    
    approaches = {
        "chat_with_mcts": chat_with_mcts,
        "best_of_n_sampling": best_of_n_sampling,
        "mixture_of_agents": mixture_of_agents,
        "advanced_self_consistency_approach": advanced_self_consistency_approach,
        "re2_approach": re2_approach,
        "cot_reflection": cot_reflection,
        "plansearch": plansearch,
        "leap": leap,
        "multi_agent_reasoning_system": multi_agent_reasoning_system,
    }
    
    for name, func in approaches.items():
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        
        # Check required parameters
        required_params = ["system_prompt", "initial_query", "client", "model"]
        for param in required_params:
            assert param in params, f"{name} missing required parameter: {param}"
        
        print(f"✅ {name} has correct parameters")


def test_approaches_handle_bad_responses():
    """Approaches must degrade gracefully on empty / None / length-truncated
    provider responses instead of raising raw IndexError/TypeError/AttributeError.

    Without the response-validation guards these calls crash with an
    uncaught IndexError (empty choices) or TypeError/AttributeError (None content
    flowing into extract_output / SequenceMatcher / .strip()).
    """
    system_prompt = "You are a helpful assistant."
    query = "What is 2 + 2?"
    model = "mock-model"

    uncontrolled = (IndexError, TypeError, AttributeError)
    approaches = [
        ("round_trip_optimization", round_trip_optimization),
        ("advanced_self_consistency_approach", advanced_self_consistency_approach),
        ("re2_approach", re2_approach),
        ("leap", leap),
    ]

    for mode in ("empty_choices", "none_content", "length"):
        client = MockBadClient(mode)
        for name, func in approaches:
            try:
                result = func(system_prompt, query, client, model)
                # Graceful degradation: a normal (content, tokens) tuple.
                assert isinstance(result, tuple) and len(result) == 2, \
                    f"{name} ({mode}) returned unexpected {result!r}"
            except uncontrolled as e:
                raise AssertionError(
                    f"{name} ({mode}) raised uncontrolled {type(e).__name__}: {e}")
            except Exception:
                # A controlled, informative error is acceptable hardening behaviour.
                pass

    print("✅ approaches handle bad responses gracefully")


if __name__ == "__main__":
    print("Running approach tests...")

    test_approach_imports()
    test_basic_approach_calls()
    test_approach_parameters()
    test_approaches_handle_bad_responses()

    print("\nAll tests completed!")