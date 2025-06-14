import asyncio
import aiohttp
import json
import time
from typing import List, Dict

async def send_request(session: aiohttp.ClientSession, request_id: int, prompt: str) -> Dict:
    """Send a single request to the vLLM server"""
    url = "http://localhost:8000/v1/chat/completions"

    payload = {
        "model": "gaunernst/gemma-3-12b-it-qat-compressed-tensors",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "stream": False
    }

    headers = {
        "Content-Type": "application/json"
    }

    start_time = time.time()

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            end_time = time.time()

            if response.status == 200:
                result = await response.json()
                return {
                    "request_id": request_id,
                    "status": "success",
                    "response_time": end_time - start_time,
                    "prompt": prompt,
                    "response": result["choices"][0]["message"]["content"],
                    "usage": result.get("usage", {})
                }
            else:
                error_text = await response.text()
                return {
                    "request_id": request_id,
                    "status": "error",
                    "response_time": end_time - start_time,
                    "prompt": prompt,
                    "error": f"HTTP {response.status}: {error_text}"
                }

    except Exception as e:
        end_time = time.time()
        return {
            "request_id": request_id,
            "status": "error",
            "response_time": end_time - start_time,
            "prompt": prompt,
            "error": str(e)
        }

async def run_parallel_requests(prompts: List[str]) -> List[Dict]:
    """Run multiple requests in parallel"""

    # Create connector with higher limits for 100 requests
    connector = aiohttp.TCPConnector(
        limit=50,  # Total connection limit (increased)
        limit_per_host=50  # Per-host connection limit (increased)
    )

    timeout = aiohttp.ClientTimeout(total=120)  # 120 second timeout (increased for load)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create tasks for all requests
        tasks = [
            send_request(session, i, prompt)
            for i, prompt in enumerate(prompts)
        ]

        # Run all requests concurrently
        print(f"Sending {len(tasks)} requests in parallel...")
        start_time = time.time()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        print(f"All requests completed in {end_time - start_time:.2f} seconds")

        return results

def print_results(results: List[Dict]):
    """Print formatted results"""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    failed = [r for r in results if isinstance(r, dict) and r.get("status") == "error"]
    exceptions = [r for r in results if not isinstance(r, dict)]

    print(f"Successful requests: {len(successful)}")
    print(f"Failed requests: {len(failed)}")
    print(f"Exceptions: {len(exceptions)}")

    if successful:
        avg_response_time = sum(r["response_time"] for r in successful) / len(successful)
        min_response_time = min(r["response_time"] for r in successful)
        max_response_time = max(r["response_time"] for r in successful)

        print(f"Average response time: {avg_response_time:.2f} seconds")
        print(f"Min response time: {min_response_time:.2f} seconds")
        print(f"Max response time: {max_response_time:.2f} seconds")

        # Calculate throughput
        total_time = max(r["response_time"] for r in results if isinstance(r, dict))
        throughput = len(successful) / total_time if total_time > 0 else 0
        print(f"Effective throughput: {throughput:.2f} requests/second")

        # Show token usage if available
        total_prompt_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in successful)
        total_completion_tokens = sum(r.get("usage", {}).get("completion_tokens", 0) for r in successful)

        if total_prompt_tokens > 0:
            print(f"Total prompt tokens: {total_prompt_tokens}")
            print(f"Total completion tokens: {total_completion_tokens}")
            print(f"Total tokens: {total_prompt_tokens + total_completion_tokens}")

            # Token throughput
            if total_time > 0:
                token_throughput = total_completion_tokens / total_time
                print(f"Token generation throughput: {token_throughput:.2f} tokens/second")

    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)

    for result in results:
        if isinstance(result, dict):
            print(f"\nRequest {result.get('request_id', 'Unknown')}:")
            print(f"  Status: {result.get('status', 'Unknown')}")
            print(f"  Response time: {result.get('response_time', 0):.2f}s")
            print(f"  Prompt: {result.get('prompt', '')[:50]}...")

            if result.get("status") == "success":
                response_preview = result.get("response", "")[:1000]
                print(f"  Response: {response_preview}{'...' if len(result.get('response', '')) > 100 else ''}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"\nException occurred: {result}")

async def main():
    # Define base test prompts
    base_prompts = [
        "You are given the following text:"
        "Autism is a neurodevelopmental disorder characterized by impaired social interaction, impaired verbal and non-verbal communication, and restricted and repetitive behavior. Parents usually notice signs in the first two years of their child's life. These signs often develop gradually, though some children with autism reach their developmental milestones at a normal pace and then regress. The diagnostic criteria require that symptoms become apparent in early childhood, typically before age three."
        "Please extract all relationships (subject, object, summary of relationship)"
    ]

    # Create 100 prompts by cycling through base prompts with variations
    prompts = []
    for i in range(500):
        base_prompt = base_prompts[i % len(base_prompts)]

        # Add slight variations to make requests unique
        if i >= len(base_prompts):
            variation_num = (i // len(base_prompts)) + 1
            if variation_num == 2:
                base_prompt = f"Can you explain {base_prompt.lower()}"
            elif variation_num == 3:
                base_prompt = f"Please describe {base_prompt.lower()}"
            elif variation_num == 4:
                base_prompt = f"Tell me about {base_prompt.lower()}"

        prompts.append(base_prompt)

    print("Starting parallel requests to vLLM server...")
    print(f"Server URL: http://localhost:8000")
    print(f"Model: gaunernst/gemma-3-12b-it-qat-compressed-tensors")
    print(f"Number of requests: {len(prompts)}")

    # Run the requests
    results = await run_parallel_requests(prompts)

    # Print results
    print_results(results)

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import aiohttp
    except ImportError:
        print("Please install aiohttp: pip install aiohttp")
        exit(1)

    # Run the async main function
    asyncio.run(main())