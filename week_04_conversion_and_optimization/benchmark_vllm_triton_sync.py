import argparse
import time
from typing import List

import numpy as np
import tritonclient.http as httpclient


def build_input_tensor(batch_size: int, prompt: str, request_idx: int, cache_buster: bool):
    prompts = []
    for i in range(batch_size):
        text = prompt
        if cache_buster:
            text = f"{prompt}\n[request_id={request_idx}, sample={i}, ts={time.time_ns()}]"
        prompts.append(text)

    payload = np.array(prompts, dtype=object).reshape(batch_size, 1)
    infer_input = httpclient.InferInput("text_input", [batch_size, 1], "BYTES")
    infer_input.set_data_from_numpy(payload)
    return [infer_input]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton vLLM Python backend model")
    parser.add_argument("--url", type=str, default="localhost:8000", help="Triton HTTP endpoint host:port")
    parser.add_argument("--model-name", type=str, default="1p_generate_answers_vllm_infer", help="Triton model name")
    parser.add_argument("--prompt", type=str, default="Explain what Triton Inference Server is.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per request")
    parser.add_argument("--requests", type=int, default=50, help="Number of measured requests")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup requests before measuring")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="HTTP client timeout in seconds")
    parser.add_argument(
        "--no-cache-buster",
        action="store_true",
        help="Disable unique prompt suffix; useful if you want response caching effects",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.requests <= 0 or args.warmup < 0:
        raise ValueError("batch-size and requests must be > 0, warmup must be >= 0")

    cache_buster = not args.no_cache_buster
    print("Benchmark config:")
    print(f"- URL: {args.url}")
    print(f"- Model: {args.model_name}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Requests: {args.requests}")
    print(f"- Warmup: {args.warmup}")
    print("- Dispatch mode: burst async (send all requests immediately)")
    print(f"- Cache buster: {cache_buster}")

    # Warmup (single client thread).
    warmup_client = httpclient.InferenceServerClient(
        url=args.url,
        verbose=False,
        connection_timeout=args.timeout_s,
        network_timeout=args.timeout_s,
    )
    warmup_output = httpclient.InferRequestedOutput("text_output")
    for i in range(args.warmup):
        warmup_inputs = build_input_tensor(args.batch_size, args.prompt, -1000 - i, cache_buster=True)
        _ = warmup_client.infer(model_name=args.model_name, inputs=warmup_inputs, outputs=[warmup_output])

    client = httpclient.InferenceServerClient(
        url=args.url,
        verbose=False,
        connection_timeout=args.timeout_s,
        network_timeout=args.timeout_s,
    )
    requested_output = httpclient.InferRequestedOutput("text_output")

    pending = []
    start_bench = time.perf_counter()
    for req_idx in range(args.requests):
        inputs = build_input_tensor(args.batch_size, args.prompt, req_idx, cache_buster)
        request_start = time.perf_counter()
        async_req = client.async_infer(
            model_name=args.model_name,
            inputs=inputs,
            outputs=[requested_output],
            request_id=str(req_idx),
        )
        pending.append((req_idx, request_start, async_req))

    latencies_ms: List[float] = []
    errors: List[str] = []
    for req_idx, request_start, async_req in pending:
        try:
            _ = async_req.get_result()
            latency_ms = (time.perf_counter() - request_start) * 1000.0
            latencies_ms.append(latency_ms)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"request={req_idx}: {exc}")

    total_s = time.perf_counter() - start_bench

    ok_requests = len(latencies_ms)
    total_samples = ok_requests * args.batch_size
    rps = ok_requests / total_s if total_s > 0 else float("nan")
    sample_throughput = total_samples / total_s if total_s > 0 else float("nan")

    print("\nResults:")
    print(f"- Successful requests: {ok_requests}/{args.requests}")
    print(f"- Total wall time: {total_s:.3f} s")
    print(f"- Throughput (requests/s): {rps:.3f}")
    print(f"- Throughput (samples/s): {sample_throughput:.3f}")
    print(f"- Latency avg: {float(np.mean(latencies_ms)):.2f} ms" if latencies_ms else "- Latency avg: n/a")
    print(f"- Latency p50: {percentile(latencies_ms, 50):.2f} ms" if latencies_ms else "- Latency p50: n/a")
    print(f"- Latency p95: {percentile(latencies_ms, 95):.2f} ms" if latencies_ms else "- Latency p95: n/a")
    print(f"- Latency p99: {percentile(latencies_ms, 99):.2f} ms" if latencies_ms else "- Latency p99: n/a")

    if errors:
        print("\nErrors:")
        for err in errors[:10]:
            print(f"- {err}")
        if len(errors) > 10:
            print(f"- ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
