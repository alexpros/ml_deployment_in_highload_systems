import argparse
import threading
import time
from typing import Dict, List

import numpy as np
import tritonclient.grpc as grpcclient
import queue
from functools import partial
from tritonclient.utils import *


def build_input_tensor(batch_size: int, prompt: str, request_idx: int, cache_buster: bool):
    prompts = []
    for i in range(batch_size):
        text = prompt
        if cache_buster:
            text = f"{prompt}\n[request_id={request_idx}, sample={i}, ts={time.time_ns()}]"
        prompts.append(text)

    payload = np.array(prompts, dtype=np.object_)
    infer_input = grpcclient.InferInput("text_input", [ 1], "BYTES")
    infer_input.set_data_from_numpy(payload)
    return [infer_input]


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton vLLM model via gRPC stream infer")
    parser.add_argument("--url", type=str, default="localhost:8001", help="Triton gRPC endpoint host:port")
    parser.add_argument("--model-name", type=str, default="1p_generate_answers_vllm_infer", help="Triton model name")
    parser.add_argument("--prompt", type=str, default="Explain what Triton Inference Server is.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per request")
    parser.add_argument("--requests", type=int, default=50, help="Number of measured requests")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Timeout waiting stream responses")
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
    print("- Dispatch mode: gRPC stream infer burst (send all requests immediately)")
    print(f"- Cache buster: {cache_buster}")


    client = grpcclient.InferenceServerClient(url=args.url, verbose=False)
    requested_output = grpcclient.InferRequestedOutput("text_output")

    errors: List[str] = []

    class UserData:
        def __init__(self):
            self._completed_requests = queue.Queue()

    def callback(user_data, result, error):
        if error:
            user_data._completed_requests.put(error)
        else:
            user_data._completed_requests.put(result)
    user_data = UserData()
    client.start_stream(callback=partial(callback, user_data))

    start_bench = time.perf_counter()

    for req_idx in range(args.requests):
        inputs = build_input_tensor(args.batch_size, args.prompt, req_idx, cache_buster)
        req_id = str(req_idx)
        client.async_stream_infer(
            model_name=args.model_name,
            inputs=inputs,
            outputs=[requested_output],
            request_id=req_id,
        )
    
    recv_count = 0
    expected_count = args.requests

    try:
        while recv_count < expected_count:
            data_item = user_data._completed_requests.get()
            
            if type(data_item) == InferenceServerException:
                print(data_item)
            
            recv_count += 1
    except KeyboardInterrupt:
        client.stop_stream(cancel_requests=True)
    finally:
        client.close()

    client.stop_stream()

    total_s = time.perf_counter() - start_bench

    ok_requests = args.requests
    total_samples = ok_requests * args.batch_size
    rps = ok_requests / total_s if total_s > 0 else float("nan")
    sample_throughput = total_samples / total_s if total_s > 0 else float("nan")

    print("\nResults:")
    print(f"- Successful requests: {ok_requests}/{args.requests}")
    print(f"- Total wall time: {total_s:.3f} s")
    print(f"- Throughput (requests/s): {rps:.3f}")
    print(f"- Throughput (samples/s): {sample_throughput:.3f}")

    if errors:
        print("\nErrors:")
        for err in errors[:10]:
            print(f"- {err}")
        if len(errors) > 10:
            print(f"- ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
