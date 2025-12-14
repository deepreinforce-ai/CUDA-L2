import torch
import threading
import random
from typing import Callable, Tuple, Any, Optional

# =============================================================================
# Store original CUDA functions at module load time (before any monkey-patching)
# =============================================================================
_original_elapsed_time = torch.cuda.Event.elapsed_time
_original_record = torch.cuda.Event.record
_original_synchronize = torch.cuda.synchronize


def defend_against_thread_injection(
    kernel: Callable,
    *args,
    **kwargs
) -> Tuple[bool, str, Any]:
    """
    Defense against thread injection attack.
    
    Thread injection spawns a background thread to do computation while
    returning an empty tensor immediately. This cheats timing but passes
    correctness checks since the thread finishes before verification.
    
    Defense: Compare thread count before and after kernel execution.
    
    Returns:
        (passed, message, output)
    """
    before = threading.active_count()
    output = kernel(*args, **kwargs)
    after = threading.active_count()
    
    if after > before:
        return False, "Kernel spawned background thread", output
    else:
        return True, "Thread injection check passed", output


def defend_against_stream_injection(
    kernel: Callable,
    *args,
    ratio_threshold: float = 2.0,
    **kwargs
) -> Tuple[bool, str, Any, Optional[float]]:
    """
    Defense against stream injection attack using hybrid approach.
    
    Stream injection runs the kernel on a separate CUDA stream, causing
    events recorded on the default stream to miss the actual computation time.
    
    Hybrid Defense:
    1. Run kernel with ORIGINAL timing (events on default stream, no extra sync)
    2. Run kernel with DEFENSE timing (sync all streams before end event)
    3. Compare: if (defense_time / original_time) > ratio_threshold, kernel is malicious
    4. If ratio is within threshold, use original timing (no overhead)
    
    Args:
        kernel: The kernel function to test
        *args: Arguments to pass to the kernel
        ratio_threshold: Maximum allowed ratio of defense_time/original_time (default 5.0)
        **kwargs: Keyword arguments to pass to the kernel
    
    Returns:
        (passed, message, output, timing_ms)
        - passed: True if no stream injection detected
        - message: Description of result
        - output: Kernel output
        - timing_ms: The timing to use (original if legit, defense if suspicious)
    """
    # Warmup
    _ = kernel(*args, **kwargs)
    torch.cuda.synchronize()
    
    # =========================================================================
    # RUN 10 ITERATIONS WITH RANDOM ORDER
    # =========================================================================
    num_iterations = 10
    original_times = []
    defense_times = []
    output = None
    
    for i in range(num_iterations):
        # Randomly decide which timing method runs first
        run_original_first = random.choice([True, False])
        
        def time_original():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = kernel(*args, **kwargs)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e), out
        
        def time_defense():
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = kernel(*args, **kwargs)
            torch.cuda.synchronize()  # Force sync ALL streams
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e), out
        
        if run_original_first:
            t, output = time_original()
            original_times.append(t)
            t, output = time_defense()
            defense_times.append(t)
        else:
            t, output = time_defense()
            defense_times.append(t)
            t, output = time_original()
            original_times.append(t)
    
    # Use median to reduce noise
    original_time_ms = sorted(original_times)[num_iterations // 2]
    defense_time_ms = sorted(defense_times)[num_iterations // 2]
    
    # =========================================================================
    # COMPARE: Large ratio indicates stream injection
    # =========================================================================
    # Avoid division by zero - if original_time is ~0, it's definitely suspicious
    if original_time_ms < 0.001:  # Less than 1µs
        ratio = float('inf')
    else:
        ratio = defense_time_ms / original_time_ms
    if ratio > ratio_threshold:
        return (
            False,
            f"Stream injection detected! Original: {original_time_ms:.3f}ms, "
            f"Defense: {defense_time_ms:.3f}ms, Ratio: {ratio:.1f}x (threshold: {ratio_threshold}x)",
            output,
            defense_time_ms  # Use defense timing for malicious kernels
        )
    else:
        return (
            True,
            f"Stream injection check passed. Ratio: {ratio:.2f}x (within {ratio_threshold}x threshold)",
            output,
            original_time_ms  # Use original timing (no overhead) for legit kernels
        )


def defend_against_lazy_evaluation(
    kernel: Callable,
    *args,
    expected_device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[bool, str, Any]:
    """
    Defense against lazy evaluation attack.
    
    Lazy evaluation returns a tensor subclass that defers computation until
    certain methods (like __eq__) are called. This cheats timing because
    the actual computation happens during correctness check, not during timing.
    
    Defense: Validate that output is a standard torch.Tensor with allocated memory.
    
    Args:
        kernel: The kernel function to test
        *args: Arguments to pass to the kernel
        expected_device: Expected device for output (inferred from first arg if None)
        **kwargs: Keyword arguments to pass to the kernel
    
    Returns:
        (passed, message, output)
    """
    output = kernel(*args, **kwargs)
    
    # Infer expected device from first input argument
    if expected_device is None:
        for arg in args:
            if isinstance(arg, torch.Tensor):
                expected_device = arg.device
                break
    
    if expected_device is None:
        expected_device = torch.device('cuda')
    
    # =========================================================================
    # Validate tensor is real and materialized, not a lazy hack
    # =========================================================================
    
    # Check 1: Must be a tensor
    if not isinstance(output, torch.Tensor):
        return False, f"Not a tensor: {type(output)}", output
    
    # Check 2: Must be standard torch.Tensor, not a subclass
    if type(output).__name__ not in ['Tensor', 'Parameter']:
        return False, f"Is {type(output).__name__}, not standard torch.Tensor", output
    
    # Check 3: Must be on correct device
    if output.device != expected_device:
        return False, f"Wrong device: {output.device} (expected {expected_device})", output
    
    # Check 4: Must have allocated storage
    storage_size = output.untyped_storage().size()
    if storage_size == 0:
        return False, "No allocated storage (likely lazy)", output
    
    # Check 5: Storage pointer must be valid
    ptr = output.data_ptr()
    if ptr == 0:
        return False, "Storage pointer is null (likely lazy)", output
    
    return True, "Lazy evaluation check passed", output


def defend_against_precision_downgrade(
    kernel: Callable,
    *args,
    expected_dtype: Optional[torch.dtype] = None,
    **kwargs
) -> Tuple[bool, str, Any]:
    """
    Defense against precision downgrade attack.
    
    Precision downgrade uses lower precision (e.g., float16 instead of float32)
    to speed up computation while sacrificing accuracy.
    
    Defense: Check if output dtype matches expected dtype.
    
    Args:
        kernel: The kernel function to test
        *args: Arguments to pass to the kernel
        expected_dtype: Expected dtype for output (inferred from first arg if None)
        **kwargs: Keyword arguments to pass to the kernel
    
    Returns:
        (passed, message, output)
    """
    output = kernel(*args, **kwargs)
    
    # Infer expected dtype from first input tensor
    if expected_dtype is None:
        for arg in args:
            if isinstance(arg, torch.Tensor):
                expected_dtype = arg.dtype
                break
    
    if expected_dtype is None:
        expected_dtype = torch.float32
    
    # Check dtype
    if output.dtype != expected_dtype:
        return False, f"Precision downgrade detected: output is {output.dtype}, expected {expected_dtype}", output
    
    return True, "Precision downgrade check passed", output


def defend_against_elapsed_time_monkey_patching() -> Tuple[bool, str]:
    """
    Defense against monkey-patching attack. 😭
    
    Monkey-patching replaces critical CUDA timing functions with fake versions
    that return instant times, making any kernel appear fast.
    
    Defense: Check if torch.cuda.Event.elapsed_time, record, or synchronize
    have been modified from their original implementations.
    
    Returns:
        (passed, message)
    """
    patched = []
    
    # Check elapsed_time
    if torch.cuda.Event.elapsed_time is not _original_elapsed_time:
        patched.append("torch.cuda.Event.elapsed_time")
    
    # Check record
    if torch.cuda.Event.record is not _original_record:
        patched.append("torch.cuda.Event.record")
    
    # Check synchronize
    if torch.cuda.synchronize is not _original_synchronize:
        patched.append("torch.cuda.synchronize")
    
    if patched:
        return False, f"Monkey-patching detected: {', '.join(patched)}"
    else:
        return True, "Monkey-patching check passed"


def run_all_defenses(
    kernel: Callable,
    *args,
    **kwargs
) -> Tuple[bool, list, Any]:
    """
    Run all defense checks against kernel attacks.
    
    Args:
        kernel: The kernel function to test
        *args: Arguments to pass to the kernel
        **kwargs: Keyword arguments to pass to the kernel
    
    Returns:
        (all_passed, results, output)
        - all_passed: True if all defense checks passed
        - results: List of (defense_name, passed, message) tuples
        - output: The kernel output (if any check succeeded)
    """
    results = []
    output = None
    
    # Defense 1: Stream injection (hybrid approach)
    passed, message, output, timing = defend_against_stream_injection(kernel, *args, **kwargs)
    results.append(("stream_injection", passed, message))
    
    # Defense 2: Thread injection
    passed, message, output = defend_against_thread_injection(kernel, *args, **kwargs)
    results.append(("thread_injection", passed, message))
    
    # Defense 3: Lazy evaluation
    passed, message, output = defend_against_lazy_evaluation(kernel, *args, **kwargs)
    results.append(("lazy_evaluation", passed, message))
    
    # Defense 4: Precision downgrade
    passed, message, output = defend_against_precision_downgrade(kernel, *args, **kwargs)
    results.append(("precision_downgrade", passed, message))
    
    # Defense 5: Elapsed time monkey patching (doesn't need kernel execution)
    passed, message = defend_against_elapsed_time_monkey_patching()
    results.append(("elapsed_time_monkey_patching", passed, message))
    
    all_passed = all(r[1] for r in results)
    
    return all_passed, results, output


# =============================================================================
# Testing
# =============================================================================

def legit_kernel(A, B):
    return torch.matmul(A, B)

def test_thread_injection():
    """Test the thread injection defense with both legitimate and malicious kernels."""
        
    # Malicious kernel - spawns background thread
    def malicious_kernel(A, B):
        out = torch.empty(A.size(0), B.size(1), device=A.device)
        
        def compute():
            result = torch.matmul(A, B)
            out.copy_(result)
        
        t = threading.Thread(target=compute)
        t.start()
        return out
    
    # Setup
    A = torch.randn(1000, 1000).cuda()
    B = torch.randn(1000, 1000).cuda()
    torch.cuda.synchronize()
    
    print("=" * 50)
    print("Testing legitimate kernel:")
    print("=" * 50)
    passed, msg, output = defend_against_thread_injection(legit_kernel, A, B)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {msg}")
    
    print()
    print("=" * 50)
    print("Testing malicious kernel (thread injection):")
    print("=" * 50)
    passed, msg, output = defend_against_thread_injection(malicious_kernel, A, B)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {msg}")


def test_stream_injection():
    """Test the stream injection defense with both legitimate and malicious kernels."""
    
    # Malicious kernel - runs on separate stream (stream injection)
    def malicious_kernel(A, B):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            return torch.matmul(A, B)
    
    # Setup
    A = torch.randn(10000, 10000).cuda()
    B = torch.randn(10000, 10000).cuda()
    torch.cuda.synchronize()
    
    print("=" * 50)
    print("Testing legitimate kernel:")
    print("=" * 50)
    passed, msg, output, timing = defend_against_stream_injection(legit_kernel, A, B)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {msg}")
    
    print()
    print("=" * 50)
    print("Testing malicious kernel (stream injection):")
    print("=" * 50)
    passed, msg, output, timing = defend_against_stream_injection(malicious_kernel, A, B)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {msg}")


def test_lazy_evaluation():
    """Test the lazy evaluation defense with both legitimate and malicious kernels."""
    
    # Malicious kernel - returns lazy tensor subclass
    class LazyMatmul(torch.Tensor):
        @staticmethod
        def __new__(cls, A, B):
            obj = torch.Tensor._make_subclass(cls, torch.empty(A.size(0), B.size(1), device=A.device))
            obj.A, obj.B = A, B
            return obj
        
        def __eq__(self, other):
            return torch.matmul(self.A, self.B) == other
    
    def malicious_kernel(A, B):
        return LazyMatmul(A, B)
    
    # Setup
    A = torch.randn(1000, 1000).cuda()
    B = torch.randn(1000, 1000).cuda()
    torch.cuda.synchronize()
    
    print("=" * 50)
    print("Testing legitimate kernel:")
    print("=" * 50)
    passed, msg, output = defend_against_lazy_evaluation(legit_kernel, A, B)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {msg}")
    
    print()
    print("=" * 50)
    print("Testing malicious kernel (lazy evaluation):")
    print("=" * 50)
    passed, msg, output = defend_against_lazy_evaluation(malicious_kernel, A, B)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {msg}")


def test_precision_downgrade():
    """Test the precision downgrade defense with both legitimate and malicious kernels."""
    
    # Malicious kernel - downgrades to float16
    def malicious_kernel(A, B):
        return torch.matmul(A.half(), B.half())  # Downgrade to float16
    
    # Setup (float32 inputs)
    A = torch.randn(1000, 1000).cuda()
    B = torch.randn(1000, 1000).cuda()
    torch.cuda.synchronize()
    
    print("=" * 50)
    print("Testing legitimate kernel:")
    print("=" * 50)
    passed, msg, output = defend_against_precision_downgrade(legit_kernel, A, B)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {msg}")
    
    print()
    print("=" * 50)
    print("Testing malicious kernel (precision downgrade):")
    print("=" * 50)
    passed, msg, output = defend_against_precision_downgrade(malicious_kernel, A, B)
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {msg}")


def test_elapsed_time_monkey_patching():
    """Test the monkey-patching defense."""
    
    print("=" * 50)
    print("Testing before monkey-patching:")
    print("=" * 50)
    passed, msg = defend_against_elapsed_time_monkey_patching()
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {msg}")
    
    print()
    print("=" * 50)
    print("Testing after monkey-patching elapsed_time:")
    print("=" * 50)
    
    # Apply monkey-patch 😭
    def _fake_elapsed_time(self, end_event):
        return 0.001  # Fake fast!
    
    torch.cuda.Event.elapsed_time = _fake_elapsed_time
    
    passed, msg = defend_against_elapsed_time_monkey_patching()
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {msg}")
    
    # Restore original
    torch.cuda.Event.elapsed_time = _original_elapsed_time
    
    print()
    print("=" * 50)
    print("Testing after restoring original:")
    print("=" * 50)
    passed, msg = defend_against_elapsed_time_monkey_patching()
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {msg}")


def test_run_all_defenses():
    """Test run_all_defenses with legitimate and various malicious kernels."""
    
    # Setup
    A = torch.randn(1000, 1000).cuda()
    B = torch.randn(1000, 1000).cuda()
    torch.cuda.synchronize()
    
    print("=" * 60)
    print("Testing legitimate kernel with ALL defenses:")
    print("=" * 60)
    all_passed, results, output = run_all_defenses(legit_kernel, A, B)
    for name, passed, msg in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}: {msg}")
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    
    print("\n")
    print("=" * 60)
    print("Testing stream injection attack with ALL defenses:")
    print("=" * 60)
    def stream_attack(A, B):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            return torch.matmul(A, B)
    
    all_passed, results, output = run_all_defenses(stream_attack, A, B)
    for name, passed, msg in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}: {msg}")
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    
    print("\n")
    print("=" * 60)
    print("Testing thread injection attack with ALL defenses:")
    print("=" * 60)
    def thread_attack(A, B):
        out = torch.empty(A.size(0), B.size(1), device=A.device)
        def compute():
            result = torch.matmul(A, B)
            out.copy_(result)
        t = threading.Thread(target=compute)
        t.start()
        return out
    
    all_passed, results, output = run_all_defenses(thread_attack, A, B)
    for name, passed, msg in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}: {msg}")
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    
    print("\n")
    print("=" * 60)
    print("Testing precision downgrade attack with ALL defenses:")
    print("=" * 60)
    def precision_attack(A, B):
        return torch.matmul(A.half(), B.half())
    
    all_passed, results, output = run_all_defenses(precision_attack, A, B)
    for name, passed, msg in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}: {msg}")
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")


if __name__ == "__main__":
    # test_thread_injection()
    # test_stream_injection()
    # test_lazy_evaluation()
    # test_precision_downgrade()
    # test_elapsed_time_monkey_patching()
    test_run_all_defenses()
