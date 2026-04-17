"""
Minimal reproduction of the VMM IPC tensor lifecycle segfault.

The crash in production occurs at model_runner.py:1537 when views of a
VMM-backed tensor are deleted after free_vmm_buffer unmaps the VA range.
This test exercises the same sequence without needing SGLang or Ray.

Requires: a CUDA GPU and the cuda-python bindings (cuda.bindings).
Run:  python tests/test_vmm_ipc_lifecycle.py
"""

import gc
import sys

import torch

from sglang.srt.weight_sync.vmm_ipc import (
    alloc_vmm_buffer,
    free_vmm_buffer,
    wrap_as_torch_uint8,
)

DEVICE = 0
BUF_SIZE = 4 * 1024 * 1024  # 4 MiB


def test_del_buf_then_free():
    """Producer path: del buf, then free_vmm_buffer.  Should not segfault."""
    torch.cuda.set_device(DEVICE)
    alloc = alloc_vmm_buffer(BUF_SIZE, DEVICE)
    buf = wrap_as_torch_uint8(alloc)
    buf[:1024].fill_(42)
    torch.cuda.synchronize(DEVICE)

    del buf
    gc.collect()
    free_vmm_buffer(alloc, close_fd=True)
    print("PASS: test_del_buf_then_free")


def test_del_views_then_buf_then_free():
    """Consumer path: create views, delete views, delete buf, then free.

    This mirrors model_runner.py:1537:
        del named_tensors, t, buf
        free_vmm_buffer(alloc, close_fd=True)
    """
    torch.cuda.set_device(DEVICE)
    alloc = alloc_vmm_buffer(BUF_SIZE, DEVICE)
    buf = wrap_as_torch_uint8(alloc)

    # Simulate reconstructing named tensors as views of buf
    views = [
        buf[0:1024].view(torch.float32).reshape(256),
        buf[1024:2048].view(torch.float16).reshape(512),
        buf[2048:4096].view(torch.bfloat16).reshape(1024),
    ]
    named_tensors = [("w1", views[0]), ("w2", views[1]), ("w3", views[2])]

    # Simulate load_weights: copy into separate tensors
    for name, t in named_tensors:
        _ = t.clone()
    torch.cuda.synchronize(DEVICE)

    # This is the exact sequence from model_runner.py:1537-1538
    del named_tensors, views, t, buf
    gc.collect()
    free_vmm_buffer(alloc, close_fd=True)
    print("PASS: test_del_views_then_buf_then_free")


def test_free_before_buf_del():
    """Demonstrates the crash: free_vmm_buffer BEFORE buf is destroyed.

    If buf (or any view) still has references when free_vmm_buffer unmaps
    the VA range, the later tensor destructor touches unmapped memory.
    """
    torch.cuda.set_device(DEVICE)
    alloc = alloc_vmm_buffer(BUF_SIZE, DEVICE)
    buf = wrap_as_torch_uint8(alloc)
    buf[:1024].fill_(42)
    torch.cuda.synchronize(DEVICE)

    # Keep an extra reference (simulates stale ref in tuple, GC cycle, etc.)
    stale_ref = buf

    del buf
    gc.collect()
    # buf is NOT destroyed because stale_ref still holds it.
    # free_vmm_buffer unmaps the VA range.
    free_vmm_buffer(alloc, close_fd=True)

    # Now dropping stale_ref triggers StorageImpl::~StorageImpl() on
    # unmapped memory -> segfault.
    print("About to drop stale_ref (may segfault)...")
    del stale_ref
    gc.collect()
    print("PASS: test_free_before_buf_del (no segfault)")


def test_view_outlives_buf():
    """A view outliving buf, with free_vmm_buffer in between.

    This is the likely production scenario: named_tensors (views of buf)
    are not fully dead when free_vmm_buffer fires.
    """
    torch.cuda.set_device(DEVICE)
    alloc = alloc_vmm_buffer(BUF_SIZE, DEVICE)
    buf = wrap_as_torch_uint8(alloc)

    view = buf[0:1024].view(torch.float32).reshape(256)

    del buf
    gc.collect()
    # buf is destroyed, but view shares the same storage.
    # Storage refcount > 0, so StorageImpl is NOT destroyed yet.
    # _vmm_dlpack_refs was on buf.__dict__ -- is it still alive?
    print(f"  view storage data_ptr: {view.storage().data_ptr():#x}")
    print(f"  view has _vmm_dlpack_refs: {hasattr(view, '_vmm_dlpack_refs')}")

    # Now unmap the VA range.
    free_vmm_buffer(alloc, close_fd=True)

    # Dropping the view triggers StorageImpl destruction on unmapped VA.
    print("About to drop view (may segfault)...")
    del view
    gc.collect()
    print("PASS: test_view_outlives_buf (no segfault)")


if __name__ == "__main__":
    tests = [
        test_del_buf_then_free,
        test_del_views_then_buf_then_free,
        test_free_before_buf_del,
        test_view_outlives_buf,
    ]

    for test in tests:
        print(f"\n--- {test.__name__} ---")
        sys.stdout.flush()
        try:
            test()
        except Exception as e:
            print(f"EXCEPTION: {e}")
        sys.stdout.flush()
