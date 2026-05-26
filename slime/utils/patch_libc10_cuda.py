"""
Binary-patch libc10_cuda.so to disable exportable IPC handles for expandable segments.

PyTorch 2.9.1 unconditionally sets requestedHandleTypes on cuMemCreate for expandable
segments. This patch NOPs out the `mov %eax, 0x44(%rsp)` instruction that writes the
handle type into CUmemAllocationProp, leaving requestedHandleTypes = 0 (non-exportable).

The patched copy is written next to the original as libc10_cuda.so.no_ipc.
Use LD_PRELOAD with the patched copy to load it instead of the original.
"""

import logging
import os
import shutil

logger = logging.getLogger(__name__)

# Each entry: (offset, original_bytes, patched_bytes, description)
# We support multiple torch builds by trying each patch profile in order.
_PATCH_PROFILES = [
    {
        "name": "torch-2.9.1+cu129 (system)",
        # In ExpandableSegment::map(), the instruction:
        #   2d4a2: 89 44 24 44   mov %eax, 0x44(%rsp)  ; prop.requestedHandleTypes = eax
        # is replaced with NOPs so requestedHandleTypes stays 0 (set by pxor+movups above).
        "offset": 0x2D4A2,
        "original": bytes([0x89, 0x44, 0x24, 0x44]),
        "patched": bytes([0x90, 0x90, 0x90, 0x90]),
        # Context: 5 bytes before the patch point for verification
        "context_offset": 0x2D49A,
        "context": bytes([0xC7, 0x44, 0x24, 0x24, 0x00, 0x00, 0x00, 0x00, 0x89, 0x44, 0x24, 0x44]),
    },
    {
        "name": "torch-2.9.1+cu129 (venv, with enable_ipc_handles)",
        # In ExpandableSegment::map(), after checking enable_ipc_handles:
        #   246da: cmpb $0x0, enable_ipc_handles
        #   246e1: 74 1c   je +0x1c   ← skip requestedHandleTypes if false
        # Change je (0x74) to jmp (0xeb) so it ALWAYS skips.
        "offset": 0x246E1,
        "original": bytes([0x74, 0x1C]),
        "patched": bytes([0xEB, 0x1C]),
        "context_offset": 0x246DE,
        "context": bytes([0x04, 0x00, 0x00, 0x74, 0x1C, 0xE8, 0xC8]),
    },
]


def _find_torch_libc10_cuda() -> str:
    import torch
    return os.path.join(os.path.dirname(torch.__file__), "lib", "libc10_cuda.so")


def _match_profile(data: bytes, profile: dict) -> bool:
    off = profile["context_offset"]
    ctx = profile["context"]
    return data[off : off + len(ctx)] == ctx


def create_patched_library(source: str | None = None, dest: str | None = None) -> str:
    if source is None:
        source = _find_torch_libc10_cuda()
    if dest is None:
        dest = source + ".no_ipc"

    with open(source, "rb") as f:
        data = bytearray(f.read())

    profile = None
    for p in _PATCH_PROFILES:
        if _match_profile(data, p):
            profile = p
            break

    if profile is None:
        snippets = "; ".join(
            f"{p['name']}: expected {p['context'].hex()} at 0x{p['context_offset']:X}, "
            f"got {data[p['context_offset']:p['context_offset']+len(p['context'])].hex()}"
            for p in _PATCH_PROFILES
        )
        raise RuntimeError(
            f"Cannot patch {source}: no matching byte pattern found. Tried: {snippets}"
        )

    off = profile["offset"]
    orig = profile["original"]
    patch = profile["patched"]

    if os.path.exists(dest):
        with open(dest, "rb") as f:
            existing = f.read()
        if existing[off : off + len(patch)] == patch and _match_profile(existing, profile):
            logger.info("Patched library already exists: %s (%s)", dest, profile["name"])
            return dest
        logger.warning("Stale patched library at %s, regenerating", dest)

    if data[off : off + len(orig)] == patch:
        logger.info("Source library is already patched (%s)", profile["name"])
        shutil.copy2(source, dest)
        return dest

    if data[off : off + len(orig)] != orig:
        raise RuntimeError(
            f"Unexpected bytes at patch offset 0x{off:X}: "
            f"expected {orig.hex()}, got {data[off:off+len(orig)].hex()}"
        )

    data[off : off + len(patch)] = patch

    with open(dest, "wb") as f:
        f.write(data)
    shutil.copymode(source, dest)

    logger.info(
        "Created patched libc10_cuda.so at %s (%s, %d bytes at 0x%X)",
        dest, profile["name"], len(patch), off,
    )
    return dest


def get_ld_preload_value(patched_path: str) -> str:
    existing = os.environ.get("LD_PRELOAD", "")
    if existing:
        return f"{patched_path}:{existing}"
    return patched_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = create_patched_library()
    print(f"Patched library: {path}")
    print(f"LD_PRELOAD={get_ld_preload_value(path)}")
