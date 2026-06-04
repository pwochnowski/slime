# Q&A

## What's your understanding of the end goal here?

We have a GPU checkpoint mechanism that is theoretically more powerful than any currently available ones. Our goal is to find workflows which can benefit from our system.

The primary one that’s being targeted at the moment is colocated RLHF. 

The current work is focused entirely on integrating our GCR into SLIME, an existing RLHF library, to be able to obtain some measurements which can be used to illustrate the benefit of our solution.
Ultimately want to target a colocated single node deployment with arbitrary parallelism.

Another use case could be starting up container images, such as inference servers without paying any startup cost due to engine/device initialization.

---

## Where exactly does GCR（or IPC processing） in — replacing the existing checkpoint/weight-sync path, or sitting alongside it?

GCR replaces the existing memory offload path. In a colocated RLHF loop, training and inference alternate on the same GPUs, and whichever process is idle has its GPU memory reclaimed. 

Today this is primarily done with `torch_memory_saver` , an offload mechanism also hooking into the CuMem API. Due to requiring selective suspend/awake this solution requires application level integration, to tag specific memory allocations and allow partial offloading. The downside of this solution is that it can’t offload all the device state, specifically proprietary CUDA driver state, or 3rd-party library allocations, are forced to remain resident.

GCR replaces this with a single whole-process C/R primitive: when a process becomes idle, GCR checkpoints it (freeing 100% of its GPU memory), and when it's needed again, GCR restores it.

A key constraint is that GCR hooks the CUDA VMM API. It cannot coexist with existing VMM-level offload mechanism in the same process. This means GCR is a full replacement, not a supplement.

---

## Which components need changes？

There are two distinct problems blocking end-to-end integration. They are largely independent of each other.

### Problem 1: Legacy CUDA IPC must be replaced with VMM IPC

**The issue.** When transferring tensors (model weights) most applications use CUDA IPC handles (`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`). These legacy IPC calls only work on memory allocated via `cudaMalloc`. GCR's preload intercepts all `cudaMalloc` calls and transparently rewrites them to use the CUDA VMM API (`cuMemCreate` + `cuMemMap`). which will cause `cudaIpcGetMemHandle` to error out. Since GCR is not integrated into the application we can't selectively disable the rewrite for specific allocations, unlike current solutions. This will prevent integration into many existing apps, since this is the process used by torch to share tensors in a multiprocessing environment.

**The fix.** Replacing the IPC mechanism with VMM-native equivalents (`cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle`, with fd transport over a Unix domain socket) allows the weight transfer to work under GCR's preload. The data flow will be identical — same two GPU copies, same bucket-at-a-time transfer — just a different handle format. 

**The concerns.** This should not introduce any performance cost beyond the API-level overhead difference, since once the buffer is shared, access patterns are unchanged. There may be some optimization needed around buffer reuse if the current path relies on caching behaviour of the legacy handles, but the fundamentals are sound.

**Status.** This has been implemented in SGLang and integrated into SLIME.

### Problem 2: Weight sync forces both processes to be alive simultaneously

This is the more pressing problem. 

**The issue.** During weight sync, the trainer and inference engine must both be GPU-resident: the trainer holds the source weights, and the inference engine needs to receive them. However as mentioned above GCR cannot partially offload a process and so it becomes all-or-nothing and we can’t selectively pause and resume tagged memory regions. This means during the sync phase, neither process can be frozen, and both must fit in GPU memory simultaneously.

Different RL libraries handle this in different ways. `verl` opts for the multi-stage awake approach, which allows the weight sync to occur and the trainer process to be offloaded before the KV cache gets brought back. `SLIME` on the other hand partially mitigates this by offloading weights (and the rest of the unused trainer state) to CPU and then loading the weights into the inference process one bucket at a time, so the inference side only needs to leave enough space for a single bucket as well as the non-offloadable parts of the trainer process.

**Proposed solution: a lightweight syncer process.** Introduce a separate process whose sole job is coordinating weight sync. During training, the syncer is frozen alongside the inference engine. At the end of a training step, the trainer writes its updated weights to a shared CPU memory region (POSIX shm or memfd), then the trainer is frozen and the syncer is thawed. The syncer drives the bucket-by-bucket transfer to the inference engine using the shared CPU buffer as its source. Once sync completes, the syncer is frozen and inference proceeds.

The key insight is that the syncer replaces the trainer during the sync phase — it's never co-resident with the trainer on GPU. Its GPU footprint is minimal: just the NCCL communicator for the trainer↔inference sync group and a single-bucket scratch buffer. This gives the inference engine significantly more memory headroom than keeping the full trainer alive would.

The cost is a shared-memory handoff (the trainer must D2H weights into a region the syncer can read) and weight transformations (Megatron→HuggingFace naming/layout) must happen in the trainer before the handoff, since the syncer is deliberately Megatron-unaware.

**Status.** This is planned but not yet implemented. The first step (v0) will be to successfully use GCR to offload the idle trainer process after the weight step. That is we size the KV pool conservatively so that both processes fit simultaneously, in order to validate the GCR freeze/thaw lifecycle end-to-end first. The syncer process is scoped as a follow-on iteration once v0 is stable.

---

1.why need handle map？ maybe from VM to new handle could work？