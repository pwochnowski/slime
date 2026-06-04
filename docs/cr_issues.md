# C/R issues with Shared-Memory

To reclaim GPU memory between phases, we use a GPU checkpoint/restore mechanism that intercepts all CUDA allocations via the VMM API (cuMemCreate / cuMemMap). When a process is suspended, GCR offloads all its GPU memory to CPU and unmaps it; when resumed, it restores memory at the same virtual addresses leaving all stateful GPU resources intact.

This ownership model creates the dependency described below: suspending SGL also offloads the shared memory, leaving MT unable to resume independently.

## Problem

Phase 0: rollout (only SGL alive)

suspend SGL       	← SGL's GPU memory offloaded
resume MT           	← MT needs SGL's memory to remap imports

Phase 1: train (only MT alive)

resume SGL

Phase 2: weight-sync (both alive)	← shared memory between MT and SGL

suspend MT        	← MT's GPU memory offloaded

During weight-sync, SGL allocates a VMM buffer, exports a shareable handle, and MT imports it so both processes see the same physical GPU memory. These shared handles persist across training iterations to avoid per-step setup cost and SGL is considered the “owner”. Consequently when SGL is suspended they are also offloaded. This creates a dependency preventing MT from being resumed, namely the memory handles of imported allocations can’t be recreated.

When both processes are suspended and resumed at the same time, there is a coordination barrier which ensures that any shared memory allocations are restored by the owning process and then re-exported/re-imported.

Thus the current work requires SGL to be awake or woken up at the same time, in order for it to map the data into device memory, so that MT can get past the barrier. Without this we need some other mechanism to ensure the data is restored into device memory so it can be accessed. Note that it is important to restore the allocation into the same device that it was offloaded from, because the C/R application is supposed to remain policy-free.

## Designs

Given an allocation A with original owner process P_O and any number of importer processes.

### Importers create local copy until exporter wakes up

When P_O gets suspended it publishes a manifest describing where an importer can find the shared memory.

When any importer wakes up it creates a local GPU handle to avoid being blocked by P_O not being active.

When any importer that has awoken goes back to sleep the changes are written back to the original position in SGL’s offload store.

When SGL wakes it can take over ownership by creating an allocation, syncing the changes and then coordinating a re-export procedure amongst any process still alive.

#### Cons:

Whilst P_O is suspended we lose the shared-memory property, which means that it is not compatible when there is more than one importer who need to see each others writes

This could be avoided by recreating the import relationship by letting one of the importers act as the temporary exporter.

### Long-lived shared-memory daemon

Transfer ownership of allocations for shareable memory into a long-lived "master daemon" to decouple their lifecycle from any single process. Any process can suspend/resume independently.

The question becomes at what point to give up ownership. It could be anytime between the memory first being allocated up to the original allocator being suspended. Another reasonable point is when the cuda export API call is intercepted.

On first glance it seems like giving up ownership on allocation results in every allocation being made into the memory daemon resulting in severe overhead due to increased IPC requirements, contention..

Whereas the problem with giving up ownership during export/suspend means that the memory needs to be moved out of the context of the original process. This results in additional copies?

However, since cuda memory allocations are tagged with a handle_type that demonstrates whether they are intended to be used for sharing we can figure out the precise memory that will be used for SM and move this into a master daemon.