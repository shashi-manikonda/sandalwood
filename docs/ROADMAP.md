# Technical Roadmap: High-Performance Taylor Function Library

This document outlines the technical trajectory for the `sandalwood` project. These tasks represent the next tier of performance and scalability beyond the current Numba-accelerated CPU implementation.

---

## 1. Full Composition Kernel (High Priority)

**Current State:**
The `TaylorMap.compose` function currently uses a hybrid approach:
1. Python-level loop over the components of the outer map.
2. Numba-accelerated inner kernels for power accumulation and arithmetic.

**Goal:**
Fuse the entire composition algorithm into a single, unified C++/Numba kernel.
* **Objective:** Eliminate the Python interpreter overhead for the outer component loop.
* **Technique:** Implement a recursive or iterative power-propagation kernel that calculates $F(G(x))$ in a single pass over the cache-resident dense buffers.

---

## 2. GPU Offloading (Scalability)

**Current State:**
The library uses `numba.prange` for CPU-based multi-threading. While efficient for moderate scales, it is limited by CPU memory bandwidth and core counts.

**Goal:**
Port the "hot" evaluation and multiplication kernels to the GPU.
* **Objective:** Achieve massively parallel polynomial evaluation ($10^7+$ points).
* **Technique:** 
    * Port `evaluate_dense_kernel` to `numba.cuda` or `cupy.RawKernel`.
    * Implement **shared memory tiling** for the power-cache to maximize throughput on NVIDIA hardware.
    * Support unified memory to minimize host-to-device transfer overhead.

---

## 3. Algorithmic Upgrade: Fast Multipole Method (FMM)

**Current State:**
The Biot-Savart solver operates at $O(N_{sources} \cdot N_{points})$ complexity.

**Goal:**
Implement a **Differential Algebra Fast Multipole Method (DA-FMM)**.
* **Objective:** Reduce the computational complexity to $O(N_{sources} + N_{points})$.
* **Technique:** 
    * Use DA Taylor expansions to represent field "multipoles" of distant source clusters.
    * Integrate with the COSY backend for high-order local expansions.
    * This is critical for scaling magnetization simulations to millions of discrete source elements.

---

## 4. Distributed Multi-Node Execution

**Goal:**
Enable large-scale parameter sweeps across high-performance clusters.
* **Objective:** Distribute `CosyScope` execution across multiple nodes.
* **Technique:** Integrate with `Dask` or `Ray` to distribute batch evaluations of Taylor maps.
