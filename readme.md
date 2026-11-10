# Static vs. Dynamic Dataflow for Sparse LLM Workloads

CMSC 818J research project (UMD, Fall 2026).

## Question

When is a dynamic SpGEMM dataflow (SegFold, ISCA 2026) actually better than static dataflows (inner product, outer product, Gustavson), and can a cheap learned predictor choose between them?

## Motivation

- SegFold reports a 1.95× geomean speedup over Spada and 5.3× over the best static Flexagon dataflow, but it was evaluated only on SuiteSparse and synthetic matrices.
- It has known weak spots: a few very dense rows (it loses to Spada on ca-GrQc), wide matrices (large K relative to M), and large density gaps between A and B (operand order matters).
- Pruned LLM and DNN layers may hit these weak spots. This has not been tested.

## Systems Compared

| System | Type | Simulator |
|---|---|---|
| SegFold | Dynamic (sub-tile) | `csegfold` (C++) from the [SegFold artifact](https://github.com/PolyArch/SegFold-AE) |
| Flexagon (IP / OP / Gustavson) | Static (selectable) | STONNE |
| Spada | Adaptive static (window-based) | Spada open-source simulator |

All three use the same Ramulator2 HBM2 memory backend, following the SegFold artifact's configs.

## Workloads

- **Validation:** 3–4 SuiteSparse matrices from SegFold's Figure 8, including ca-GrQc.
- **Synthetic sweep:** vary density, row-length skew (injected dense rows), M:K aspect ratio, and A/B density ratio.
- **Real layers (~30–50 matrices, about 1K–4K per side):**
  - Small LLM weights pruned with Wanda (50–90% unstructured) and 2:4 structured
  - Sparse activations from ReLU-based layers (e.g., OPT) or sparse attention masks
  - Pruned DNN layers from the Flexagon evaluation
  - One representative layer per type (QKV, FFN up, FFN down), cropped or tiled

## Metrics

- Cycles and cycles per MAC
- SegFold speedup over the best static dataflow per matrix (oracle static)
- Effect of operand order (A×B vs. Bᵀ×Aᵀ)

## Features Logged per Matrix

Density of A and B, row-length mean/variance/max, M:K ratio, A/B density ratio, structured vs. unstructured sparsity.

## Learned Predictor

A small decision tree that chooses between dynamic (SegFold), the best static dataflow, and operand order.

- **Metric:** regret, i.e. latency lost compared with the oracle choice
- **Split:** leave-one-model-out

## Plan

1. **Setup:** build the SegFold artifact (Docker available) on a machine with at least 64 GB of RAM; reproduce Figure 8 on a few matrices and check the CSVs against the expected results.
2. **Baselines:** set up STONNE (Flexagon) and Spada with matching configs.
3. **Synthetic sweep:** map where dynamic wins or loses.
4. **Real layers:** run pruned LLM and DNN layers and compare against the synthetic map.
5. **Predictor:** train the decision tree and evaluate it by regret.
6. **Write-up:** report the results.

## Expected Result

"Dynamic dataflow wins on X but loses on Y in real sparse LLM layers; a cheap predictor recovers Z% of the gap."

## Risks

- Simulator memory and runtime on large layers. Keep matrices small and use `--jobs` and `--timeout`.
- Mapping real activation sparsity onto SpGEMM inputs. Choose the activation source early.

## Logging In to the Server

The `nexus` entry in `~/.ssh/config` points to `cdollo@nexusclass.umiacs.umd.edu` and keeps one connection open for 8 hours.

```bash
ssh -fN nexus          # log in once in the background (password + Duo push)
ssh nexus              # open a shell; no prompt while that connection is open
ssh -O check nexus     # check whether the connection is still open
ssh -O exit nexus      # close the connection
```

After a reboot or 8 hours idle, run `ssh -fN nexus` again.

## References

- Wu et al., *SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow*, ISCA 2026. [arXiv:2606.26701](https://arxiv.org/abs/2606.26701)
- Yadav, Namjoo, Asgari, *Misam: Machine Learning Assisted Dataflow Selection in Accelerators for Sparse Matrix Multiplication*, MICRO 2025.
- Yadav, Asgari, *DynaFlow: An ML Framework for Dynamic Dataflow Selection in SpGEMM Accelerators*, IEEE CAL 2025.
- Muñoz-Martínez et al., *Flexagon: A Multi-Dataflow Sparse-Sparse Matrix Multiplication Accelerator*, ASPLOS 2023.
- Li et al., *Spada: Accelerating Sparse Matrix Multiplication with Adaptive Dataflow*, ASPLOS 2023.