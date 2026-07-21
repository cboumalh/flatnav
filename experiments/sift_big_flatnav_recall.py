import argparse
import gc
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import flatnav
import hnswlib
import numpy as np
from flatnav.data_type import DataType

from data_loader import get_data_loader


FLATNAV_DATA_TYPES = {
    "float32": DataType.float32,
    "uint8": DataType.uint8,
    "int8": DataType.int8,
}


def compute_recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    found_set = set(found[:k].tolist())
    truth_set = set(truth[:k].tolist())
    if not truth_set:
        return 0.0
    return len(found_set.intersection(truth_set)) / float(k)


def _to_float32_contiguous(batch: np.ndarray) -> np.ndarray:
    if batch.dtype == np.float32 and batch.flags.c_contiguous:
        return batch
    return np.ascontiguousarray(batch, dtype=np.float32)


def run_faiss_flatl2_validation(
    train_data: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    max_queries: int,
    num_threads: int,
    progress_interval: int = 1000,
) -> Dict[str, object]:
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for --validate-faiss-flatl2. Install with `pip install faiss-cpu`."
        ) from exc

    faiss_threads = max(1, num_threads)
    try:
        faiss.omp_set_num_threads(faiss_threads)
    except AttributeError:
        logging.warning("FAISS does not expose omp_set_num_threads; using library default thread count.")

    eval_queries = len(queries)
    if max_queries > 0:
        eval_queries = min(eval_queries, max_queries)

    if eval_queries <= 0:
        return {
            "enabled": True,
            "executed": False,
            "reason": "No queries available for FAISS validation.",
        }

    faiss_index = faiss.IndexFlatL2(train_data.shape[1])
    faiss_index.add(_to_float32_contiguous(train_data))
    recalls: List[float] = []
    batch_size = max(1, progress_interval)

    for start in range(0, eval_queries, batch_size):
        end = min(start + batch_size, eval_queries)
        _, faiss_neighbors = faiss_index.search(
            _to_float32_contiguous(queries[start:end]),
            k,
        )

        recalls.extend(
            compute_recall_at_k(faiss_neighbors[i], ground_truth[start + i], k)
            for i in range(end - start)
        )

        if (end % progress_interval == 0) or (end == eval_queries):
            logging.info("FAISS validation processed %d/%d queries", end, eval_queries)

    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    min_recall = float(np.min(recalls)) if recalls else 0.0
    max_recall = float(np.max(recalls)) if recalls else 0.0

    if avg_recall < 0.99:
        verdict = (
            "Provided ground truth is inconsistent with FAISS exact L2 neighbors. "
            "This indicates a data/ground-truth mismatch rather than FlatNav graph quality."
        )
    else:
        verdict = (
            "Provided ground truth matches FAISS exact L2 neighbors. "
            "Recall calculation logic is likely correct."
        )

    return {
        "enabled": True,
        "executed": True,
        "num_threads": int(faiss_threads),
        "num_queries_evaluated": int(eval_queries),
        "k": int(k),
        "avg_recall_vs_provided_ground_truth": avg_recall,
        "min_recall_vs_provided_ground_truth": min_recall,
        "max_recall_vs_provided_ground_truth": max_recall,
        "verdict": verdict,
    }


def build_flatnav_index_from_graph_file(
    train_data: np.ndarray,
    metric: str,
    num_node_links: int,
    mtx_filename: str,
) -> Tuple[Any, float]:
    dataset_size, dim = train_data.shape

    logging.info("Creating FlatNav index from graph file %s", mtx_filename)
    build_start = time.time()
    index = flatnav.index.create(
        distance_type=metric,
        index_data_type=FLATNAV_DATA_TYPES["float32"],
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=num_node_links,
        verbose=False,
        collect_stats=False,
    )

    graph_load_start = time.time()
    index.allocate_nodes(data=train_data).build_graph_links(mtx_filename)
    graph_load_sec = time.time() - graph_load_start
    logging.info("Loaded graph links from %s in %.2f sec", mtx_filename, graph_load_sec)

    save_dir = Path("/data/index")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "flatnav.index"
    index.save(str(save_path))
    logging.info("Saved FlatNav index to %s", save_path)

    return index, time.time() - build_start


def build_flatnav_index_from_hnsw_graph(
    train_data: np.ndarray,
    metric: str,
    num_node_links: int,
    ef_construction: int,
    num_build_threads: int,
    build_batch_size: int,
    graph_tmp_dir: str,
    save_mtx: str,
) -> Tuple[Any, float]:
    dataset_size, dim = train_data.shape
    hnsw_space = metric if metric == "l2" else "ip"

    hnsw_index = hnswlib.Index(space=hnsw_space, dim=dim)
    hnsw_index.init_index(
        max_elements=dataset_size,
        ef_construction=ef_construction,
        M=max(2, num_node_links // 2),
    )
    hnsw_index.set_num_threads(max(1, num_build_threads))

    logging.info("Building HNSW base layer graph in batches")
    build_start = time.time()
    for start in range(0, dataset_size, build_batch_size):
        end = min(start + build_batch_size, dataset_size)
        batch = _to_float32_contiguous(train_data[start:end])
        labels = np.arange(start, end, dtype=np.int32)
        hnsw_index.add_items(batch, labels)
        if (end == dataset_size) or ((end // build_batch_size) % 5 == 0):
            logging.info("HNSW added %d/%d vectors", end, dataset_size)

    if save_mtx:
        save_mtx_path = Path(save_mtx).expanduser().resolve()
        save_mtx_path.parent.mkdir(parents=True, exist_ok=True)
        mtx_filename = str(save_mtx_path)
        remove_mtx_after_load = False
    else:
        graph_tmp_dir_path = Path(graph_tmp_dir).expanduser().resolve()
        graph_tmp_dir_path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".mtx", delete=False, dir=graph_tmp_dir_path
        ) as tmp:
            mtx_filename = tmp.name
        remove_mtx_after_load = True

    save_start = time.time()
    hnsw_index.save_base_layer_graph(filename=mtx_filename)
    save_sec = time.time() - save_start
    mtx_size_bytes = os.path.getsize(mtx_filename)
    mtx_size_gib = mtx_size_bytes / float(1024**3)
    save_mib_per_sec = (mtx_size_bytes / float(1024**2)) / max(save_sec, 1e-9)
    logging.info(
        "Saved base-layer graph to %s (%.2f GiB) in %.2f sec (%.2f MiB/s)",
        mtx_filename,
        mtx_size_gib,
        save_sec,
        save_mib_per_sec,
    )

    # Release HNSW memory before building FlatNav to reduce memory pressure.
    del hnsw_index
    gc.collect()

    try:
        index, _ = build_flatnav_index_from_graph_file(
            train_data=train_data,
            metric=metric,
            num_node_links=num_node_links,
            mtx_filename=mtx_filename,
        )
    finally:
        if remove_mtx_after_load:
            try:
                os.remove(mtx_filename)
            except OSError:
                pass

    return index, time.time() - build_start


def run_recall_only(
    dataset_path: str,
    queries_path: str,
    gtruth_path: str,
    metric: str,
    num_node_links: int,
    ef_construction: int,
    ef_search_values: List[int],
    num_build_threads: int,
    num_search_threads: int,
    build_batch_size: int,
    graph_tmp_dir: str,
    save_mtx: str,
    existing_mtx: str,
    num_queries: int,
    k: int,
    validate_faiss_flatl2: bool,
    faiss_validate_queries: int,
    faiss_num_threads: int,
    search_only: bool = False,
    exp_id: str = "",
) -> Dict[str, object]:
    loader = get_data_loader(
        train_dataset_path=dataset_path,
        queries_path=queries_path,
        ground_truth_path=gtruth_path,
    )
    train_data, queries, ground_truth = loader.load_data()

    if num_queries > 0:
        limit = min(num_queries, queries.shape[0], ground_truth.shape[0])
        queries = queries[:limit]
        ground_truth = ground_truth[:limit]

    train_data = train_data.astype(np.float32, copy=False)
    queries = _to_float32_contiguous(queries)
    ground_truth = np.ascontiguousarray(ground_truth, dtype=np.int32)

    dataset_size = train_data.shape[0]
    dim = train_data.shape[1]
    effective_k = min(k, ground_truth.shape[1])
    if effective_k <= 0:
        raise ValueError("Ground truth has no neighbors to evaluate recall.")

    faiss_validation: Dict[str, object] = {
        "enabled": False,
        "executed": False,
        "reason": "Validation not requested.",
    }

    if validate_faiss_flatl2:
        if metric != "l2":
            faiss_validation = {
                "enabled": True,
                "executed": False,
                "reason": "--validate-faiss-flatl2 only supports metric=l2.",
            }
            logging.warning(faiss_validation["reason"])
        else:
            effective_faiss_threads = (
                num_search_threads if faiss_num_threads <= 0 else faiss_num_threads
            )
            logging.info(
                "Running FAISS FlatL2 validation on up to %d queries with %d thread(s)",
                faiss_validate_queries if faiss_validate_queries > 0 else len(queries),
                effective_faiss_threads,
            )
            faiss_validation = run_faiss_flatl2_validation(
                train_data=train_data,
                queries=queries,
                ground_truth=ground_truth,
                k=effective_k,
                max_queries=faiss_validate_queries,
                num_threads=effective_faiss_threads,
            )
            logging.info(
                "FAISS validation avg recall@%d vs provided ground truth: %.6f",
                effective_k,
                faiss_validation.get("avg_recall_vs_provided_ground_truth", 0.0),
            )

    if existing_mtx:
        mtx_path = Path(existing_mtx).expanduser().resolve()
        if not mtx_path.is_file():
            raise FileNotFoundError(f"Existing MTX file not found: {mtx_path}")
        index, build_time_sec = build_flatnav_index_from_graph_file(
            train_data=train_data,
            metric=metric,
            num_node_links=num_node_links,
            mtx_filename=str(mtx_path),
        )
    else:
        if num_build_threads != 1:
            logging.warning(
                "num_build_threads=%d requested. If you still hit native crashes, retry with --num-build-threads 1.",
                num_build_threads,
            )

        index, build_time_sec = build_flatnav_index_from_hnsw_graph(
            train_data=train_data,
            metric=metric,
            num_node_links=num_node_links,
            ef_construction=ef_construction,
            num_build_threads=num_build_threads,
            build_batch_size=build_batch_size,
            graph_tmp_dir=graph_tmp_dir,
            save_mtx=save_mtx,
        )
    logging.info("Build complete in %.2f sec", build_time_sec)

    # Memory optimization: if existing_mtx is used, the base nodes might be memory mapped or we can at least invoke gc
    gc.collect()

    index.set_num_threads(num_search_threads)
    logging.info("Set FlatNav search threads to %d", num_search_threads)
    
    if search_only and exp_id:
        os.makedirs("/tmp/measurement", exist_ok=True)
        ready_file = f"/tmp/measurement/{exp_id}.ready"
        logging.info("Signaling ready for measurement at %s", ready_file)
        Path(ready_file).touch()
        # Sleep slightly to let the polling script detect and attach
        time.sleep(1)
    else:
        logging.info("Running sequential queries without signaling for search-only benchmarking.")

    results: Dict[str, Dict[str, float]] = {}

    for ef_search in ef_search_values:
        logging.info("Running sequential queries with ef_search=%d", ef_search)
        start = time.time()
        recalls = []
        failed_queries = 0
        num_queries_total = len(queries)
        progress_checkpoints = {
            max(1, int(round((num_queries_total * step) / 10.0)))
            for step in range(1, 11)
        }

        for i, query in enumerate(queries):
            try:
                _, neighbors = index.search_single(
                    query=query,
                    ef_search=ef_search,
                    K=effective_k,
                    num_initializations=100,
                )
            except RuntimeError:
                failed_queries += 1
                continue
            recalls.append(compute_recall_at_k(neighbors, ground_truth[i], effective_k))

            processed = i + 1
            if processed in progress_checkpoints:
                running_avg_recall = float(np.mean(recalls)) if recalls else 0.0
                logging.info(
                    "Processed %d/%d queries, running_recall@%d=%.6f, failed=%d",
                    processed,
                    num_queries_total,
                    effective_k,
                    running_avg_recall,
                    failed_queries,
                )

        total_sec = time.time() - start
        avg_recall = float(np.mean(recalls)) if recalls else 0.0
        results[str(ef_search)] = {
            "recall": avg_recall,
            "total_time_sec": total_sec,
            "avg_time_ms_per_query": (total_sec * 1000.0) / max(1, len(queries)),
            "failed_queries": failed_queries,
        }
        logging.info(
            "ef_search=%d recall@%d=%.6f total_time=%.2fs failed=%d",
            ef_search,
            effective_k,
            avg_recall,
            total_sec,
            failed_queries,
        )

    # Explicitly release native resources to reduce teardown-related native errors.
    del index
    gc.collect()

    return {
        "dataset": dataset_path,
        "queries": queries_path,
        "ground_truth": gtruth_path,
        "metric": metric,
        "dataset_size": int(dataset_size),
        "dim": int(dim),
        "num_queries": int(len(queries)),
        "num_node_links": num_node_links,
        "ef_construction": ef_construction,
        "num_build_threads": num_build_threads,
        "num_search_threads": num_search_threads,
        "k": effective_k,
        "build_time_sec": build_time_sec,
        "faiss_flatl2_validation": faiss_validation,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FlatNav and run sequential query recall only."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to training dataset (.fvecs/.npy/.bin supported by data_loader).",
    )
    parser.add_argument("--queries", required=True, help="Path to queries file.")
    parser.add_argument("--gtruth", required=True, help="Path to ground-truth file.")
    parser.add_argument("--metric", default="l2", choices=["l2", "angular"])
    parser.add_argument("--num-node-links", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--num-build-threads", type=int, default=1)
    parser.add_argument("--num-search-threads", type=int, default=1)
    parser.add_argument(
        "--build-batch-size",
        type=int,
        default=250000,
        help="Number of vectors per HNSW add batch.",
    )
    parser.add_argument(
        "--graph-tmp-dir",
        default=str(Path(tempfile.gettempdir()).resolve()),
        help="Directory where temporary .mtx graph is written. Use a fast local SSD path.",
    )
    parser.add_argument(
        "--existing-mtx",
        default="",
        help="Optional path to a prebuilt HNSW base-layer .mtx graph. If set, HNSW build/dump is skipped.",
    )
    parser.add_argument(
        "--save-mtx",
        default="",
        help="Optional path to persist generated HNSW base-layer .mtx for reuse in future runs.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=0,
        help="Optional query limit for faster runs (0 means all).",
    )
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument(
        "--validate-faiss-flatl2",
        action="store_true",
        help="Validate provided ground truth using FAISS IndexFlatL2.",
    )
    parser.add_argument(
        "--faiss-validate-queries",
        type=int,
        default=1000,
        help="Maximum number of queries used for FAISS validation (0 means all).",
    )
    parser.add_argument(
        "--faiss-num-threads",
        type=int,
        default=0,
        help="Thread count for FAISS validation (0 means use --num-search-threads).",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path to save results as JSON.",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Write to /tmp/measurement/ to signal ready for search benchmarking.",
    )
    parser.add_argument(
        "--exp-id",
        default="",
        help="Experiment ID for the /tmp/measurement/ file if --search-only is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    summary = run_recall_only(
        dataset_path=args.dataset,
        queries_path=args.queries,
        gtruth_path=args.gtruth,
        metric=args.metric,
        num_node_links=args.num_node_links,
        ef_construction=args.ef_construction,
        ef_search_values=args.ef_search,
        num_build_threads=args.num_build_threads,
        num_search_threads=args.num_search_threads,
        build_batch_size=args.build_batch_size,
        graph_tmp_dir=args.graph_tmp_dir,
        save_mtx=args.save_mtx,
        existing_mtx=args.existing_mtx,
        num_queries=args.num_queries,
        k=args.k,
        validate_faiss_flatl2=args.validate_faiss_flatl2,
        faiss_validate_queries=args.faiss_validate_queries,
        faiss_num_threads=args.faiss_num_threads,
        search_only=args.search_only,
        exp_id=args.exp_id,
    )

    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logging.info("Saved results to %s", output_path)


if __name__ == "__main__":
    main()