"""Test EvalHub adapter: read JSON from /data, convert to pandas, report metrics to MLflow."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from evalhub.adapter import (
    FrameworkAdapter,
    JobCallbacks,
    JobPhase,
    JobResults,
    JobSpec,
    JobStatus,
    JobStatusUpdate,
)
from evalhub.adapter.models.job import ErrorInfo, MessageInfo
from evalhub.models.api import EvaluationResult

logger = logging.getLogger(__name__)

DATA_DIR = Path("/data")

# Static dummy metrics returned on every run (for testing/demo without real data).
STATIC_DUMMY_METRICS: list[EvaluationResult] = [
    EvaluationResult(
        metric_name="accuracy",
        metric_value=0.85,
        metric_type="float",
        num_samples=100,
        metadata={"description": "dummy static metric"},
    ),
    EvaluationResult(
        metric_name="f1_score",
        metric_value=0.82,
        metric_type="float",
        num_samples=100,
    ),
    EvaluationResult(
        metric_name="latency_p99_ms",
        metric_value=150,
        metric_type="int",
        num_samples=100,
    ),
    EvaluationResult(
        metric_name="throughput",
        metric_value=42.5,
        metric_type="float",
        num_samples=1,
    ),
]
TEST_DATA_DIR = Path("/test_data")
_DATA_SUFFIXES = (".json", ".jsonl")


def _first_datafile_in_dir(path: Path) -> Path | None:
    """Return the first data file (.json or .jsonl) in path, or None."""
    if not path.exists() or not path.is_dir():
        return None
    for f in sorted(path.iterdir()):
        if f.suffix.lower() in _DATA_SUFFIXES and f.is_file():
            return f
    return None


def _log_dir_contents(path: Path) -> None:
    """Log what is actually present in a directory to aid debugging."""
    if not path.exists():
        logger.warning("Directory %s does not exist", path)
        return
    all_files = sorted(path.iterdir()) if path.is_dir() else []
    if not all_files:
        logger.warning("Directory %s exists but is empty", path)
    else:
        logger.info("Directory %s contains %d item(s): %s", path, len(all_files), [f.name for f in all_files])
        data_files = [f for f in all_files if f.suffix.lower() in _DATA_SUFFIXES and f.is_file()]
        if data_files:
            logger.info("Recognised data files: %s", [f.name for f in data_files])
        else:
            logger.warning("No recognised data files (%s) found in %s", _DATA_SUFFIXES, path)


def _resolve_data_path(config: JobSpec) -> Path | None:
    """Resolve the data file path, returning None if no file can be found.

    Search order:
      1. Explicit path from parameters (data_path or datafile key)
      2. /test_data  — populated by the eval-hub-init S3 download sidecar
      3. /data       — default data volume (local / manually mounted)
    """
    bc = config.parameters or {}
    explicit = bc.get("data_path") or bc.get("datafile")
    if explicit:
        p = Path(explicit) if Path(explicit).is_absolute() else DATA_DIR / explicit
        logger.info("Explicit data path configured: %s", p)
        if p.exists():
            logger.info("Found configured data file: %s", p)
            return p
        logger.warning("Configured data path %s does not exist", p)
        _log_dir_contents(DATA_DIR)
        return None

    # Check /test_data first — this is where eval-hub-init places S3 downloads
    logger.info("No explicit data path configured, checking %s (S3 downloads)", TEST_DATA_DIR)
    first_test = _first_datafile_in_dir(TEST_DATA_DIR)
    if first_test is not None:
        logger.info("Found data file in %s: %s", TEST_DATA_DIR, first_test)
        return first_test
    if TEST_DATA_DIR.exists():
        _log_dir_contents(TEST_DATA_DIR)

    # Fall back to /data
    logger.info("Nothing usable in %s, searching %s", TEST_DATA_DIR, DATA_DIR)
    default_json = DATA_DIR / "datafile.json"
    if default_json.exists():
        logger.info("Found default data file: %s", default_json)
        return default_json

    first = _first_datafile_in_dir(DATA_DIR)
    if first is not None:
        logger.info("Using first data file discovered in %s: %s", DATA_DIR, first)
        return first

    logger.warning("No data file found in %s or %s", TEST_DATA_DIR, DATA_DIR)
    _log_dir_contents(DATA_DIR)
    return None


def _load_json_to_dataframe(path: Path) -> pd.DataFrame:
    """Load JSON or JSONL from path into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        return pd.DataFrame(records)
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict) and "data" in data:
        return pd.DataFrame(data["data"])
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return pd.DataFrame(v)
        return pd.DataFrame([data])
    raise ValueError(f"Unsupported JSON structure: root type {type(data)}")


def _build_results_json(results: JobResults, config: JobSpec) -> bytes:
    """Serialise job results to JSON bytes for MLflow artifact upload."""
    payload = {
        "job_id": results.id,
        "benchmark_id": results.benchmark_id,
        "model_name": results.model_name,
        "overall_score": results.overall_score,
        "num_examples_evaluated": results.num_examples_evaluated,
        "duration_seconds": results.duration_seconds,
        "completed_at": results.completed_at.isoformat(),
        "experiment_name": config.experiment_name,
        "metrics": [
            {
                "name": r.metric_name,
                "value": r.metric_value,
                "type": r.metric_type,
                "num_samples": r.num_samples,
                **({"metadata": r.metadata} if r.metadata else {}),
            }
            for r in results.results
        ],
        "evaluation_metadata": results.evaluation_metadata,
    }
    return json.dumps(payload, indent=2).encode()


def _build_report_html(results: JobResults, config: JobSpec) -> bytes:
    """Build a simple HTML evaluation report for MLflow artifact upload."""
    rows = "\n".join(
        f"<tr><td>{r.metric_name}</td><td>{r.metric_value}</td>"
        f"<td>{r.metric_type}</td><td>{r.num_samples}</td></tr>"
        for r in results.results
    )
    score_str = f"{results.overall_score:.6f}" if results.overall_score is not None else "N/A"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>EvalHub Evaluation Report — {results.id}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2em; }}
    h1 {{ font-size: 1.4em; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
    th {{ background: #f0f0f0; }}
    .meta {{ margin-bottom: 1.5em; }}
    .meta dt {{ font-weight: bold; display: inline; }}
    .meta dd {{ display: inline; margin: 0 1em 0 0.5em; }}
  </style>
</head>
<body>
  <h1>Evaluation Report</h1>
  <dl class="meta">
    <dt>Job ID</dt><dd>{results.id}</dd>
    <dt>Benchmark</dt><dd>{results.benchmark_id}</dd>
    <dt>Model</dt><dd>{results.model_name}</dd>
    <dt>Experiment</dt><dd>{config.experiment_name or "—"}</dd>
    <dt>Overall score</dt><dd>{score_str}</dd>
    <dt>Examples evaluated</dt><dd>{results.num_examples_evaluated}</dd>
    <dt>Duration</dt><dd>{results.duration_seconds:.2f}s</dd>
    <dt>Completed at</dt><dd>{results.completed_at.isoformat()}</dd>
  </dl>
  <h2>Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Value</th><th>Type</th><th>Samples</th></tr></thead>
    <tbody>
{rows}
    </tbody>
  </table>
</body>
</html>
"""
    return html.encode()


class TestEvalHubAdapter(FrameworkAdapter):
    """EvalHub adapter that reads a datafile from /data, shows it as a dataframe, and sends metrics to MLflow."""

    def run_benchmark_job(self, config: JobSpec, callbacks: JobCallbacks) -> JobResults:
        start_time = time.time()
        logger.info("Starting test EvalHub job %s for benchmark %s", config.id, config.benchmark_id)

        try:
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.INITIALIZING,
                    progress=0.0,
                    message=MessageInfo(
                        message="Resolving data path",
                        message_code="initializing",
                    ),
                )
            )
            data_path = _resolve_data_path(config)
            if data_path is None:
                logger.warning(
                    "No data file found — completing job with zero metrics. "
                    "Mount a .json or .jsonl file at %s or set data_path in parameters.",
                    DATA_DIR,
                )
                n_rows, n_cols = 0, 0
                df = pd.DataFrame()
            else:
                callbacks.report_status(
                    JobStatusUpdate(
                        status=JobStatus.RUNNING,
                        phase=JobPhase.LOADING_DATA,
                        progress=0.2,
                        message=MessageInfo(
                            message=f"Loading {data_path} and converting to dataframe",
                            message_code="loading_data",
                        ),
                    )
                )
                df = _load_json_to_dataframe(data_path)
                n_rows, n_cols = df.shape
                logger.info("DataFrame from %s: shape=%s columns=%s", data_path, df.shape, list(df.columns))
                logger.info("DataFrame head:\n%s", df.head().to_string())
                if n_rows == 0:
                    logger.warning("DataFrame loaded from %s is empty", data_path)

            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.RUNNING,
                    phase=JobPhase.RUNNING_EVALUATION,
                    progress=0.5,
                    message=MessageInfo(
                        message="Computing metrics",
                        message_code="running_evaluation",
                    ),
                )
            )
            # Start with static dummy metrics (always returned)
            results_list: list[EvaluationResult] = list(STATIC_DUMMY_METRICS)
            # Add metrics derived from the dataframe (numeric columns only)
            numeric = df.select_dtypes(include=["number"])
            if not numeric.empty:
                for col in numeric.columns:
                    series = numeric[col].dropna()
                    if len(series) > 0:
                        mean_val = float(series.mean())
                        results_list.append(
                            EvaluationResult(
                                metric_name=f"{col}_mean",
                                metric_value=round(mean_val, 6),
                                metric_type="float",
                                num_samples=len(series),
                                metadata={"min": float(series.min()), "max": float(series.max())},
                            )
                        )
            # Always add row_count and column_count
            results_list.append(
                EvaluationResult(
                    metric_name="row_count",
                    metric_value=n_rows,
                    metric_type="int",
                    num_samples=n_rows,
                )
            )
            results_list.append(
                EvaluationResult(
                    metric_name="column_count",
                    metric_value=n_cols,
                    metric_type="int",
                    num_samples=1,
                )
            )
            overall_score = (
                sum(r.metric_value for r in results_list if isinstance(r.metric_value, (int, float)))
                / len(results_list)
                if results_list
                else None
            )

            duration = time.time() - start_time
            job_results = JobResults(
                id=config.id,
                benchmark_id=config.benchmark_id,
                benchmark_index=config.benchmark_index,
                model_name=config.model.name,
                results=results_list,
                overall_score=overall_score,
                num_examples_evaluated=n_rows,
                duration_seconds=round(duration, 2),
                completed_at=datetime.now(UTC),
                evaluation_metadata={
                    "data_path": str(data_path) if data_path else "none",
                    "columns": list(df.columns),
                },
                oci_artifact=None,
            )
            return job_results

        except Exception as e:
            logger.exception("Test EvalHub job %s failed", config.id)
            callbacks.report_status(
                JobStatusUpdate(
                    status=JobStatus.FAILED,
                    error=ErrorInfo(message=str(e), message_code="job_failed"),
                    error_details={"exception_type": type(e).__name__},
                )
            )
            raise


def main() -> None:
    """Entry point for running the adapter as an EvalHub K8s Job."""
    import os
    import sys

    from evalhub.adapter import DefaultCallbacks

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    job_spec_path = os.environ.get("EVALHUB_JOB_SPEC_PATH", "/meta/job.json")
    try:
        adapter = TestEvalHubAdapter(job_spec_path=job_spec_path)
        logger.info("Loaded job %s", adapter.job_spec.id)
        logger.info("Benchmark: %s", adapter.job_spec.benchmark_id)

        oci_auth = os.environ.get("OCI_AUTH_CONFIG_PATH")
        callbacks = DefaultCallbacks(
            job_id=adapter.job_spec.id,
            benchmark_id=adapter.job_spec.benchmark_id,
            benchmark_index=adapter.job_spec.benchmark_index,
            provider_id=adapter.job_spec.provider_id,
            sidecar_url=adapter.job_spec.callback_url,
            oci_auth_config_path=Path(oci_auth) if oci_auth else None,
            oci_insecure=os.environ.get("OCI_REGISTRY_INSECURE", "false").lower() == "true",
        )

        results = adapter.run_benchmark_job(adapter.job_spec, callbacks)
        logger.info("Job completed: %s", results.id)
        callbacks.report_results(results)

        from evalhub.adapter.mlflow import MlflowArtifact

        callbacks.mlflow.save(
            results,
            adapter.job_spec,
            artifacts=[
                MlflowArtifact("results.json", _build_results_json(results, adapter.job_spec), "application/json"),
                MlflowArtifact("report.html", _build_report_html(results, adapter.job_spec), "text/html"),
            ],
        )
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error("Job spec or data not found: %s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception:
        logger.exception("Job failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
