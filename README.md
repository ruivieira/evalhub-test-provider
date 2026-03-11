# evalhub-test-provider

Test EvalHub provider used to validate the adapter SDK and cluster deployment.

## Behavior

- **Data**: Reads a JSON datafile from `/data` (e.g. `datafile.json` or first `.json`/`.jsonl` in `/data`).
- **Processing**: Converts the JSON to a pandas DataFrame and logs its shape and head.
- **Metrics**: Computes simple metrics (row/column count, mean of numeric columns) and reports them to EvalHub.
- **MLflow**: Sends metrics to MLflow when `experiment_name` is set in the job spec (via `DefaultCallbacks.report_metrics_to_mlflow`).

## Dependencies

- **eval-hub-sdk**: Installed from upstream `dev` branch (`eval-hub-sdk[adapter] @ git+https://github.com/opendatahub-io/eval-hub-sdk.git@dev`). To use a local SDK (e.g. `../eval-hub-sdk`), replace the dependency in `pyproject.toml` with `eval-hub-sdk[adapter] @ file:///${PROJECT_ROOT}/../eval-hub-sdk`.
- **pandas**: For DataFrame handling.

## Build on cluster

From the `poc` directory:

```bash
just ocp-build-test-provider
```

## Deploy provider ConfigMap

```bash
just deploy-test-provider
```

The full dev stack (including this provider) is deployed with:

```bash
just deploy-evalhub-dev
```
