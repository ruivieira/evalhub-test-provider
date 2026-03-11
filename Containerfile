# Test EvalHub provider: reads JSON from /data, shows dataframe, sends metrics to MLflow.
# Uses eval-hub-sdk from upstream dev branch.
FROM registry.access.redhat.com/ubi9/python-312:latest

USER 0

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e .

USER 1001

ENTRYPOINT ["python", "-m", "evalhub_test_provider.adapter"]
