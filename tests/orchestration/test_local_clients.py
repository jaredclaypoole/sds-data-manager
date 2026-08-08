"""Tests for local replacements of AWS clients."""

from sds_data_manager.lambda_code.SDSCode.local_clients import LocalBatchClient


def _submit(client, name="test-job"):
    return client.submit_job(
        jobName=name,
        jobQueue="local-queue",
        jobDefinition="local-definition:1",
        containerOverrides={"command": ["instrument", "level"]},
    )


def test_describe_jobs_returns_submitted_job():
    client = LocalBatchClient()
    submitted = _submit(client)

    response = client.describe_jobs(jobs=[submitted["jobId"]])

    assert response == {
        "jobs": [
            {
                "jobId": "local-1",
                "jobName": "test-job",
                "jobQueue": "local-queue",
                "jobDefinition": "local-definition:1",
                "status": "SUBMITTED",
                "container": {"command": ["instrument", "level"]},
            }
        ]
    }


def test_describe_jobs_reports_execution_result(monkeypatch):
    client = LocalBatchClient(run_locally=True)

    def execute_successfully(job_name, argv, record):
        record["returncode"] = 0
        record["stdout"] = ""
        record["stderr"] = ""

    monkeypatch.setattr(client, "_execute_subprocess", execute_successfully)
    submitted = _submit(client)

    assert client.describe_jobs(jobs=[submitted["jobId"]])["jobs"][0]["status"] == (
        "SUCCEEDED"
    )


def test_describe_jobs_omits_unknown_jobs():
    client = LocalBatchClient()
    _submit(client)

    assert client.describe_jobs(jobs=["unknown-job-id"]) == {"jobs": []}
