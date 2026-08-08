"""Local drop-in replacements for AWS boto3 clients.

These classes intercept boto3 API calls and redirect them to local execution,
making it possible to run the orchestration code without real AWS infrastructure.

They are only wired in by ``sds_data_manager.orchestration.local_dagster``; nothing
imports them on the deployed path, so AWS behavior is unchanged.
"""

import hashlib
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches the shape of a real ECR image URI so that the parsing in
# imap_job._get_container_image_digest works unmodified:
#   <registryId>.dkr.ecr.<region>.amazonaws.com/<repositoryName>:<tag>
LOCAL_REGISTRY_ID = "000000000000"
LOCAL_REGISTRY = f"{LOCAL_REGISTRY_ID}.dkr.ecr.us-west-2.amazonaws.com"


# N818: the name has to match boto3's exactly - imap_job catches this by
# attribute off the client, so an ...Error rename would break the lookup.
class ImageNotFoundException(Exception):  # noqa: N818
    """Stand-in for ECR_CLIENT.exceptions.ImageNotFoundException."""


class _Exceptions:
    """Namespace mimicking a boto3 client's ``.exceptions`` attribute."""

    ImageNotFoundException = ImageNotFoundException


class LocalECRClient:
    """Intercept ECR API calls and return synthetic image metadata.

    Replace ``imap_job.ECR_CLIENT`` with an instance of this class.
    """

    exceptions = _Exceptions

    def __init__(self, processing_version: str | None = None):
        """Record the imap_processing version the digest should be derived from.

        Parameters
        ----------
        processing_version : str, optional
            Version string to fold into the synthetic digest. Defaults to the
            installed imap_processing version.
        """
        if processing_version is None:
            try:
                import imap_processing  # noqa: PLC0415

                processing_version = imap_processing.__version__
            except ImportError:
                processing_version = "unknown"
        self.processing_version = processing_version

    def describe_images(self, registryId, repositoryName, imageIds):  # noqa: N803
        """Return a stable synthetic digest for a local "image".

        The digest is derived from the repository, tag and installed
        imap_processing version. imap_job folds this digest into the dependency
        hash, which is how duplicate jobs are detected, so it must be stable
        across runs but must change when the processing code changes -
        otherwise upgrading imap_processing would leave old jobs looking
        identical and they would be skipped.
        """
        tag = imageIds[0].get("imageTag", "latest")
        seed = f"{repositoryName}:{tag}:{self.processing_version}"
        digest = hashlib.sha256(seed.encode()).hexdigest()
        return {"imageDetails": [{"imageDigest": f"sha256:{digest}"}]}


class LocalBatchClient:
    """Intercept AWS Batch API calls and run jobs locally via subprocess.

    Replace ``imap_job.BATCH_CLIENT`` with an instance of this class.
    """

    def __init__(
        self,
        run_locally: bool = False,
        timeout: int = 3600,
        allow_upload: bool = False,
        on_complete=None,
        run_in_process: bool = False,
    ):
        """Configure how submitted jobs are handled.

        Parameters
        ----------
        run_locally : bool
            When True, submit_job actually runs imap_processing.cli and blocks
            until it finishes. When False (the default) the command is recorded
            and logged but not executed, which is the safer way to inspect what
            the pipeline wants to do.
        timeout : int
            Seconds to allow a locally executed job before killing it. Only
            applies to subprocess execution; an in-process call cannot be
            interrupted this way.
        allow_upload : bool
            imap_job always appends --upload-to-sdc, and imap-data-access is
            normally configured against the real SDC with a live API key, so a
            local run would push locally-produced files to production. The flag
            is stripped unless this is explicitly set True.
        on_complete : callable, optional
            Called with the submission record after a locally executed job
            finishes. Used to stand in for the S3/Batch events that would
            normally trigger the indexer lambda.
        run_in_process : bool
            Call imap_processing.cli.main() directly instead of spawning a
            subprocess, so a debugger attached to the Dagster process can step
            into instrument code. Only meaningful alongside run_locally.

            This is a debugging aid, not a faster execution mode. Dropping the
            process boundary means spiceypy's kernel pool, imap_processing's
            logging configuration and any leaked memory now persist across every
            job in the run, and a crash in a CDF library takes the whole Dagster
            process with it rather than failing one job. Prefer the subprocess
            path for anything resembling a backfill.
        """
        self.run_locally = run_locally
        self.timeout = timeout
        self.allow_upload = allow_upload
        self.on_complete = on_complete
        self.run_in_process = run_in_process
        self.submitted: list[dict] = []
        self._counter = 0

    def describe_job_definitions(self, jobDefinitionName, status=None):  # noqa: N803
        """Return a synthetic ACTIVE job definition for any requested name.

        The image URI is built to look like a real ECR URI because
        imap_job._get_container_image_digest parses it apart to call ECR.
        """
        # "ProcessingJob-swapi" or "ProcessingJob-swapi-l3" -> "swapi"
        instrument = jobDefinitionName.split("-")[1]
        return {
            "jobDefinitions": [
                {
                    "jobDefinitionName": jobDefinitionName,
                    "revision": 1,
                    "status": "ACTIVE",
                    "containerProperties": {
                        "image": f"{LOCAL_REGISTRY}/{instrument}-repo:latest"
                    },
                }
            ]
        }

    def describe_jobs(self, jobs):
        """Return submitted local jobs using the boto3 Batch response shape.

        Unknown job IDs are omitted, matching AWS Batch. Jobs that were only
        recorded are reported as ``SUBMITTED``; executed jobs are reported as
        ``SUCCEEDED`` or ``FAILED`` according to their return code.
        """
        requested_job_ids = set(jobs)
        described_jobs = []
        for record in self.submitted:
            if record["jobId"] not in requested_job_ids:
                continue
            if "returncode" not in record:
                status = "SUBMITTED"
            elif record["returncode"] == 0:
                status = "SUCCEEDED"
            else:
                status = "FAILED"
            described_jobs.append(
                {
                    "jobId": record["jobId"],
                    "jobName": record["jobName"],
                    "jobQueue": record["jobQueue"],
                    "jobDefinition": record["jobDefinition"],
                    "status": status,
                    "container": {"command": record["command"]},
                }
            )
        return {"jobs": described_jobs}

    def submit_job(
        self,
        jobName,  # noqa: N803
        jobQueue,  # noqa: N803
        jobDefinition,  # noqa: N803
        containerOverrides,  # noqa: N803
        retryStrategy=None,  # noqa: N803
        **kwargs,
    ):
        """Run the batch command locally instead of submitting it to AWS.

        The command built by imap_job is the imap_processing CLI argument list,
        so it is handed to ``python -m imap_processing.cli`` unchanged.
        """
        self._counter += 1
        job_id = f"local-{self._counter}"
        command = list(containerOverrides["command"])
        if not self.allow_upload and "--upload-to-sdc" in command:
            command.remove("--upload-to-sdc")
            logger.info(
                "[local batch] stripped --upload-to-sdc from %s; pass "
                "allow_upload=True to publish to the real SDC",
                jobName,
            )
        record = {
            "jobId": job_id,
            "jobName": jobName,
            "jobQueue": jobQueue,
            "jobDefinition": jobDefinition,
            "command": command,
        }
        self.submitted.append(record)

        argv = [sys.executable, "-m", "imap_processing.cli", *command]
        if not self.run_locally:
            logger.info(
                "[local batch] recorded (not executed) %s: %s",
                jobName,
                " ".join(argv),
            )
            return {"jobId": job_id, "jobName": jobName}

        if self.run_in_process:
            self._execute_in_process(jobName, command, record)
        else:
            self._execute_subprocess(jobName, argv, record)

        if record["returncode"] != 0:
            logger.error(
                "[local batch] %s failed (rc=%s)\n%s",
                jobName,
                record["returncode"],
                record.get("stderr", "")[-4000:],
            )
        else:
            logger.info("[local batch] %s finished successfully", jobName)

        if self.on_complete is not None:
            # Stands in for the S3 / Batch state-change events that trigger the
            # indexer lambda on AWS. Failing here must not look like a job
            # failure, so it is logged rather than raised.
            try:
                self.on_complete(record)
            except Exception:
                logger.exception("[local batch] on_complete failed for %s", jobName)

        return {"jobId": job_id, "jobName": jobName}

    def _execute_subprocess(self, jobName, argv, record):  # noqa: N803
        """Run the CLI in a child process and capture its output."""
        logger.info("[local batch] running %s: %s", jobName, " ".join(argv))
        result = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=os.environ.copy(),
            check=False,
        )
        record["returncode"] = result.returncode
        record["stdout"] = result.stdout
        record["stderr"] = result.stderr

    def _execute_in_process(self, jobName, command, record):  # noqa: N803
        """Call the CLI's entrypoint directly so a debugger can step into it.

        ``imap_processing.cli.main`` takes no arguments and reads ``sys.argv``,
        so the argument list is swapped in around the call. Anything the CLI
        would have communicated through an exit status has to be translated back
        into a returncode here, because callers (and on_complete) only look at
        that field.
        """
        # Imported lazily: the deployed path never takes this branch, and the
        # import pulls in the full instrument stack.
        import imap_processing.cli  # noqa: PLC0415

        logger.info(
            "[local batch] running %s in-process: imap_cli %s",
            jobName,
            " ".join(command),
        )
        saved_argv = sys.argv
        sys.argv = ["imap_cli", *command]
        try:
            imap_processing.cli.main()
            record["returncode"] = 0
            record["stderr"] = ""
        except SystemExit as exc:
            # argparse calls sys.exit() on a bad argument list rather than
            # raising, and main() itself may exit non-zero.
            code = exc.code
            record["returncode"] = 0 if code is None else int(code)
            record["stderr"] = f"SystemExit({code})"
        except Exception as exc:
            # A subprocess run would have turned this into a returncode, so do
            # the same rather than letting it escape into the Dagster op and
            # skip the on_complete indexing hook.
            logger.exception("[local batch] %s raised in-process", jobName)
            record["returncode"] = 1
            record["stderr"] = traceback.format_exc()
            record["exception"] = exc
        finally:
            sys.argv = saved_argv


class LocalUploadResponse:
    """Minimal stand-in for the requests.Response returned after an S3 upload."""

    def __init__(self, status_code: int, path):
        """Store the resulting status code and the path written to."""
        self.status_code = status_code
        self.path = path

    def __bool__(self):
        """Callers only test truthiness to decide whether the upload worked."""
        return 200 <= self.status_code < 300


def local_upload_dependency_file(self, dependency_file_path, serialized_dependencies):
    """Write a job's dependency JSON to the local data directory.

    Drop-in replacement for ``IMAPJobHandler.upload_dependency_file``, which
    normally asks the upload API for a presigned URL and PUTs the file to S3.

    The file is written to the path imap-data-access already computed for it,
    which is what makes this work end to end: ``imap_data_access.download()``
    returns early when the destination already exists, so imap_cli resolves
    ``--dependency <name>.json`` from disk without any network call.

    The real method raises KeyError if the file exists, which never happens on a
    fresh AWS container but happens constantly on reruns locally, so identical
    content is treated as reuse instead - mirroring the 409 branch upstream.
    """
    path = Path(dependency_file_path)

    if path.is_file():
        existing = path.read_text()
        if existing == serialized_dependencies:
            logger.info("[local upload] dependency file already present: %s", path)
            return LocalUploadResponse(200, path)
        logger.warning(
            "[local upload] overwriting dependency file with changed content: %s", path
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized_dependencies)
    logger.info("[local upload] wrote dependency file: %s", path)
    return LocalUploadResponse(200, path)


class LocalSQSClient:
    """Intercept SQS API calls for local running.

    Replace ``imap_job.SQS_CLIENT`` and ``reprocessing.SQS_CLIENT`` with an
    instance of this class. Messages are held in memory; receive_message
    returns nothing, so the reprocessing sensor simply finds an empty queue.
    """

    def __init__(self):
        """Start with an empty in-memory queue."""
        self.messages: list[dict] = []
        self.deleted: list[dict] = []

    @staticmethod
    def _ok(**extra):
        return {"ResponseMetadata": {"HTTPStatusCode": 200}, **extra}

    def send_message(self, QueueUrl, MessageBody, **kwargs):  # noqa: N803
        """Record a message instead of sending it."""
        self.messages.append({"QueueUrl": QueueUrl, "Body": MessageBody})
        logger.info("[local sqs] send_message to %s: %s", QueueUrl, MessageBody)
        return self._ok(MessageId=f"local-{len(self.messages)}")

    def receive_message(self, QueueUrl, **kwargs):  # noqa: N803
        """Return no messages, so reprocessing sensors idle harmlessly."""
        return self._ok(Messages=[])

    def delete_message(self, QueueUrl, ReceiptHandle, **kwargs):  # noqa: N803
        """Record the delete instead of performing it."""
        self.deleted.append({"QueueUrl": QueueUrl, "ReceiptHandle": ReceiptHandle})
        return self._ok()
