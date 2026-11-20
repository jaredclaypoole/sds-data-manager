# Prepare dependency JSON locally

Resolve a fresh dependency set for a configured processing job using the data-access
APIs and the same dependency handlers as production. No AWS credentials, direct
database access, Dagster instance, input-file download, upload, or job submission
is needed.

With `IMAP_API_KEY` set in your environment, run from the repository:

```bash
poetry run python -m scripts.dependency.prepare_local_job \
  --instrument mag \
  --data-level l1b \
  --descriptor norm-magi \
  --start-date 20260101 \
  --end-date 20260102 \
  --output-dir ./dependencies
```

The descriptor identifies the **job** in the dependency YAML; it can differ from an
output descriptor (e.g. MAG L1A uses `all`). Dates accept `YYYYMMDD` or ISO timestamps.
Naive times mean UTC; offsets are converted to UTC. End times are exclusive.
For repoint-partitioned jobs, also supply `--repointing 123` and that repoint's time
window. The window is not inferred from the pointing table.

The command prints the local JSON path. Filenames use `DependencyFilePath` with no
dependency hash; existing files are never overwritten. The payload contains the
same `dependency` and `version` fields used by processing. Use the path with local
processing's `--dependency` option. Processing can download the inputs afterward.

## What is shared and what differs locally

`APIDependencySource` implements `DependencySource`. Configuration and product rules
come from `DependencyConfigReader` and `DependencyHandlerRegistry`, including MAG's
previous-day L1C input and padded L1D SPICE window, Hi's nearest DE repoints, and
L3's omission of science inputs. No parallel local resolver or product registry is
maintained.

The API source uses current indexed science files instead of Dagster materialization
history. It selects the latest major/minor version per file series, groups files by
date/repoint, and filters out the exclusive end boundary. Science queries use file
start **days**. Provide the product's full processing window: an upstream product
starting before the requested start day is not discovered merely because its
coverage extends into the window. Intraday repoint jobs require an explicit repoint
number; daily inputs are retained alongside that repoint's files.

Ancillary queries include older open-ended files and return the highest version per
overlapping start date; the shared handler selects the newest start. Spin queries
filter overlapping records locally and use the shared supersession/priority rules.
Repoint queries choose the latest filename and check end coverage; L3 keeps its
existing relaxed coverage rule. SPICE selection uses `/metakernel?list_files=true`,
the shared approximate J2000 conversion, and the production kernel-presence checks.
As in production's dependency gathering, complete SPICE coverage is not required.

Local preparation passes `wait_for_dependencies=False`. Hi still selects the same
nearest available DE repoints, but does not consult running jobs or wait for future
pointing-table entries. The API source deliberately rejects readiness queries
rather than fabricating metadata. Production handlers default to waiting as before.

Output major versions come from YAML. A shared next minor version is estimated as
one above the largest minor in indexed outputs for the start day/repoint, or one
when none exist. Science queries use the API's latest-major default. Running jobs
and production version reservation are ignored; SPICE outputs use the initial local
estimate. This filename/version is for local processing, not production submission.

## Authentication and implementation

`imap-data-access` reads `IMAP_API_KEY` and `IMAP_DATA_ACCESS_URL` (production by
default). Science and ancillary reads use its public `query()` function. For
`/spin-table`, `/repoint-table`, and `/metakernel`, `DataAccessClient` isolates calls
to the library's private `_get_base_url` and `_make_request` helpers because public
wrappers do not exist. This preserves API-key/token authentication and HTTP error
handling. HTTP and missing-required-input errors prevent a dependency file from
being written.
