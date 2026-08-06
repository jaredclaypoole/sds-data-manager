# Download scripts

One script per kind of file. Each takes a required start date and an optional
end date (`YYYYMMDD`, end defaulting to today) and downloads into
`imap_data_access.config["DATA_DIR"]`, which `IMAP_DATA_DIR` sets. Files already
on disk are skipped, so re-running a script tops up the archive rather than
fetching it again. `--dry-run` lists what would be fetched.

| Script | Files | Source |
| --- | --- | --- |
| `spice.py` | SPICE kernels | `/spice-query` |
| `tables.py` | Repoint, spin and thruster | `/repoint-table`, `/spin-table`, `/small-forces-table` |
| `science.py` | Instrument science data | `/query?table=science` |
| `ancillary.py` | Calibration and configuration | `/query?table=ancillary` |
| `oneoff.py` | Specific products or named files | `/query?table=science` |

```sh
python scripts/download/spice.py 20260501 --minimal
python scripts/download/tables.py 20260501 --types repoint spin
python scripts/download/science.py 20260501 20260601 --instrument lo --data-level l0
python scripts/download/science.py 20260501 --product lo l1b histrates
python scripts/download/ancillary.py 20260501 --instrument lo
python scripts/download/oneoff.py 20260501 --product lo l1b histrates
```

To create a local database (see next paragraph),
you can run `python scripts/download/create_db.py`.
Note that postgres needs to be installed on your machine,
and a database named "postgres" needs to exist
(which happens by default).

Downloading only puts files on disk. To make them visible to the local database
(and so to the query APIs and the Dagster sensors), index them afterwards with
`scripts/download/index.py`, which needs the sds-data-manager environment:

```sh
../sds-data-manager/.venv/bin/python scripts/download/index.py --dry-run
../sds-data-manager/.venv/bin/python scripts/download/index.py
../sds-data-manager/.venv/bin/python scripts/download/index.py --create-tables  # empty db
```

`--create-tables` builds any tables that are missing and stamps the alembic
version, which is what an emptied-out database needs: alembic's own root
revision begins by altering tables that nothing in the chain creates, so
`alembic upgrade head` cannot build the schema from nothing.

Notes:

- **Authentication.** `IMAP_API_KEY` (or `IMAP_ACCESS_TOKEN`) selects the
  authenticated endpoints. Without one the archive answers science and ancillary
  queries with an empty list rather than an error, which looks exactly like
  having nothing to download; the scripts say so when they find nothing.
- **SPICE volume.** Kernels supersede each other, so downloading everything that
  touches a date range fetches far more than is needed. `spice.py` keeps only
  the newest version of each kernel by default, and `--minimal` additionally
  drops kernels that a more recently ingested kernel of the same type already
  covers. For a three month range that is the difference between ~360 files and
  ~24.
- **Repoint, spin and thruster files** land under `imap/spice/` but come from
  their own tables rather than the kernel table, which is why `tables.py` is
  separate from `spice.py`. Those endpoints ignore most of their own date
  parameters, so it filters client side.
- **Raw packets** come from WebPODA rather than the SDC archive. Use
  `scripts/01webpoda_download.py`.
