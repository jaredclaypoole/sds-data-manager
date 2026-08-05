"""Materialize a single asset partition in-process, under a debugger.

`dagster dev` runs an asset body three process hops away from your terminal (the
code-location gRPC server launches a run process, which the multiprocess
executor fans out again), so a plain ``breakpoint()`` in the op never gets a
usable stdin. ``dagster.materialize`` instead performs a single-threaded,
in-process run: the asset body executes in *this* process, so pdb and IDE
breakpoints behave normally.

Paired with IMAP_RUN_JOBS_IN_PROCESS the imap_processing CLI is also called
directly rather than shelled out to, which means you can step from run_job all
the way down into instrument code in one session.

Edit ASSET and DATE at the bottom to target something else - the partition key
is looked up from the date, since those keys encode repoint boundaries from
pointing_table and cannot be written by hand. Then:

    poetry run python -m pdb scripts/debug_materialize.py

or point your IDE's debug configuration at this file.
"""

import logging
import os
import sys

from dotenv import load_dotenv

# Must happen before the orchestration modules are imported: DATABASE_URL,
# IMAP_DATA_DIR and DAGSTER_HOME are all read at import time, and a plain script
# does not get the .env loading that `dagster dev` performs for you.
load_dotenv()

# Both are read at import time by local_dagster, so they have to be set before
# the import below - not later, next to the materialize call.
#   IMAP_RUN_JOBS_LOCALLY=0   resolve dependencies and write the dependency JSON,
#                             but record the command instead of running it.
#   IMAP_RUN_JOBS_IN_PROCESS=0  run imap_cli in a subprocess after all, which
#                             isolates spice/logging state but puts instrument
#                             code back out of the debugger's reach.
os.environ["IMAP_RUN_JOBS_LOCALLY"] = "1"
os.environ["IMAP_RUN_JOBS_IN_PROCESS"] = "1"

from dagster import DagsterInstance, materialize  # noqa: E402

from sds_data_manager.orchestration import (  # noqa: E402
    dagster_utilities,
    local_dagster,
)


def partitions_def_name_for(asset_name: str) -> str:
    """Return the name of the partitions definition an asset is partitioned by."""
    for assets_def in local_dagster.defs.assets:
        keys = list(getattr(assets_def, "keys", None) or [assets_def.key])
        if asset_name in [key.to_user_string() for key in keys]:
            partitions_def = getattr(assets_def, "partitions_def", None)
            name = getattr(partitions_def, "name", None)
            if name is None:
                raise RuntimeError(f"{asset_name!r} is not dynamically partitioned")
            return name
    raise RuntimeError(f"no asset named {asset_name!r}")


def partition_key_for_date(
    instance: DagsterInstance,
    asset_name: str,
    date: str,
) -> str:
    """Find the partition key of `asset_name` whose window starts on `date`.

    Partition keys look like 'repoint250_2026-05-16T10:03:28_to_2026-05-17T10:03:12'
    and are created by the sensors in custom_partitions.py, so they cannot be
    constructed by hand - the boundaries come from pointing_table.

    A calendar day overlaps two repoint windows, because repoints fall mid-day.
    Matching on the window's *start* date picks exactly one, and it is the same
    one the file names use: imap_lo_l1b_histrates_20260516-repoint00250 lives in
    the partition starting 2026-05-16T10:03:28.
    """
    def_name = partitions_def_name_for(asset_name)
    keys = instance.get_dynamic_partitions(def_name)
    if not keys:
        raise RuntimeError(
            f"no {def_name} keys registered - run the {def_name} sensor in "
            f"`dagster dev` first (see custom_partitions.py)"
        )

    dated = []
    for key in keys:
        start, _ = dagster_utilities.parse_dates_from_partition_key(key)
        if start:
            dated.append((start, key))
    dated.sort()

    matches = [key for start, key in dated if start.strftime("%Y-%m-%d") == date]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"{date} matches several {def_name} keys: {matches}")

    # Nothing on that date - show the surrounding keys rather than just failing,
    # since the usual cause is asking for a date outside the loaded range.
    ordered = [key for _, key in dated]
    nearby = [*ordered[:2], "...", *ordered[-2:]]
    raise RuntimeError(
        f"no {def_name} key starts on {date}.\n  Registered range:\n    "
        + "\n    ".join(nearby)
    )


def selection_for(asset_name: str) -> list[str]:
    """Return every asset name produced by the node that produces asset_name.

    imap_job builds its multi_assets without can_subset=True, so Dagster rejects
    a selection that covers only part of a node. That matches how the pipeline
    actually works - one imap_cli invocation emits every product of the job, so
    lo_l1b_goodtimes was never separable from lo_l1b_bgrates - but it means the
    selection has to name all of them.
    """
    for assets_def in local_dagster.defs.assets:
        # defs.assets holds AssetsDefinitions (.keys) and AssetSpecs (.key).
        keys = list(getattr(assets_def, "keys", None) or [assets_def.key])
        names = [key.to_user_string() for key in keys]
        if asset_name in names:
            return names
    raise RuntimeError(f"no asset named {asset_name!r}")


if __name__ == "__main__":
    ASSET = "lo_l1b_goodtimes"
    DATE = "2026-05-16"
    # ASSET = "lo_l2_l090enansnbshsfnspramhae6deg6mo"
    # DATE = "2026-01-17"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    selection = selection_for(ASSET)
    print(f"selecting: {selection}")

    # DagsterInstance.get() reads DAGSTER_HOME. This matters more than it looks:
    # these partitions are dynamic, so the key only exists because the
    # add_repoint_partitions sensor registered it against that instance. An
    # ephemeral instance (the default when this argument is omitted) would not
    # know the key and the run would fail before reaching any asset code.
    with DagsterInstance.get() as instance:
        partition_key = partition_key_for_date(instance, ASSET, DATE)
        print(f"partition: {partition_key}")

        result = materialize(
            list(local_dagster.defs.assets),
            # These assets declare upstreams via `deps=`, not `ins=`, so nothing
            # tries to load upstream contents and selecting one node is valid.
            selection=selection,
            partition_key=partition_key,
            instance=instance,
        )

    # Reported rather than assumed: these are multi_assets with optional outs,
    # so a run can succeed while emitting only some of a job's products.
    for event in result.get_asset_materialization_events():
        print(f"materialized: {event.asset_key.to_user_string()}")

    print(f"success: {result.success}")
    sys.exit(0 if result.success else 1)
