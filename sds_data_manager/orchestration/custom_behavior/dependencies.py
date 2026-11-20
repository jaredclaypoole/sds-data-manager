"""Product-specific dependency selection shared by local and Dagster callers."""

from __future__ import annotations

import datetime
import re

import numpy as np
from imap_data_access import processing_input

from sds_data_manager.orchestration import types
from sds_data_manager.orchestration.dependency_handler_registry import (
    DependencyHandlerRegistry,
)
from sds_data_manager.orchestration.job_dependencies import (
    JobDependencyHandler,
    MissingDependenciesError,
)

HI_GOODTIMES_NUM_NEAREST_REPOINTS = 8


@DependencyHandlerRegistry.register("hi", "l1b", "45sensor-goodtimes")
@DependencyHandlerRegistry.register("hi", "l1b", "90sensor-goodtimes")
class HiGoodtimesDependencies(JobDependencyHandler):
    """Overriding parts of the Hi processing pipeline."""

    # Override this function from IMAPJobHandler
    def get_science_files_inputs(
        self,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
        repointing: int | None = None,
    ) -> list[processing_input.ScienceInput]:
        """Override the behavior of IMAPJobHandler.get_science_files_inputs."""
        target_pointing_number = repointing

        science_processing_inputs = super().get_science_files_inputs(
            target_start, target_end, repointing
        )

        for input in self.job_config.science_inputs:
            if "-de" not in input.descriptor:
                continue

            repoint_list = self.get_n_nearest_repoints(input, target_pointing_number)
            # A neighboring repoint's L1B DE job is still INPROGRESS.
            # Wait for it to resolve rather than using a stale file.
            if repoint_list is None:
                raise MissingDependenciesError(
                    f"Hi Goodtimes: skipping repoint {target_pointing_number} "
                    "- a neighboring repoint's L1B DE job is in progress."
                )

            num_future = np.sum(np.array(repoint_list) > target_pointing_number)
            min_future_repoints = HI_GOODTIMES_NUM_NEAREST_REPOINTS // 2
            if self.wait_for_dependencies and num_future < min_future_repoints:
                # With data every repoint the window is +/- 4, so wait only
                # for its future edge. Once that pointing is in
                # PointingTable, missing Hi DE means Hi has no data there.
                required_future_pointing = target_pointing_number + min_future_repoints
                if not self.dependency_source.pointing_exists(required_future_pointing):
                    raise MissingDependenciesError(
                        f"Hi Goodtimes: skipping repoint "
                        f"{target_pointing_number} - pointing "
                        f"{required_future_pointing} does not exist yet, "
                        "waiting for more data to fill in."
                    )

            if not repoint_list:
                continue

            file_groups = self.dependency_source.science_files_by_repoint(
                input, repoint_list
            )
            neighbor_files = []
            for file_names in file_groups:
                if file_names is not None:
                    neighbor_files.extend(file_names)

            if neighbor_files:
                # Apply the same version-renaming strategy as
                # IMAPJobHandler.get_science_files_inputs so neighboring
                # files are named consistently with the base science
                # inputs. This implementation will leave filenames with the
                # new versioning convention unaltered.
                pattern = re.compile(r"v(\d{3})\.(cdf|pkts)$")
                renamed_neighbor_files = [
                    pattern.sub(r"v001.0\1.\2", file) for file in neighbor_files
                ]
                self.dependency_logger.info(
                    "Hi Goodtimes adding neighboring L1B DE files: "
                    f"{renamed_neighbor_files}"
                )
                science_processing_inputs.append(
                    processing_input.ScienceInput(*list(set(renamed_neighbor_files)))
                )
        return science_processing_inputs

    def get_n_nearest_repoints(
        self,
        dependency: types.DependencyNode,
        repoint: int,
    ) -> list | None:
        """Get N files nearest to a target repoint.

        Finds N files nearest by repoint number. Does NOT include the target
        repoint itself.

        Parameters
        ----------
        dependency : types.DependencyNode
            Dataclass containing source, data_type, descriptor.
        repoint : int
            Target repoint number.

        Returns
        -------
        dict or None
            Metadata records for N nearest files. Empty list if no neighbors
            exist. None if skip_if_inprogress=True and any of N nearest are
            INPROGRESS.
        """
        # Get available repoints from existing files
        available_repoints = np.array(
            self.dependency_source.available_repoints(dependency)
        )

        # Also get inprogress repoints from running jobs
        inprogress_repoints = np.array(
            self.dependency_source.inprogress_repoints(dependency)
            if self.wait_for_dependencies
            else []
        )
        all_repoints = np.union1d(available_repoints, inprogress_repoints)

        # Verify target exists (in available files or inprogress jobs)
        if repoint not in all_repoints:
            self.dependency_logger.info(
                f"Target repoint {repoint} not found for {dependency}"
            )
            return []

        # Remove target, sort by distance then repoint, take N nearest
        other_repoints = all_repoints[all_repoints != repoint]
        if len(other_repoints) == 0:
            return []

        distances = np.abs(other_repoints - repoint)
        sort_indices = np.lexsort((other_repoints, distances))
        nearest_repoints = other_repoints[sort_indices][
            :HI_GOODTIMES_NUM_NEAREST_REPOINTS
        ]

        # Check if any of N nearest are inprogress
        if len(inprogress_repoints) > 0:
            inprogress_nearest = nearest_repoints[
                np.isin(nearest_repoints, inprogress_repoints)
            ]
            if len(inprogress_nearest) > 0:
                self.dependency_logger.info(
                    f"Skipping: nearest repoints {inprogress_nearest.tolist()} "
                    f"have INPROGRESS jobs for {dependency}"
                )
                return None

        # Get actual records via get_files (handles versioning)
        nearest_repoints_list = nearest_repoints.tolist()

        return nearest_repoints_list


@DependencyHandlerRegistry.register("mag", "l1d", "norm-srf")
class MagL1DDependencies(JobDependencyHandler):
    """Query SPICE for the padded MAG L1D processing window."""

    def get_spice_file_inputs(
        self,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
    ) -> list[str]:
        """Include the 30-minute buffer on either side of the science day."""
        return super().get_spice_file_inputs(
            target_start - datetime.timedelta(minutes=30),
            target_end + datetime.timedelta(minutes=30),
        )


@DependencyHandlerRegistry.register("mag", "l1c", "norm-mago")
@DependencyHandlerRegistry.register("mag", "l1c", "norm-magi")
class MagL1CDependencies(JobDependencyHandler):
    """Include the previous day's L1C when it exists."""

    def get_science_files_inputs(
        self,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
        repointing: int | None = None,
    ) -> list[processing_input.ScienceInput]:
        """Return the base science inputs plus the previous day's L1C, if any."""
        science_processing_inputs = super().get_science_files_inputs(
            target_start, target_end, repointing
        )

        previous_day_l1c = types.DependencyNode(
            source="mag",
            data_type="l1c",
            descriptor=self.job_config.descriptor,
            required=False,
            trigger_job=False,
        )
        # The query window ends at target_start so that the strict overlap
        # check in get_all_files_in_time_range cannot match the current day's
        # own partition, which starts exactly at target_start. The job never
        # receives its own output as input.
        file_groups = self.dependency_source.science_files(
            previous_day_l1c,
            target_start - datetime.timedelta(days=1),
            target_start,
        )

        if not file_groups:
            self.dependency_logger.info(
                "No previous day L1C found; MAG L1C processes this day alone."
            )
            return science_processing_inputs

        # This one-day window can only overlap the previous day's partition,
        # whose latest materialization carries a single file.
        previous_day_file = file_groups[0][0]

        # Apply the same version-renaming strategy as
        # IMAPJobHandler.get_science_files_inputs so the previous day's file is
        # named consistently with the base science inputs.
        pattern = re.compile(r"v(\d{3})\.(cdf|pkts)$")
        renamed_previous_day_file = pattern.sub(r"v001.0\1.\2", previous_day_file)
        self.dependency_logger.info(
            f"MAG L1C adding the previous day's L1C: {renamed_previous_day_file}"
        )
        science_processing_inputs.append(
            processing_input.ScienceInput(renamed_previous_day_file)
        )

        return science_processing_inputs


@DependencyHandlerRegistry.register("lo", "l3", "all-maps")
@DependencyHandlerRegistry.register("hi", "l3", "sp-maps")
@DependencyHandlerRegistry.register("hi", "l3", "hic-maps")
@DependencyHandlerRegistry.register("glows", "l3b", "ion-rate-profile")
@DependencyHandlerRegistry.register("ultra", "l3", "u45-maps")
@DependencyHandlerRegistry.register("ultra", "l3", "u90-maps")
@DependencyHandlerRegistry.register("ultra", "l3", "ulc-sp-maps")
@DependencyHandlerRegistry.register("ultra", "l3", "ulc-nsp-maps")
class L3Dependencies(JobDependencyHandler):
    """Let L3 processing discover science inputs and use the latest repoint file."""

    def get_science_files_inputs(
        self,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
        repointing: int | None = None,
    ) -> list[processing_input.ScienceInput]:
        """Return no science files without an error."""
        return []

    def get_repoint_file_inputs(
        self,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
    ) -> list[str] | None:
        """Return the repoint file needed to cover a time range."""
        min_python_date = datetime.datetime(1, 1, 1)
        file = self.dependency_source.repoint_files(min_python_date, min_python_date)
        return file


@DependencyHandlerRegistry.register("spacecraft", "l1a", "pointing-attitude")
class PointingAttitudeDependencies(JobDependencyHandler):
    """Build pointing-attitude dependencies without science inputs."""

    def get_science_files_inputs(
        self,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
        repointing: int | None = None,
    ) -> list[processing_input.ScienceInput]:
        """Override default behavior to return nothing."""
        return []
