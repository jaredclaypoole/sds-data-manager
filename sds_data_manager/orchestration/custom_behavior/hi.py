"""Override behavior for HI processing."""

import re

import numpy as np
from imap_data_access import processing_input

from sds_data_manager.orchestration import imap_job, types
from sds_data_manager.orchestration.job_handler_registry import JobBuilderRegistry

HI_GOODTIMES_NUM_NEAREST_REPOINTS = 8


@JobBuilderRegistry.register("hi", "l1b", "45sensor-goodtimes")
@JobBuilderRegistry.register("hi", "l1b", "90sensor-goodtimes")
class HiGoodtimesJob(imap_job.IMAPJobHandler):
    """Overriding parts of the Hi processing pipeline."""

    # Override this function from IMAPJobHandler
    def get_science_files_inputs(self, target_start, target_end, repointing=None):
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
                raise imap_job.MissingDependenciesError(
                    f"Hi Goodtimes: skipping repoint {target_pointing_number} "
                    "- a neighboring repoint's L1B DE job is in progress."
                )

            num_future = np.sum(np.array(repoint_list) > target_pointing_number)
            min_future_repoints = HI_GOODTIMES_NUM_NEAREST_REPOINTS // 2
            if num_future < min_future_repoints:
                # With data every repoint the window is +/- 4, so wait only
                # for its future edge. Once that pointing is in
                # PointingTable, missing Hi DE means Hi has no data there.
                required_future_pointing = target_pointing_number + min_future_repoints
                if not self.dependency_source.pointing_exists(required_future_pointing):
                    raise imap_job.MissingDependenciesError(
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
