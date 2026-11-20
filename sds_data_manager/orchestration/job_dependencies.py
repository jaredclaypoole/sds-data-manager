"""Dependency gathering shared by IMAP job handlers."""

from __future__ import annotations

import copy
import datetime
import logging
import os
import re
from typing import TYPE_CHECKING, Self

from imap_data_access import processing_input

from sds_data_manager.orchestration.dependency_metadata import DependencySource

if TYPE_CHECKING:
    from sds_data_manager.orchestration.types import ProcessingJobNode


class MissingDependenciesError(Exception):
    """Error to raise from get_dependencies function if crucial data is not found."""

    pass


class JobDependencyHandler:
    """Gather dependencies using the owning handler's job_config."""

    def __init__(
        self,
        job_config: ProcessingJobNode,
        source: DependencySource,
        logger: logging.Logger | None = None,
    ) -> None:
        """Configure dependency selection without a database or Dagster context."""
        self.job_config = job_config
        self.dependency_source = source
        self.dependency_logger = logger or logging.getLogger(__name__)

    def with_dependency_source(
        self, source: DependencySource, logger: logging.Logger | None = None
    ) -> Self:
        """Bind a run-local copy, preserving custom subclass overrides."""
        handler = copy.copy(self)
        handler.dependency_source = source
        handler.dependency_logger = logger or logging.getLogger(__name__)
        return handler

    def get_ancillary_files_inputs(
        self, target_start: datetime.datetime, target_end: datetime.datetime
    ) -> list[processing_input.AncillaryInput]:
        """Get the ancillary file dependencies needed to cover a time range."""
        # Query for ancillary files where the start_date is less than or equal to
        # the input end_date, and the end_date is either greater than or equal to the
        # input start_date or is None. For example, if the input start_date is
        # '20240524' and the end_date is '20240527', the query could return an ancillary
        # file with the date range ('20240525', '20240528').
        anc_processing_inputs = []
        for input in self.job_config.inputs:
            if input.data_type == "ancillary":
                records = self.dependency_source.ancillary_records(
                    input, target_start, target_end
                )
                records = sorted(records, key=lambda x: x.start_date, reverse=True)[0:1]
                # Check requirement here if needed or not
                if not records and input.required:
                    raise MissingDependenciesError(
                        f"""Missing dependency for
                        {input.to_dagster_name()}
                        between {target_start} and {target_end}"""
                    )
                filenames = [os.path.basename(record.file_path) for record in records]
                anc_processing_inputs.append(
                    processing_input.AncillaryInput(*filenames)
                )
        return anc_processing_inputs

    def get_spin_files_inputs(
        self, target_start: datetime.datetime, target_end: datetime.datetime
    ) -> list[str] | None:
        """Return the spin file dependencies needed to cover a time range."""
        spin_files = self.dependency_source.spin_files(
            target_start.replace(hour=0, minute=0, second=0), target_end
        )
        if not spin_files and self.job_config.spin_input.required:
            raise MissingDependenciesError(
                f"""Missing dependency for
                {self.job_config.spin_input.to_dagster_name()}
                between {target_start} and {target_end}"""
            )
        return spin_files

    def get_repoint_file_inputs(
        self, target_start: datetime.datetime, target_end: datetime.datetime
    ) -> list[str] | None:
        """Return the repoint file needed to cover a time range."""
        file = self.dependency_source.repoint_files(target_start, target_end)
        if not file and self.job_config.repoint_input.required:
            raise MissingDependenciesError(
                f"""Missing dependency for
                {self.job_config.repoint_input.to_dagster_name()}
                between {target_start} and {target_end}"""
            )

        return file

    def get_spice_file_inputs(
        self, target_start: datetime.datetime, target_end: datetime.datetime
    ) -> list[str]:
        """Return the spice files needed to cover a time range."""
        spice_files, missing_kernel_types = self.dependency_source.spice_files(
            self.job_config.spice_types, target_start, target_end
        )
        if (
            missing_kernel_types
            and len(missing_kernel_types) > 0
            and self.job_config.spice_inputs
        ):
            # If there are missing kernels, raise failure
            raise MissingDependenciesError(
                f"Missing SPICE kernel types: {', '.join(missing_kernel_types)} "
                f"Found the following files: {', '.join(spice_files)} "
                f"between {target_start} and {target_end}"
            )
        # spice_files is only None when the metakernel API returned a non-200
        # response.
        if spice_files is None:
            raise MissingDependenciesError(
                f"The Metakernel API request failed when querying for kernel "
                f"types: {', '.join(self.job_config.spice_types)} "
                f"between {target_start} and {target_end}. "
                f"Check the metakernel api logs."
            )

        return spice_files

    def get_science_files_inputs(
        self,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
        repointing: int | None = None,
    ) -> list[processing_input.ScienceInput]:
        """Return the science file dependencies needed to cover a time range."""
        science_processing_inputs = []
        for input in self.job_config.science_inputs:
            dep_name = input.to_dagster_name()
            found_dep = False
            file_groups = self.dependency_source.science_files(
                input, target_start, target_end
            )
            science_files = []
            for file_names in file_groups:
                if file_names is not None:
                    found_dep = True
                    if file_names:
                        self.dependency_logger.info(
                            f"The file names of the matching partition: {file_names}"
                        )
                    science_files.extend(file_names)

            if not found_dep and input.required:
                # If we found nothing and this is required, don't return anything.
                raise MissingDependenciesError(
                    f"""Not enough information to process.
                       Missing {dep_name} in range {target_start!s} to {target_end!s}"""
                )
            if science_files:
                pattern = re.compile(r"v(\d{3})\.(cdf|pkts)$")
                renamed_science_files = [
                    pattern.sub(r"v001.0\1.\2", file) for file in science_files
                ]
                science_processing_inputs.append(
                    processing_input.ScienceInput(*list(set(renamed_science_files)))
                )

        if not science_processing_inputs and self.job_config.science_inputs:
            # Science inputs were configured but none were found.
            raise MissingDependenciesError(
                f"No science files were discovered between {target_start} and "
                f"{target_end}. This job require at least one science file."
            )

        return science_processing_inputs

    def get_dependencies(
        self,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
        repointing: int | None = None,
    ) -> processing_input.ProcessingInputCollection:
        """Get the dependencies for the job using the DependencyResolver."""
        # Iterate through each upstream dependency
        processing_inputs = processing_input.ProcessingInputCollection()
        self.dependency_logger.info(
            f"""Checking for all dependencies existing between
            {target_start} and {target_end}"""
        )

        # Get Science files
        science_files = self.get_science_files_inputs(
            target_start, target_end, repointing
        )
        for inputs in science_files:
            # TODO: ADD IN THE MAJOR VERSION NUMBER HANDLING WHEN CONSTRUCTING
            # ProcessingInputs.
            processing_inputs.add(inputs)

        # Get Ancillary files
        if self.job_config.ancillary_inputs:
            ancillary_files = self.get_ancillary_files_inputs(target_start, target_end)
            if ancillary_files:
                for inputs in ancillary_files:
                    processing_inputs.add(inputs)

        # Get the Repoint file
        if self.job_config.repoint_input:
            repoint = self.get_repoint_file_inputs(target_start, target_end)
            if repoint:
                processing_inputs.add(processing_input.RepointInput(repoint[0]))

        # Get SPICE files
        if self.job_config.spice_inputs:
            spice_files = self.get_spice_file_inputs(target_start, target_end)
            if spice_files:
                processing_inputs.add(processing_input.SPICEInput(*spice_files))

        # Get Spin files
        if self.job_config.spin_input:
            spin_files = self.get_spin_files_inputs(target_start, target_end)
            if spin_files:
                processing_inputs.add(processing_input.SpinInput(*spin_files))

        return processing_inputs
