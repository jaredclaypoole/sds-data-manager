"""Dependency gathering shared by IMAP job handlers."""

import datetime
import os
import re

from dagster import AssetExecutionContext
from imap_data_access import processing_input
from sqlalchemy import and_, func, or_

from sds_data_manager.lambda_code.SDSCode.database import models
from sds_data_manager.orchestration import repoint_file, spice, spin


class MissingDependenciesError(Exception):
    """Error to raise from get_dependencies function if crucial data is not found."""

    pass


class JobDependencyHandler:
    """Gather dependencies using the owning handler's job_config."""

    def get_ancillary_files_inputs(
        self, session, target_start: datetime.datetime, target_end: datetime.datetime
    ) -> list[str]:
        """Get the ancillary file dependencies needed to cover a time range."""
        # Query for ancillary files where the start_date is less than or equal to
        # the input end_date, and the end_date is either greater than or equal to the
        # input start_date or is None. For example, if the input start_date is
        # '20240524' and the end_date is '20240527', the query could return an ancillary
        # file with the date range ('20240525', '20240528').
        anc_processing_inputs = []
        for input in self.job_config.inputs:
            if input.data_type == "ancillary":
                table = models.AncillaryFiles
                type_specific_conditions = []
                type_specific_conditions.append(
                    and_(
                        table.start_date < target_end,
                        or_(table.end_date >= target_start, table.end_date.is_(None)),
                    )
                )
                filter_conditions = [
                    table.instrument == input.source,
                    table.descriptor == input.descriptor,
                    *type_specific_conditions,
                ]
                max_version_query = (
                    session.query(
                        table.start_date,
                        func.max(table.version).label("latest_version"),
                    )
                    .filter(*filter_conditions)
                    .group_by(table.start_date)
                    .subquery()
                )
                # Query records
                records = (
                    session.query(table)
                    .join(
                        max_version_query,
                        (table.start_date == max_version_query.c.start_date)
                        & (table.version == max_version_query.c.latest_version),
                    )
                    .filter(*filter_conditions)
                    .all()
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
        self, session, target_start: datetime.datetime, target_end: datetime.datetime
    ) -> list[str]:
        """Return the spin file dependencies needed to cover a time range."""
        spin_files = spin.get_upstream_dependency_inputs_spin(
            target_start.replace(hour=0, minute=0, second=0), target_end, False, session
        )
        if not spin_files and self.job_config.spin_input.required:
            raise MissingDependenciesError(
                f"""Missing dependency for
                {self.job_config.spin_input.to_dagster_name()}
                between {target_start} and {target_end}"""
            )
        return spin_files

    def get_repoint_file_inputs(self, session, target_start, target_end):
        """Return the repoint file needed to cover a time range."""
        file = repoint_file.get_upstream_dependency_inputs_repoint(
            target_start, target_end, session
        )
        if not file and self.job_config.repoint_input.required:
            raise MissingDependenciesError(
                f"""Missing dependency for
                {self.job_config.repoint_input.to_dagster_name()}
                between {target_start} and {target_end}"""
            )

        return file

    def get_spice_file_inputs(self, session, target_start, target_end):
        """Return the spice files needed to cover a time range."""
        spice_files, missing_kernel_types = spice.get_upstream_dependency_inputs_spice(
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

    def get_science_files_inputs(self, context, target_start, target_end):
        """Return the science file dependencies needed to cover a time range."""
        science_processing_inputs = []
        for input in self.job_config.science_inputs:
            dep_name = input.to_dagster_name()
            found_dep = False
            metadata_list = input.get_all_files_in_time_range(
                context, target_start, target_end
            )
            science_files = []
            for metadata in metadata_list:
                if "file_names" in metadata:
                    found_dep = (
                        True  # We can finally say we have found at least one dependency
                    )
                    # Dagster wraps metadata in a MetadataValue object,
                    # so we call .value
                    file_names = metadata["file_names"].value
                    # Handle both single strings and lists of files safely
                    if isinstance(file_names, str):
                        file_names = [file_names]
                    if file_names:
                        context.log.info(
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
        session,
        context: AssetExecutionContext,
        target_start: datetime.datetime,
        target_end: datetime.datetime,
    ):
        """Get the dependencies for the job using the DependencyResolver."""
        # Iterate through each upstream dependency
        processing_inputs = processing_input.ProcessingInputCollection()
        context.log.info(
            f"""Checking for all dependencies existing between
            {target_start} and {target_end}"""
        )

        # Get Science files
        science_files = self.get_science_files_inputs(context, target_start, target_end)
        for inputs in science_files:
            # TODO: ADD IN THE MAJOR VERSION NUMBER HANDLING WHEN CONSTRUCTING
            # ProcessingInputs.
            processing_inputs.add(inputs)

        # Get Ancillary files
        if self.job_config.ancillary_inputs:
            ancillary_files = self.get_ancillary_files_inputs(
                session, target_start, target_end
            )
            if ancillary_files:
                for inputs in ancillary_files:
                    processing_inputs.add(inputs)

        # Get the Repoint file
        if self.job_config.repoint_input:
            repoint = self.get_repoint_file_inputs(session, target_start, target_end)
            if repoint:
                processing_inputs.add(processing_input.RepointInput(repoint[0]))

        # Get SPICE files
        if self.job_config.spice_inputs:
            spice_files = self.get_spice_file_inputs(session, target_start, target_end)
            if spice_files:
                processing_inputs.add(processing_input.SPICEInput(*spice_files))

        # Get Spin files
        if self.job_config.spin_input:
            spin_files = self.get_spin_files_inputs(session, target_start, target_end)
            if spin_files:
                processing_inputs.add(processing_input.SpinInput(*spin_files))

        return processing_inputs
