"""SPICE kernel categories and load order, independent of AWS access."""

from collections.abc import Collection
from dataclasses import dataclass, field
from enum import Enum, auto


class LeapsecondKernels(Enum):
    """Container for Leapsecond Kernel Types."""

    LEAPSECONDS = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "leapseconds_category"


class PlanetaryConstantsKernels(Enum):
    """Container for Planetary Contants Kernel Types."""

    PLANETARY_CONSTANTS = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "planetary_constants_category"


class ScienceFramesKernels(Enum):
    """Container for Science Frames Kernel Type."""

    SCIENCE_FRAMES = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "science_frames_category"


class IMAPFramesKernels(Enum):
    """Container for IMAP Frames Kernel Type."""

    IMAP_FRAMES = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "imap_frames_category"


class SpacecraftClockKernels(Enum):
    """Container for Spacecraft Clock Kernel Types."""

    SPACECRAFT_CLOCK = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "spacecraft_clock_category"


class PlanetaryEphemerisKernels(Enum):
    """Container for Planetary Ephemeris Kernel Types."""

    PLANETARY_EPHEMERIS = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "planetary_ephemeris_category"


class SpacecraftEphemerisKernels(Enum):
    """Container for Spacecraft Ephemeris Kernel Types, highest priority first."""

    EPHEMERIS_RECONSTRUCTED = auto()
    # Predicted ephemeris comes from each new orbit determination and is
    # delivered more often than nominal ephemeris, so it is preferred.
    EPHEMERIS_PREDICTED = auto()
    EPHEMERIS_NOMINAL = auto()
    EPHEMERIS_90DAYS = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "spacecraft_ephemeris_category"


class SpacecraftAttitudeKernels(Enum):
    """Container for Spacecraft Attitude Kernel Types."""

    ATTITUDE_HISTORY = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "spacecraft_attitude_category"


class EarthAttitudeKernels(Enum):
    """Container for Earth Attitude Kernel Types."""

    EARTH_ATTITUDE = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "earth_attitude_category"


class PointingAttitudeKernels(Enum):
    """Container for Pointing Attitude Kernel Types."""

    POINTING_ATTITUDE = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "pointing_attitude_category"


class LoPivotAttitudeKernels(Enum):
    """Container for Lo Pivot Attitude Kernel Types."""

    LO_PIVOT_ATTITUDE = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "lo_pivot_attitude_category"


class SpinTables(Enum):
    """Container for spin file types. These are CSV files, not SPICE kernels."""

    SPIN = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "spin_category"


class ThrusterFiles(Enum):
    """Container for small forces file types. These are not SPICE kernels."""

    THRUSTER = auto()

    @staticmethod
    def spice_category_name():
        """Category of SPICE file."""
        return "thruster_category"


# The longest gap to ignore, in seconds, between consecutive spin or small forces
# files. They are delivered as pieces of one continuous record. Adjacent spin
# files overlap or gap by up to 20 ms, while a real gap is at least one spin
# (about 15 s).
SEAM_GAP_TOLERANCE = 1.0


@dataclass
class KernelCollection:
    """Collection of SPICE kernel types for IMAP."""

    imap_spice_load_order: list = field(
        default_factory=lambda: [
            LeapsecondKernels,
            PlanetaryConstantsKernels,
            IMAPFramesKernels,
            ScienceFramesKernels,
            SpacecraftClockKernels,
            EarthAttitudeKernels,
            PlanetaryEphemerisKernels,
            SpacecraftEphemerisKernels,
            SpacecraftAttitudeKernels,
            PointingAttitudeKernels,
            LoPivotAttitudeKernels,
            SpinTables,
            ThrusterFiles,
        ]
    )
    # Categories that are only loaded when their types are requested by name.
    # Spin and small forces files are not SPICE kernels, so they must not appear
    # in a default metakernel.
    opt_in_categories: list = field(default_factory=lambda: [SpinTables, ThrusterFiles])
    # Seconds of gap to ignore between files, for each category name
    gap_tolerances: dict = field(
        default_factory=lambda: {
            SpinTables.spice_category_name(): SEAM_GAP_TOLERANCE,
            ThrusterFiles.spice_category_name(): SEAM_GAP_TOLERANCE,
        }
    )

    @property
    def file_types(self):
        """Return all kernel members in lowercase."""
        members = []
        for kernel_class in self.imap_spice_load_order:
            members.extend([member.name.lower() for member in kernel_class])
        return members

    @property
    def category_types(self):
        """Collect all kernel category type strings."""
        return [
            kernel_class.spice_category_name()
            for kernel_class in self.imap_spice_load_order
        ]

    def requested_types(self, file_types: Collection[str] | None = None) -> dict:
        """Return the categories and types to load for a metakernel request.

        Parameters
        ----------
        file_types : Collection[str], optional
            Upper-case type names requested. If not given, every type except
            those in opt-in categories is loaded.

        Returns
        -------
        dict
            Each category to load, in load order, mapped to its types to load.
        """
        requested = {}
        for kernel_class in self.imap_spice_load_order:
            if file_types:
                members = [m for m in kernel_class if m.name in file_types]
            elif kernel_class in self.opt_in_categories:
                members = []
            else:
                members = list(kernel_class)
            if members:
                requested[kernel_class] = members
        return requested
