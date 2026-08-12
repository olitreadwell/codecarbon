from dataclasses import dataclass, field
from typing import Optional

from codecarbon.core.units import Energy, Power, Time
from codecarbon.external.logger import logger


@dataclass
class RAPLFile:
    # RAPL device being measured
    name: str
    # Path to file containing RAPL reading
    path: str
    # Path to corresponding file containing maximum possible RAPL reading
    max_path: str
    # Energy consumed in kWh
    energy_delta: Energy = field(default_factory=lambda: Energy(0))
    # Power based on reading
    power: Power = field(default_factory=lambda: Power(0))
    # Last energy reading in kWh, None if it could not be read
    last_energy: Optional[Energy] = field(default_factory=lambda: Energy(0))
    # Max value energy can hold before it wraps
    max_energy_reading: Energy = field(default_factory=lambda: Energy(0))

    def __post_init__(self):
        self._warned_uncorrectable_wrap = False
        self.last_energy = self._get_value()
        try:
            with open(self.max_path, "r") as f:
                max_micro_joules = float(f.read())
                self.max_energy_reading = Energy.from_ujoules(max_micro_joules)
        except Exception as e:
            # If we cannot read the max range, log and set to 0 so wrap detection
            # will be effectively disabled for this file.
            if isinstance(e, PermissionError):
                logger.warning(
                    "Unable to read max_energy_range_uj from %s due to permission error: %s",
                    self.max_path,
                    e,
                )
            else:
                logger.debug(
                    "Unable to read max_energy_range_uj from %s: %s",
                    self.max_path,
                    e,
                )
            self.max_energy_reading = Energy.from_ujoules(0)

    def _get_value(self) -> Optional[Energy]:
        """
        Reads the value in the file at the path, or None if it cannot be read.

        None (rather than a 0 reading) matters : a 0 would look like a
        wrap-around and inject a spurious delta of up to one full counter range.
        """
        try:
            with open(self.path, "r") as f:
                micro_joules = float(f.read())
                return Energy.from_ujoules(micro_joules)
        except Exception as e:
            # Be tolerant to transient IO / permission errors while reading energy.
            if isinstance(e, PermissionError):
                logger.warning(
                    "Unable to read RAPL value from %s due to permission error: %s",
                    self.path,
                    e,
                )
            else:
                logger.debug("Unable to read RAPL value from %s: %s", self.path, e)
            return None

    def start(self) -> None:
        self.last_energy = self._get_value()

    def _skip_sample(self, new_last_energy: Optional[Energy]) -> None:
        self.energy_delta = Energy(0)
        self.power = Power(0)
        self.last_energy = new_last_energy

    def delta(self, duration: Time) -> None:
        """
        Compute the energy used since last call.
        """
        new_last_energy = energy = self._get_value()
        if energy is None or self.last_energy is None:
            # No usable reading, now or at the previous sample : we have no
            # baseline to compute a delta from, so re-baseline silently.
            self._skip_sample(new_last_energy)
            return
        if self.last_energy > energy:
            logger.debug(
                f"In RAPLFile : Current energy value ({energy}) is lower than previous value ({self.last_energy}). Assuming wrap-around! Source file : {self.path}"
            )
            energy = energy + self.max_energy_reading
            if self.last_energy > energy:
                # The counter went backwards and we cannot correct it (unknown
                # max_energy_range_uj, driver reset, suspend/resume...). Skip
                # this sample rather than reporting a negative energy delta,
                # which would corrupt the totals.
                if not self._warned_uncorrectable_wrap:
                    self._warned_uncorrectable_wrap = True
                    logger.warning(
                        "In RAPLFile : counter went backwards and cannot be corrected for %s; skipping this sample (warned once).",
                        self.path,
                    )
                self._skip_sample(new_last_energy)
                return
        self.power = self.power.from_energies_and_delay(
            energy, self.last_energy, duration
        )
        self.energy_delta = energy - self.last_energy
        self.last_energy = new_last_energy
