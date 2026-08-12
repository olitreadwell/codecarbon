"""
SCI (Software Carbon Intensity) output handler for CodeCarbon.

Writes a report in the shape defined by the Green Software Foundation's
Software Carbon Intensity specification, standardized as ISO/IEC 21031:2024::

    SCI = (E * I + M) / R

CodeCarbon measures ``E`` (energy consumed) and applies ``I`` (carbon
intensity). ``R``, the functional unit, and ``M``, the embodied emissions,
are declarations only the user can make, so they are never guessed: an
undeclared term is reported as undeclared.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from codecarbon.external.logger import logger
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData, TaskEmissionsData

SCI_SPEC_VERSION = "ISO/IEC 21031:2024"


@dataclass
class FunctionalUnit:
    """The ``R`` term: what one unit of the software is, and how many happened."""

    name: str
    count: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "FunctionalUnit":
        return cls(name=data.get("name", ""), count=data.get("count", 0.0))


@dataclass
class EmbodiedDeclaration:
    """The ``M`` term: user-declared embodied emissions, in gCO2eq."""

    gco2e: float = 0.0
    source: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "EmbodiedDeclaration":
        return cls(gco2e=data.get("gCO2e", 0.0), source=data.get("source", ""))


def map_emissions_to_sci(
    emissions: EmissionsData,
    functional_unit: Optional[FunctionalUnit] = None,
    embodied: Optional[EmbodiedDeclaration] = None,
    reporter: Optional[dict] = None,
    boundary: str = "",
) -> dict:
    """
    Build an SCI report dictionary from CodeCarbon emissions data.

    ``I`` is derived from the measured values as ``emissions * 1000 /
    energy_consumed`` rather than recomputed, so the report is consistent by
    construction with the CSV output, and already accounts for cloud region,
    PUE and the country/region fallbacks.

    Args:
        emissions: CodeCarbon emissions data from a completed run.
        functional_unit: The user-declared ``R`` term.
        embodied: The user-declared ``M`` term.
        reporter: Free-form reporter identification (organization, contact).
        boundary: Free-form description of the measurement boundary.

    Returns:
        A JSON-serializable dict. ``sci`` is ``None``, with a ``status``
        explaining why, when ``R`` was not declared.
    """
    energy_kwh = emissions.energy_consumed
    intensity = emissions.emissions * 1000 / energy_kwh if energy_kwh else 0.0
    embodied = embodied or EmbodiedDeclaration()
    unit_count = functional_unit.count if functional_unit else 0.0

    report = {
        "specVersion": SCI_SPEC_VERSION,
        "generatedBy": f"codecarbon {emissions.codecarbon_version}",
        "runId": str(emissions.run_id),
        "projectName": emissions.project_name,
        "timestamp": emissions.timestamp,
        "sci": None,
        "unit": None,
        "terms": {
            "E_kWh": energy_kwh,
            "I_gCO2e_per_kWh": intensity,
            "M_gCO2e": embodied.gco2e,
            "R": unit_count,
            "R_name": functional_unit.name if functional_unit else None,
        },
        "provenance": {
            "I_source": (
                "codecarbon regional intensity, "
                f"country_iso_code={emissions.country_iso_code}"
            ),
            "M_source": embodied.source or "not declared",
            "measurementBoundary": boundary
            or "Application only; excludes client devices and network.",
            "durationSeconds": emissions.duration,
            "hardware": {
                "cpu": emissions.cpu_model,
                "gpu": emissions.gpu_model,
                "ramTotalGB": emissions.ram_total_size,
            },
            "pue": getattr(emissions, "pue", 1),
        },
    }
    if reporter:
        report["reporter"] = reporter

    if unit_count > 0:
        report["sci"] = (energy_kwh * intensity + embodied.gco2e) / unit_count
        report["unit"] = f"gCO2eq per {functional_unit.name or 'functional unit'}"
    else:
        report["status"] = (
            "No functional unit (R) declared, so SCI could not be computed. "
            "Declare one via SCIOutput(functional_unit=...), "
            "set_functional_unit_count() or a context file."
        )
    return report


class SCIOutput(BaseOutput):
    """
    Output handler that writes SCI-formatted JSON reports.

    Usage:
        # Programmatic
        handler = SCIOutput(
            functional_unit=FunctionalUnit(name="inference request", count=10_000),
        )
        tracker = EmissionsTracker(output_handlers=[handler])

        # From context file
        handler = SCIOutput.from_file("sci_context.json")
        tracker = EmissionsTracker(output_handlers=[handler])
    """

    def __init__(
        self,
        output_dir: str = ".",
        functional_unit: Optional[FunctionalUnit] = None,
        embodied: Optional[EmbodiedDeclaration] = None,
        reporter: Optional[dict] = None,
        boundary: str = "",
    ):
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir
        self._functional_unit = functional_unit
        self._embodied = embodied
        self._reporter = reporter
        self._boundary = boundary

    @classmethod
    def from_file(cls, context_file_path: str, output_dir: str = ".") -> "SCIOutput":
        """
        Load the user-declared SCI terms from a JSON file.

        The context file holds what CodeCarbon cannot know::

            {
              "functionalUnit": {"name": "inference request", "count": 10000},
              "embodied": {"gCO2e": 42.5, "source": "manufacturer LCA, 4y"},
              "reporter": {"organization": "Acme"},
              "boundary": "Application only."
            }

        Args:
            context_file_path: Path to the SCI context JSON file.
            output_dir: Directory to write output reports to.

        Returns:
            A configured SCIOutput instance.

        Raises:
            FileNotFoundError: If the context file does not exist.
            json.JSONDecodeError: If the context file contains invalid JSON.
        """
        try:
            with open(context_file_path) as f:
                context = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"SCI context file not found: {context_file_path}")

        functional_unit = None
        embodied = None

        if "functionalUnit" in context:
            functional_unit = FunctionalUnit.from_dict(context["functionalUnit"])

        if "embodied" in context:
            embodied = EmbodiedDeclaration.from_dict(context["embodied"])

        return cls(
            output_dir=output_dir,
            functional_unit=functional_unit,
            embodied=embodied,
            reporter=context.get("reporter"),
            boundary=context.get("boundary", ""),
        )

    def set_functional_unit_count(self, count: float, name: Optional[str] = None):
        """Declare how many functional units the run covered, once it is known."""
        if self._functional_unit is None:
            self._functional_unit = FunctionalUnit(name=name or "", count=count)
        else:
            self._functional_unit.count = count
            if name is not None:
                self._functional_unit.name = name

    def _write(self, report: dict, file_name: str):
        file_path = os.path.join(self._output_dir, file_name)
        with open(file_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"SCI report saved to {os.path.abspath(file_path)}")

    def out(self, total: EmissionsData, delta: EmissionsData):
        """Write the final SCI report as a JSON file."""
        try:
            report = map_emissions_to_sci(
                total,
                functional_unit=self._functional_unit,
                embodied=self._embodied,
                reporter=self._reporter,
                boundary=self._boundary,
            )
            self._write(report, f"sci_report_{total.run_id}.json")
        except Exception as e:
            logger.error(f"Failed to write SCI report: {e}", exc_info=True)

    def task_out(self, data: List[TaskEmissionsData], experiment_name: str):
        """Write one SCI report per task, sharing the run-level declarations."""
        try:
            reports = []
            for task in data:
                report = map_emissions_to_sci(
                    task,
                    functional_unit=self._functional_unit,
                    embodied=self._embodied,
                    reporter=self._reporter,
                    boundary=self._boundary,
                )
                report["taskName"] = task.task_name
                reports.append(report)
            if not reports:
                return
            self._write(
                {
                    "specVersion": SCI_SPEC_VERSION,
                    "experimentName": experiment_name,
                    "tasks": reports,
                },
                f"sci_report_tasks_{data[0].run_id}.json",
            )
        except Exception as e:
            logger.error(f"Failed to write SCI task report: {e}", exc_info=True)

    def live_out(self, total: EmissionsData, delta: EmissionsData):
        """No-op: SCI reports are final, not incremental."""
