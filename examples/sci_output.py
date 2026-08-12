"""
Write a Software Carbon Intensity (ISO/IEC 21031) report.

The functional unit `R` is a declaration: only you know what one unit of your
software is. Here it is one "inference request", counted as the workload runs.
"""

from codecarbon import EmissionsTracker
from codecarbon.output_methods.sci import FunctionalUnit, SCIOutput

REQUESTS = 100


def handle_request(number):
    a = 0
    for i in range(int(1e6)):
        a = a + i**number
    return a


sci = SCIOutput(
    functional_unit=FunctionalUnit(name="inference request"),
    # embodied=EmbodiedDeclaration(gCO2e=42.5, source="vendor LCA, 4y amortization"),
)
tracker = EmissionsTracker(
    measure_power_secs=10,
    output_methods=["csv"],
    output_handlers=[sci],
)
try:
    tracker.start()
    for i in range(REQUESTS):
        handle_request(i % 3)
finally:
    sci.set_functional_unit_count(REQUESTS)
    emissions = tracker.stop()

print(f"Emissions: {emissions} kg")
print(f"SCI report written to ./sci_report_{tracker.run_id}.json")
