"""The Java and Python sides must agree, exactly.

MiniRedis computes features twice: once in Java, live,
to decide an eviction, and once in Python, offline, to
build the training set. If those two ever drift the
model is scored on vectors it was never trained on, and
nothing in either language will complain - the numbers
are all plausible, just wrong.

So the agreement is asserted here rather than assumed.
Two things are checked:

* the feature list is identical and in the same order,
  read straight out of ``MLFeatures.java``;
* the arithmetic matches, by running the same operations
  through both implementations and comparing vectors.

The arithmetic tests need a JDK. They skip rather than
fail when there is not one, so the suite stays useful on
a machine that only has Python.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from ml.config import EWMA_ALPHA, FEATURE_NAMES
from ml.features import KeyStats

SRC = Path(__file__).resolve().parent.parent

JAVA_FEATURES = SRC / "MLFeatures.java"
JAVA_KEY_STATS = SRC / "KeyStats.java"


# ---------------------------------------------------------------
# Locating a JDK
# ---------------------------------------------------------------


def find_jdk() -> tuple[str, str] | None:
    """Return (javac, java), or None when there is no JDK.

    ``java`` is often not on PATH on a machine where the
    only JDK came bundled with an editor extension, so
    the usual places are searched too.
    """
    javac = shutil.which("javac")
    java = shutil.which("java")

    if javac and java:
        return javac, java

    candidates: list[Path] = []

    java_home = os.environ.get("JAVA_HOME")

    if java_home:
        candidates.append(Path(java_home) / "bin")

    # JDKs shipped with the VS Code Java extension.
    extensions = Path.home() / ".vscode" / "extensions"

    if extensions.is_dir():
        for entry in extensions.glob("redhat.java-*/jre/*/bin"):
            candidates.append(entry)

    for folder in candidates:
        javac_path = folder / "javac.exe"
        java_path = folder / "java.exe"

        if not javac_path.exists():
            javac_path = folder / "javac"
            java_path = folder / "java"

        if javac_path.exists() and java_path.exists():
            return str(javac_path), str(java_path)

    return None


JDK = find_jdk()

needs_jdk = pytest.mark.skipif(
    JDK is None, reason="no JDK found, skipping cross language checks"
)


# ---------------------------------------------------------------
# The feature list
# ---------------------------------------------------------------


def java_feature_names() -> list[str]:
    """Read MLFeatures.NAMES out of the Java source.

    Parsing the source rather than running it keeps this
    check working without a JDK, which is the check most
    likely to catch a drift in the first place.
    """
    text = JAVA_FEATURES.read_text(encoding="utf-8")

    match = re.search(
        r"String\[\]\s+NAMES\s*=\s*\{(.*?)\}", text, re.DOTALL
    )

    assert match, "could not find NAMES in MLFeatures.java"

    return re.findall(r'"([^"]+)"', match.group(1))


def test_feature_names_match() -> None:
    assert java_feature_names() == list(FEATURE_NAMES)


def test_feature_names_are_unique() -> None:
    names = list(FEATURE_NAMES)

    assert len(names) == len(set(names))


def test_ewma_alpha_matches() -> None:
    """A different smoothing constant would shift one
    feature on one side only, which is exactly the kind
    of drift that is invisible until the model is wrong."""
    text = JAVA_KEY_STATS.read_text(encoding="utf-8")

    match = re.search(
        r"EWMA_ALPHA\s*=\s*([0-9.]+)", text
    )

    assert match, "could not find EWMA_ALPHA in KeyStats.java"

    assert float(match.group(1)) == pytest.approx(EWMA_ALPHA)


# ---------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------


# Operations are (operation, offset in ms, value size).
# The cases below cover the branches that differ: a key
# seen once, a key with a single interval, a key with
# many, and a key whose reads and writes are mixed.
SCENARIOS: dict[str, list[tuple[str, int, int]]] = {
    "single access": [
        ("SET", 0, 40),
    ],
    "one interval": [
        ("SET", 0, 40),
        ("GET", 1500, -1),
    ],
    "regular reads": [
        ("SET", 0, 128),
        ("GET", 1000, -1),
        ("GET", 2000, -1),
        ("GET", 3000, -1),
        ("GET", 4000, -1),
    ],
    "irregular reads": [
        ("SET", 0, 12),
        ("GET", 50, -1),
        ("GET", 9000, -1),
        ("GET", 9100, -1),
        ("GET", 30000, -1),
    ],
    "mixed operations": [
        ("SET", 0, 10),
        ("GET", 500, -1),
        ("SET", 1200, 250),
        ("GET", 1800, -1),
        ("EXISTS", 2400, -1),
        ("GET", 5000, -1),
    ],
    "rewritten": [
        ("SET", 0, 10),
        ("SET", 100, 20),
        ("SET", 200, 30),
    ],
    "zero gap": [
        ("SET", 0, 40),
        ("GET", 0, -1),
        ("GET", 0, -1),
    ],
}

# Where to stand when the vector is taken, relative to
# the first operation. The last one is well beyond the
# final access, which is the state an eviction decision
# is actually made in.
OBSERVATION_OFFSETS = (0, 5_000, 60_000)


JAVA_HARNESS = """
import java.util.ArrayList;
import java.util.List;

/** Prints feature vectors as JSON for the contract test. */
public class ContractHarness {

    public static void main(String[] args) {

        long base = 1_700_000_000_000L;

        StringBuilder out = new StringBuilder("{");

        boolean firstScenario = true;

        for (String spec : args) {

            String[] halves = spec.split("=", 2);

            String name = halves[0];

            if (!firstScenario) {
                out.append(",");
            }

            firstScenario = false;

            out.append("\\"").append(name).append("\\":[");

            List<long[]> observations = new ArrayList<>();

            KeyStats stats = null;

            String[] steps = halves[1].split(";");

            for (String step : steps) {

                String[] parts = step.split(",");

                String operation = parts[0];
                long offset = Long.parseLong(parts[1]);
                int size = Integer.parseInt(parts[2]);

                if (stats == null) {
                    stats = new KeyStats(base + offset);
                }

                stats.record(operation, base + offset, size);
            }

            long last =
                    Long.parseLong(
                            steps[steps.length - 1].split(",")[1]);

            long[] offsets = {0L, 5000L, 60000L};

            boolean firstVector = true;

            for (long extra : offsets) {

                if (!firstVector) {
                    out.append(",");
                }

                firstVector = false;

                double[] vector =
                        stats.toFeatureVector(base + last + extra);

                out.append("[");

                for (int i = 0; i < vector.length; i++) {

                    if (i > 0) {
                        out.append(",");
                    }

                    out.append(vector[i]);
                }

                out.append("]");
            }

            out.append("]");
        }

        out.append("}");

        System.out.println(out);
    }
}
"""


def python_vectors(
    steps: list[tuple[str, int, int]]
) -> list[list[float]]:
    base = 1_700_000_000_000

    stats: KeyStats | None = None

    for operation, offset, size in steps:
        if stats is None:
            stats = KeyStats(base + offset)

        stats.record(operation, base + offset, size)

    assert stats is not None

    last = steps[-1][1]

    return [
        stats.to_feature_vector(base + last + extra)
        for extra in OBSERVATION_OFFSETS
    ]


@pytest.fixture(scope="module")
def java_vectors() -> dict[str, list[list[float]]]:
    """Compile and run the harness once for all cases."""
    assert JDK is not None

    javac, java = JDK

    with tempfile.TemporaryDirectory() as work:
        harness = Path(work) / "ContractHarness.java"
        harness.write_text(JAVA_HARNESS, encoding="utf-8")

        compile_result = subprocess.run(
            [
                javac,
                "-d",
                work,
                str(harness),
                str(SRC / "KeyStats.java"),
                str(SRC / "MLFeatures.java"),
            ],
            capture_output=True,
            text=True,
        )

        assert compile_result.returncode == 0, compile_result.stderr

        specs = [
            name
            + "="
            + ";".join(
                f"{operation},{offset},{size}"
                for operation, offset, size in steps
            )
            for name, steps in SCENARIOS.items()
        ]

        run_result = subprocess.run(
            [java, "-cp", work, "ContractHarness", *specs],
            capture_output=True,
            text=True,
        )

        assert run_result.returncode == 0, run_result.stderr

        return json.loads(run_result.stdout)


@needs_jdk
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_java_and_python_agree(
    scenario: str, java_vectors: dict[str, list[list[float]]]
) -> None:
    expected = python_vectors(SCENARIOS[scenario])
    actual = java_vectors[scenario]

    assert len(actual) == len(expected)

    for observation, (java_row, python_row) in enumerate(
        zip(actual, expected, strict=True)
    ):
        assert len(java_row) == len(FEATURE_NAMES)

        for name, java_value, python_value in zip(
            FEATURE_NAMES, java_row, python_row, strict=True
        ):
            assert java_value == pytest.approx(
                python_value, rel=1e-12, abs=1e-12
            ), (
                f"{scenario}, observation {observation}, "
                f"feature {name}: java {java_value} "
                f"!= python {python_value}"
            )


@needs_jdk
def test_vectors_are_finite(
    java_vectors: dict[str, list[list[float]]]
) -> None:
    """A NaN reaches the service as null and is rejected,
    so a feature that can divide by zero has to be caught
    here rather than at eviction time."""
    import math

    for scenario, rows in java_vectors.items():
        for row in rows:
            for name, value in zip(FEATURE_NAMES, row, strict=True):
                assert math.isfinite(value), (
                    f"{scenario}: {name} is {value}"
                )
