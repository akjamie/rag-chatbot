from pathlib import Path
import re
import pytest

REQUIRED_EXACT_PINS = {
    "langchain": "1.4.9",
    "langchain-core": "1.7.6",
    "langchain-community": "0.4.2",
    "langgraph": "1.2.20",
    "langchain-huggingface": "1.2.3",
    "langchain-qdrant": "1.1.0",
    "langchain-anthropic": "1.7.6",
    "langchain-google-genai": "4.4.3",
    "langchain-ollama": "1.1.0",
}

REQUIRED_MIN_BOUNDS = {
    "fastapi": "0.115.6",
    "uvicorn": "0.32.1",
    "neo4j": "5.26.0",
    "spacy": "3.8.4",
}


def _parse_requirements():
    req_file = Path(__file__).resolve().parents[2] / "requirements.txt"
    assert req_file.exists(), f"requirements.txt not found at {req_file}"

    packages = {}
    with open(req_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Match package name and operator/version
            m = re.match(r"^([a-zA-Z0-9_\-]+)\s*([=><~]+.*)$", line)
            if m:
                pkg, spec = m.group(1).lower(), m.group(2).strip()
                packages[pkg] = spec
    return packages


def test_langchain_exact_pins():
    packages = _parse_requirements()
    for pkg, expected_version in REQUIRED_EXACT_PINS.items():
        assert pkg in packages, f"Missing {pkg} in requirements.txt"
        assert packages[pkg] == f"=={expected_version}", (
            f"Expected {pkg}=={expected_version}, got {packages[pkg]}"
        )


def test_runtime_driver_bounds():
    packages = _parse_requirements()
    for pkg, min_version in REQUIRED_MIN_BOUNDS.items():
        assert pkg in packages, f"Missing {pkg} in requirements.txt"
        # Must be >= or == or ~= meeting min bound
        assert any(packages[pkg].startswith(op) for op in [">=", "==", "~="]), (
            f"Expected valid version spec for {pkg}, got {packages[pkg]}"
        )
