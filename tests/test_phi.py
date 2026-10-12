import pytest

from guardrail_proxy.phi.recognizers import dea_valid, luhn_valid, npi_valid
from guardrail_proxy.vault import PseudonymVault


def test_luhn():
    assert luhn_valid("79927398713")
    assert not luhn_valid("79927398710")


@pytest.mark.parametrize(
    ("npi", "ok"),
    [("1245319599", True), ("1245319598", False), ("3245319599", False), ("124531959", False)],
)
def test_npi_checksum(npi, ok):
    assert npi_valid(npi) is ok


@pytest.mark.parametrize(
    ("dea", "ok"),
    [("AB1234563", True), ("AB1234564", False), ("ab1234563", False), ("A11234563", False)],
)
def test_dea_checksum(dea, ok):
    assert dea_valid(dea) is ok


def entities(detector, text):
    return {(s.entity, s.text) for s in detector.detect(text)}


def test_clinical_identifiers_are_detected(phi_detector):
    text = (
        "Patient Maria Garcia, DOB: 07/22/1984, MRN: 00482913, NPI 1245319599, "
        "DEA AB1234563, member ID: XKP-88213409, SSN 123-45-6789, lives at 42 Maple Street."
    )
    found = entities(phi_detector, text)
    assert ("PERSON", "Maria Garcia") in found
    assert ("DATE_OF_BIRTH", "07/22/1984") in found
    assert ("MRN", "00482913") in found  # the span is the number, not "MRN: 00482913"
    assert ("NPI", "1245319599") in found
    assert ("DEA_NUMBER", "AB1234563") in found
    assert ("MEMBER_ID", "XKP-88213409") in found
    assert ("US_SSN", "123-45-6789") in found
    assert ("STREET_ADDRESS", "42 Maple Street") in found


def test_dosing_language_and_drug_names_survive(phi_detector):
    text = (
        "Take metformin 500 mg twice daily for 3 days, then increase. Continue Lipitor "
        "and Warfarin; recheck INR in 2 weeks."
    )
    assert phi_detector.detect(text) == []


def test_placeholders_are_not_redetected(phi_detector):
    assert phi_detector.detect("Please follow up with <PERSON_1> about <MRN_1>.") == []


def test_pseudonymize_roundtrip(phi_detector):
    vault = PseudonymVault()
    text = "John Smith (MRN 00482913) asked about John Smith's refill."
    out, _ = phi_detector.pseudonymize(text, vault)
    assert "John Smith" not in out and "00482913" not in out
    assert out.count("<PERSON_1>") == 2  # same value -> same token
    assert vault.rehydrate(out) == text


def test_vault_rehydrates_mangled_tokens():
    vault = PseudonymVault()
    token = vault.tokenize("PERSON", "Jane Doe")
    assert token == "<PERSON_1>"
    for variant in ["<PERSON_1>", "[PERSON_1]", "PERSON_1", "<person_1>"]:
        assert vault.rehydrate(f"Hello {variant}.") == "Hello Jane Doe."
    assert vault.rehydrate("Unknown <PERSON_9> stays.") == "Unknown <PERSON_9> stays."
