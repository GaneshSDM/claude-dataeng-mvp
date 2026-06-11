"""Tests for PII detector."""
import pytest
import pandas as pd
from core.pii.detector import scan_pii, PII_CATEGORIES


class TestPIIDetector:
    def test_detects_email(self):
        df = pd.DataFrame({"contact": ["alice@example.com", "bob@test.com", "noise"]})
        findings = scan_pii(df, "test")
        emails = [f for f in findings if f.category == "email"]
        assert len(emails) > 0
        assert emails[0].matches >= 2

    def test_detects_credit_card(self):
        df = pd.DataFrame({"cc_number": ["4111-1111-1111-1111", "5500-0000-0000-0004", "noise"]})
        findings = scan_pii(df, "test")
        cards = [f for f in findings if f.category == "credit_card"]
        assert len(cards) > 0
        assert cards[0].matches >= 2

    def test_detects_phone(self):
        df = pd.DataFrame({"phone": ["+1-555-123-4567", "555-123-4567", "noise"]})
        findings = scan_pii(df, "test")
        phones = [f for f in findings if f.category == "phone"]
        assert len(phones) > 0

    def test_detects_ip_address(self):
        df = pd.DataFrame({"ip_address": ["192.168.1.1", "10.0.0.255", "noise"]})
        findings = scan_pii(df, "test")
        ips = [f for f in findings if f.category == "ip_address"]
        assert len(ips) > 0

    def test_detects_ssn(self):
        df = pd.DataFrame({"ssn": ["123-45-6789", "987-65-4321", "noise"]})
        findings = scan_pii(df, "test")
        ssns = [f for f in findings if f.category == "ssn"]
        assert len(ssns) > 0

    def test_detects_url(self):
        df = pd.DataFrame({"website": ["https://example.com/page", "http://test.org", "noise"]})
        findings = scan_pii(df, "test")
        urls = [f for f in findings if f.category == "url"]
        assert len(urls) > 0

    def test_detects_zip_code(self):
        df = pd.DataFrame({"zip": ["90210", "12345-6789", "noise"]})
        findings = scan_pii(df, "test")
        zips = [f for f in findings if f.category == "zip_code"]
        assert len(zips) > 0

    def test_detects_date_of_birth(self):
        df = pd.DataFrame({"dob": ["1990-01-15", "1985-06-22", "noise"]})
        findings = scan_pii(df, "test")
        dobs = [f for f in findings if f.category == "date_of_birth"]
        assert len(dobs) > 0

    def test_clean_data_returns_empty(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": ["30", "25", "35"]})
        findings = scan_pii(df, "test")
        assert len(findings) == 0 or all(f.confidence == "low" for f in findings)

    def test_column_hints_focus_scan(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "email_addr": ["a@x.com", "b@y.com"]})
        findings = scan_pii(df, "test", column_hints=["email_addr"])
        assert len(findings) > 0
        assert findings[0].column == "email_addr"

    def test_all_categories_defined(self):
        assert len(PII_CATEGORIES) == 15
        expected = ["email", "phone", "credit_card", "ssn", "ip_address", "date_of_birth",
                     "passport", "driver_license", "bank_account", "health_insurance",
                     "street_address", "zip_code", "url", "api_key", "password_hash"]
        assert PII_CATEGORIES == expected