"""PII detector — 15 categories, regex + entropy-based detection."""
from __future__ import annotations
import re
import math
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class PiiFinding:
    table: str
    column: str
    category: str
    matches: int
    sample_values: list[str]
    confidence: str        # "high" | "medium" | "low"
    severity: str          # "critical" | "high" | "medium" | "low"


PII_RULES: list[tuple[str, str, str, list[str]]] = [
    ("email", "high", r'[\w\.+-]+@[\w\.-]+\.\w+', ["email", "e-mail", "mail", "contact"]),
    ("phone", "high", r'\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b', ["phone", "mobile", "tel", "contact", "fax"]),
    ("credit_card", "critical", r'\b(?:\d{4}[-\s]?){3}\d{4}\b', ["card", "cc", "credit", "payment"]),
    ("ssn", "critical", r'\b\d{3}-\d{2}-\d{4}\b', ["ssn", "social", "security", "tax_id"]),
    ("ip_address", "medium", r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ["ip", "ip_address", "ipaddr"]),
    ("date_of_birth", "high", r'\b\d{4}-\d{2}-\d{2}\b', ["dob", "birth", "birth_date", "born"]),
    ("passport", "critical", r'\b[A-Z]{1,2}\d{6,9}\b', ["passport", "passport_no"]),
    ("driver_license", "high", r'\b[A-Z]\d{3,7}\b', ["license", "driver", "driving"]),
    ("bank_account", "critical", r'\b\d{8,17}\b', ["account", "bank", "iban", "aba"]),
    ("health_insurance", "high", r'\b[A-Z]{2,3}\d{6,12}\b', ["insurance", "health", "medicare", "medicaid"]),
    ("street_address", "medium", r'\b\d{1,5}\s+[A-Za-z]+\s+(Street|Ave|Road|Drive|Lane|Blvd|Court|Way)\b', ["address", "street", "location", "addr"]),
    ("zip_code", "medium", r'\b\d{5}(?:-\d{4})?\b', ["zip", "postal", "postcode", "zip_code"]),
    ("url", "low", r'https?://[^\s/$.?#].[^\s]*', ["url", "website", "link", "web"]),
    ("api_key", "critical", r'\b(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|AIza[0-9A-Za-z_-]{35})\b', ["api_key", "api", "secret", "token"]),
    ("password_hash", "critical", r'\b\$2[ayb]\$\d{2}\$[./A-Za-z0-9]{53}\b', ["password", "hash", "bcrypt", "passwd"]),
]

PII_CATEGORIES = [r[0] for r in PII_RULES]


def _entropy(s: str) -> float:
    """Shannon entropy — high entropy suggests encoded/token data."""
    if not s:
        return 0.0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)


def scan_pii(df: pd.DataFrame, table_name: str, column_hints: Optional[list[str]] = None) -> list[PiiFinding]:
    """Scan a DataFrame for PII across 15 categories.
    Args:
        df: DataFrame to scan (first 1000 rows sample recommended).
        table_name: Name of source table for reporting.
        column_hints: Optional column names to focus on.
    Returns:
        List of PiiFinding dataclass instances.
    """
    findings = []
    cols_to_scan = column_hints or [c for c in df.select_dtypes(include="object").columns]

    for col in cols_to_scan:
        if col not in df.columns:
            continue
        col_lower = col.lower()
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue

        for category, severity, pattern, keywords in PII_RULES:
            col_context_match = any(kw in col_lower for kw in keywords)
            sample = series.head(200) if col_context_match else series.head(50)
            matches = sample.str.findall(pattern)
            valid_matches = []
            for m in matches:
                for val in m:
                    cleaned = re.sub(r'[^a-zA-Z0-9]', '', val)
                    entropy = _entropy(cleaned) if cleaned else 0.0
                    if category in ("credit_card", "api_key", "password_hash") and entropy < 0.1:
                        continue
                    if category == "zip_code" and len(val) < 5:
                        continue
                    valid_matches.append(val)

            if valid_matches:
                confidence = "medium"
                if any(kw in col_lower for kw in keywords):
                    confidence = "high"
                findings.append(PiiFinding(
                    table=table_name,
                    column=col,
                    category=category,
                    matches=len(valid_matches),
                    sample_values=list(set(valid_matches))[:5],
                    confidence=confidence,
                    severity=severity,
                ))

    return findings