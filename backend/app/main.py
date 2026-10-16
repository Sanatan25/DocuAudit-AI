
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.document_processor import process_pdf
from app.ml.financial_anomaly import detect_financial_anomalies
from app.ml.document_ai import analyze_business_document_with_ollama


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

app = FastAPI(
    title="DocuAudit AI API",
    version="0.3.0",
    description="PDF document auditing and analysis API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_FILES = 3

SERVICES = {"document", "financial", "invoice"}


# ==================================================
# 2. HEALTH CHECK
# ==================================================

@app.get("/api/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "DocuAudit AI",
    }


# ==================================================
# 3. PDF EXTRACTION
# ==================================================

async def extract_pdf(upload: UploadFile) -> dict[str, Any]:
    filename = upload.filename or ""

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    content = await upload.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail=f"{filename} is empty.",
        )

    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{filename} exceeds the 15 MB limit.",
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail=f"{filename} does not appear to be a valid PDF.",
        )

    try:
        return process_pdf(filename, content)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {filename} as a PDF.",
        ) from exc


# ==================================================
# 4. FINDING AND EVIDENCE HELPERS
# ==================================================

def finding(
    title: str,
    severity: str,
    description: str,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "title": title,
        "severity": severity,
        "description": description,
        "evidence": evidence,
    }


def get_page_evidence(
    doc: dict[str, Any],
    page_number: int,
) -> dict[str, Any]:
    page_text = ""

    for page in doc.get("pages", []):
        if page.get("page") == page_number:
            page_text = page.get("text", "")
            break

    return {
        "filename": doc.get("filename", "document.pdf"),
        "page": page_number,
        "text": page_text[:1500],
    }


def all_page_evidence(
    docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence = []

    for doc in docs:
        pages = doc.get("pages", [])

        if pages:
            evidence.append(
                get_page_evidence(doc, pages[0]["page"])
            )

    return evidence


# ==================================================
# 5. AI DOCUMENT AUDITOR
# ==================================================

def document_audit(
    docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Audit contracts and business documents using explainable text rules.
    Existing extraction, evidence helpers, API routes, financial analysis,
    invoice matching, and frontend response structures remain unchanged.
    """
    findings: list[dict[str, Any]] = []

    party_pattern = re.compile(
        r"\b(prepared\s+by|prepared\s+for|author|client|"
        r"company|organization|organisation|parties|vendor|"
        r"supplier|customer|submitted\s+by|contact\s+person)\b",
        re.IGNORECASE,
    )

    report_pattern = re.compile(
        r"\b(report|assessment|study|analysis|proposal)\b",
        re.IGNORECASE,
    )

    section_groups = {
        "purpose or objective": (
            r"\bpurpose\b", r"\bobjectives?\b",
            r"\baims?\b", r"\boverview\b",
        ),
        "methodology or approach": (
            r"\bmethodology\b", r"\bmethods?\b",
            r"\bapproach\b", r"\bimplementation\b",
            r"\bworkflow\b", r"\barchitecture\b",
        ),
        "results or findings": (
            r"\bresults?\b", r"\bfindings?\b",
            r"\boutcomes?\b", r"\bevaluation\b",
            r"\bperformance\b",
        ),
        "conclusion or recommendations": (
            r"\bconclusion\b", r"\brecommendations?\b",
            r"\bsummary\b", r"\bfuture work\b",
            r"\bnext steps\b",
        ),
        "references or sources": (
            r"\breferences\b", r"\bbibliography\b",
            r"\bsources\b", r"\bcitations\b",
        ),
    }

    placeholder_pattern = re.compile(
        r"\b(?:TODO|TBD|PLACEHOLDER|LOREM IPSUM)\b"
        r"|\bINSERT\s+(?:TEXT|HERE|NAME|DATE)\b"
        r"|\bXXX{2,}\b",
        re.IGNORECASE,
    )

    # These checks are deliberately phrased as review signals, not legal conclusions.
    contract_type_pattern = re.compile(
        r"\b(contract|agreement|terms\s+and\s+conditions|"
        r"service\s+agreement|vendor\s+agreement|"
        r"purchase\s+agreement|statement\s+of\s+work|"
        r"master\s+services)\b",
        re.IGNORECASE,
    )

    business_type_pattern = re.compile(
        r"\b(invoice|purchase\s+order|quotation|quote|"
        r"vendor|supplier|customer|service\s+fee|payment\s+terms|"
        r"delivery\s+terms|statement\s+of\s+work)\b",
        re.IGNORECASE,
    )

    risky_clause_rules = [
        (
            "Potentially excessive late-payment charge",
            re.compile(
                r"\b(?:\d+(?:\.\d+)?\s*%\s*(?:per\s+day|daily)|"
                r"\d+(?:\.\d+)?\s*%\s*per\s+day)\b",
                re.IGNORECASE,
            ),
            "A daily percentage-based late charge appears in the text. "
            "Verify the rate, calculation method, applicable law, and commercial terms.",
        ),
        (
            "Broad liability exclusion",
            re.compile(
                r"(?:not\s+responsible\s+for\s+any|"
                r"(?:exclude|excludes|excluding|excluded)\s+all\s+liability|"
                r"under\s+any\s+circumstances)",
                re.IGNORECASE,
            ),
            "The wording may broadly exclude responsibility or liability. "
            "Review its scope, exceptions, and interaction with other clauses.",
        ),
        (
            "Unrestricted unilateral price changes",
            re.compile(
                r"(?:change|increase|adjust|revise)\s+(?:the\s+)?prices?"
                r".{0,100}(?:at\s+any\s+time|without\s+(?:prior\s+)?notice|"
                r"without\s+(?:the\s+)?(?:customer|client)\s+consent)",
                re.IGNORECASE,
            ),
            "The text may permit price changes without adequate notice or approval. "
            "Verify the change process and customer protections.",
        ),
        (
            "Unrestricted subcontracting",
            re.compile(
                r"(?:may|can|shall\s+be\s+entitled\s+to)\s+subcontract"
                r".{0,120}(?:without\s+(?:prior\s+)?notice|without\s+consent|"
                r"without\s+approval|any\s+portion)",
                re.IGNORECASE,
            ),
            "The text may permit subcontracting without notice or consent. "
            "Review responsibility, confidentiality, security, and approval requirements.",
        ),
        (
            "Use of customer data without explicit consent",
            re.compile(
                r"(?:use|process|share)\s+(?:production\s+)?(?:customer|client)\s+data"
                r".{0,140}(?:without\s+(?:prior\s+)?(?:written\s+)?consent|"
                r"without\s+permission|for\s+(?:testing|demonstrations))",
                re.IGNORECASE,
            ),
            "The clause may allow customer data to be used beyond the service purpose. "
            "Verify consent, permitted purposes, safeguards, and applicable obligations.",
        ),
        (
            "One-sided immediate termination right",
            re.compile(
                r"(?:vendor|supplier|provider)\s+may\s+terminate"
                r".{0,180}(?:immediately|without\s+notice|for\s+any\s+reason)",
                re.IGNORECASE | re.DOTALL,
            ),
            "The vendor appears to have a broad immediate termination right. "
            "Compare it with the customer's termination rights and required notice.",
        ),
        (
            "Indefinite customer-data retention or broad reuse",
            re.compile(
                r"(?:retain|store|keep)\s+(?:customer\s+)?data\s+(?:indefinitely|forever|without\s+time\s+limit)|"
                r"use\s+(?:customer\s+)?data\s+for\s+any\s+business\s+purpose",
                re.IGNORECASE,
            ),
            "The wording appears to allow indefinite retention or broad reuse of customer data. "
            "Review retention limits, deletion obligations, permitted purposes, and safeguards.",
        ),
        (
            "Security controls or breach-notification terms may be inadequate",
            re.compile(
                r"(?:no\s+commitment\s+to\s+encryption|no\s+defined\s+security\s+incident\s+notification|"
                r"at\s+its\s+sole\s+discretion.{0,100}(?:inform|notify).{0,80}(?:breach|incident))",
                re.IGNORECASE | re.DOTALL,
            ),
            "The agreement appears to leave important security controls or breach notification undefined. "
            "Review encryption, access controls, response timelines, and escalation requirements.",
        ),
        (
            "Service-level remedies are not clearly defined",
            re.compile(
                r"(?:uptime|availability)\s*(?:target|of|is|at)?\s*\d{2,3}(?:\.\d+)?\s*%",
                re.IGNORECASE,
            ),
            "An uptime target is stated, but verify that measurement rules, exclusions, incident response targets, "
            "and service credits or other remedies are also defined.",
        ),
    ]

    # Terms whose absence may be worth reviewing in a contract.
    contract_presence_checks = [
        (
            "Effective date",
            re.compile(r"\b(effective\s+date|commencement\s+date|effective\s+as\s+of)\b", re.I),
        ),
        (
            "Payment terms",
            re.compile(r"\b(payment\s+terms|invoice\s+due|payment\s+due|shall\s+pay|service\s+fee|price|consideration)\b", re.I),
        ),
        (
            "Termination terms",
            re.compile(r"\b(terminat(?:e|ion|ed)|expiry|expiration)\b", re.I),
        ),
        (
            "Confidentiality terms",
            re.compile(r"\b(confidential(?:ity)?|non-disclosure|proprietary\s+information)\b", re.I),
        ),
        (
            "Data protection or security terms",
            re.compile(r"\b(data\s+protection|data\s+security|encryption|security\s+incident|breach\s+notification|personal\s+data)\b", re.I),
        ),
        (
            "Governing law or dispute resolution",
            re.compile(r"\b(governing\s+law|jurisdiction|dispute\s+resolution|arbitration|mediation)\b", re.I),
        ),
        (
            "Signatures or execution",
            re.compile(r"\b(signature|signed\s+by|authori[sz]ed\s+representative|in\s+witness\s+whereof|execution)\b", re.I),
        ),
    ]

    for doc in docs:
        text = doc.get("text", "")
        filename = doc.get("filename", "document.pdf")
        pages = doc.get("pages", [])

        if not text.strip():
            findings.append(
                finding(
                    "Text extraction required",
                    "review",
                    f"No selectable text was extracted from {filename}. OCR may be required.",
                    all_page_evidence([doc]),
                )
            )
            continue

        scanned_pages = doc.get("likely_scanned_pages", [])
        if scanned_pages:
            findings.append(
                finding(
                    "Pages may require OCR",
                    "review",
                    f"{filename} may contain scanned pages: {scanned_pages}. Verify their contents.",
                    [get_page_evidence(doc, number) for number in scanned_pages],
                )
            )

        if not party_pattern.search(text):
            first_page = pages[0]["page"] if pages else 1
            findings.append(
                finding(
                    "Review document parties or organization names",
                    "review",
                    f"Common author, client, or organization indicators were not found in {filename}. "
                    "This does not prove that the details are missing.",
                    [get_page_evidence(doc, first_page)],
                )
            )

        for page in pages:
            page_text = page.get("text", "")
            match = placeholder_pattern.search(page_text)
            if match:
                findings.append(
                    finding(
                        "Possible placeholder or unfinished text",
                        "review",
                        f"{filename}, page {page.get('page')}, contains possible placeholder text: {match.group(0)!r}.",
                        [get_page_evidence(doc, page.get("page", 1))],
                    )
                )

        # Keep the existing report checks for documents that look like reports,
        # but do not apply report-section requirements to contracts/business docs.
        is_contract = bool(contract_type_pattern.search(text))
        is_business_doc = bool(business_type_pattern.search(text))

        if report_pattern.search(text) and not is_contract and not is_business_doc:
            missing = []
            for section, patterns in section_groups.items():
                if not any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                    missing.append(section)

            if missing:
                first_page = pages[0]["page"] if pages else 1
                findings.append(
                    finding(
                        "Potentially missing common report sections",
                        "review",
                        f"{filename}: these sections were not clearly identified: {', '.join(missing)}. "
                        "Verify against the report's requirements.",
                        [get_page_evidence(doc, first_page)],
                    )
                )

        if len(text.strip()) < 100:
            first_page = pages[0]["page"] if pages else 1
            findings.append(
                finding(
                    "Limited text extracted",
                    "review",
                    f"Only {len(text.strip())} characters were extracted from {filename}. Verify extraction quality.",
                    [get_page_evidence(doc, first_page)],
                )
            )

        if is_contract or is_business_doc:
            # Use the page containing each matching clause as the evidence source.
            for title, pattern, explanation in risky_clause_rules:
                match = pattern.search(text)
                if match:
                    match_page = 1
                    match_text = match.group(0)
                    for page in pages:
                        page_text = page.get("text", "")
                        page_match = pattern.search(page_text)
                        if page_match:
                            match_page = page.get("page", 1)
                            match_text = page_match.group(0)
                            break

                    findings.append(
                        finding(
                            title,
                            "review",
                            f"{filename}: {explanation} Matched wording: “{match_text[:240]}”. "
                            "This is a review signal, not a legal conclusion.",
                            [get_page_evidence(doc, match_page)],
                        )
                    )

        if is_contract:
            # Detect contradictory commercial amounts without assuming which one is correct.
            amount_matches = list(re.finditer(
                r"\b(?:INR|Rs\.?|₹)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
                text, re.IGNORECASE,
            ))
            amount_values = []
            for amount_match in amount_matches:
                raw_amount = amount_match.group(1).replace(",", "")
                try:
                    amount_value = float(raw_amount)
                except ValueError:
                    continue
                if amount_value not in amount_values:
                    amount_values.append(amount_value)
            if len(amount_values) >= 3 and re.search(r"\b(monthly\s+fee|monthly\s+charge|monthly\s+price)\b", text, re.I):
                first_page = pages[0]["page"] if pages else 1
                findings.append(finding(
                    "Conflicting commercial amounts require reconciliation",
                    "review",
                    f"{filename}: multiple distinct currency amounts appear in the agreement ({', '.join(f'INR {v:,.2f}' for v in amount_values[:6])}). "
                    "Confirm which amount governs; do not assume that any one value is correct.",
                    [get_page_evidence(doc, first_page)],
                ))

            # Flag dates only when explicit start/commencement and expiry dates can be parsed.
            start_match = re.search(r"(?:commencement|start|effective)\s+date\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})", text, re.I)
            end_match = re.search(r"(?:expiry|expiration|end)\s+date\s*[:\-]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})", text, re.I)
            if start_match and end_match:
                from datetime import datetime
                def parse_contract_date(value: str):
                    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%B %d, %Y", "%Y-%m-%d"):
                        try:
                            return datetime.strptime(value.replace(",", ""), fmt).date()
                        except ValueError:
                            pass
                    return None
                start_date = parse_contract_date(start_match.group(1))
                end_date = parse_contract_date(end_match.group(1))
                if start_date and end_date and end_date < start_date:
                    first_page = pages[0]["page"] if pages else 1
                    findings.append(finding(
                        "Contract dates appear inconsistent",
                        "review",
                        f"{filename}: expiry date {end_match.group(1)} appears earlier than commencement date {start_match.group(1)}. Verify the intended dates.",
                        [get_page_evidence(doc, first_page)],
                    ))

            missing_terms = [
                label for label, pattern in contract_presence_checks
                if not pattern.search(text)
            ]
            if missing_terms:
                first_page = pages[0]["page"] if pages else 1
                findings.append(
                    finding(
                        "Potentially missing contract terms",
                        "review",
                        f"{filename}: these common contract topics were not clearly identified: "
                        f"{', '.join(missing_terms)}. Confirm whether they are required for this agreement "
                        "and whether equivalent wording appears elsewhere.",
                        [get_page_evidence(doc, first_page)],
                    )
                )

        # A business document with no obvious date or reference can be reviewed,
        # but absence is not treated as proof of an error.
        if is_business_doc:
            if not re.search(r"\b(?:date|dated|effective\s+date|invoice\s+date|order\s+date)\b", text, re.I):
                first_page = pages[0]["page"] if pages else 1
                findings.append(
                    finding(
                        "Review document date",
                        "review",
                        f"{filename}: a clear document or transaction date was not identified. Verify manually.",
                        [get_page_evidence(doc, first_page)],
                    )
                )

    # Optional open-source LLM analysis via local Ollama.
    # Rule-based findings above remain active even when Ollama is unavailable.
    for doc in docs:
        doc_text = doc.get("text", "")
        filename = doc.get("filename", "document.pdf")
        if not doc_text.strip():
            continue
        try:
            ai_findings = analyze_business_document_with_ollama(
                text=doc_text,
                filename=filename,
                pages=doc.get("pages", []),
            )
            findings.extend(ai_findings)
        except Exception:
            # Keep the existing rule-based audit available if local AI fails.
            # Ollama diagnostics are intentionally not allowed to break the API.
            continue

    return findings


# ==================================================
# 6. FINANCIAL REPORT DETECTIVE
# ==================================================

def financial_audit(
    docs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    ml_diagnostics: list[dict[str, Any]] = []

    negative_amount_pattern = re.compile(
        r"\b(?:total|revenue|sales|profit|amount|balance)\b"
        r"[^\n]{0,35}?"
        r"(?:₹|INR|USD|\$)?\s*-\s*\d[\d,]*"
        r"(?:\.\d{1,2})?",
        re.IGNORECASE,
    )

    financial_terms = re.compile(
        r"\b(revenue|sales|profit|expense|balance|financial|"
        r"assets|liabilities|equity|cash flow|receivables|payables)\b",
        re.IGNORECASE,
    )

    for doc in docs:
        text = doc.get("text", "")
        filename = doc.get("filename", "document.pdf")
        pages = doc.get("pages", [])

        if not text.strip():
            findings.append(
                finding(
                    "Financial analysis requires readable text",
                    "review",
                    f"No readable text was extracted from {filename}. "
                    "OCR or another source PDF may be required.",
                    all_page_evidence([doc]),
                )
            )

            ml_diagnostics.append({
                "filename": filename,
                "status": "no_extractable_text",
                "candidate_count": 0,
                "categories_tested": 0,
                "anomaly_count": 0,
                "message": "ML analysis was skipped.",
            })
            continue

        # Run ML against the actual text extracted from this PDF.
        try:
            ml_result = detect_financial_anomalies(
                text=text,
                filename=filename,
            )

            ml_diagnostics.append({
                "filename": filename,
                "status": ml_result.get("status", "unknown"),
                "candidate_count": ml_result.get("candidate_count", 0),
                "category_counts": ml_result.get("category_counts", {}),
                "categories_tested": ml_result.get("categories_tested", 0),
                "anomaly_count": ml_result.get("anomaly_count", 0),
                "message": ml_result.get("message", ""),
            })

            for anomaly in ml_result.get("findings", []):
                evidence = {
                    "filename": filename,
                    "page": None,
                    "text": anomaly.get("evidence", ""),
                }

                amount = anomaly.get("amount", 0)
                line_number = anomaly.get("line_number", "?")
                category = anomaly.get("category", "uncategorized")
                score = anomaly.get("anomaly_score", 0)

                findings.append(
                    finding(
                        "ML detected unusual financial amount",
                        "review",
                        (
                            f"{filename}, extracted line {line_number}, "
                            f"contains {amount:,.2f} in the inferred "
                            f"{category.replace('_', ' ')} category. "
                            f"{anomaly.get('message', '')} "
                            f"Anomaly score: {score}. This is a review "
                            "signal, not proof of an error or fraud."
                        ),
                        [evidence],
                    )
                )

        except Exception as exc:
            # Report an explicit diagnostic rather than silently
            # returning an apparently successful empty ML result.
            ml_diagnostics.append({
                "filename": filename,
                "status": "error",
                "candidate_count": 0,
                "categories_tested": 0,
                "anomaly_count": 0,
                "message": f"ML analysis failed: {type(exc).__name__}",
            })

            findings.append(
                finding(
                    "Financial ML analysis could not complete",
                    "review",
                    f"ML analysis failed for {filename}. Check backend "
                    "logs for the underlying error.",
                    all_page_evidence([doc]),
                )
            )

        for page in pages:
            page_text = page.get("text", "")

            if negative_amount_pattern.search(page_text):
                findings.append(
                    finding(
                        "Review negative financial amount",
                        "review",
                        f"{filename}, page {page.get('page')}, contains "
                        "a potentially negative labelled amount. Negative "
                        "values can be legitimate; verify the context.",
                        [
                            get_page_evidence(
                                doc, page.get("page", 1)
                            )
                        ],
                    )
                )

        if not financial_terms.search(text):
            first_page = pages[0]["page"] if pages else 1
            findings.append(
                finding(
                    "Financial content not clearly identified",
                    "review",
                    f"Common financial terms were not found in {filename}. "
                    "Confirm that this is the intended financial report.",
                    [get_page_evidence(doc, first_page)],
                )
            )

    return findings, ml_diagnostics


# ==================================================
# 7. INVOICE AND PURCHASE ORDER MATCHER
# ==================================================

def invoice_audit(
    docs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare invoice and PO PDFs using deterministic checks plus optional Ollama assistance.

    Numerical values are compared in Python. Ollama is used only to suggest
    semantic matches for line-item descriptions that do not match literally.
    Findings are review signals and do not automatically approve or reject payment.
    """
    if len(docs) < 2:
        return [finding(
            "Two documents required", "review",
            "Upload at least an invoice and a purchase order to perform a comparison.", [],
        )]

    unreadable = [doc for doc in docs if not doc.get("text", "").strip()]
    if unreadable:
        return [finding(
            "Document requires text extraction", "review",
            f"{doc.get('filename', 'Document')} has no selectable text. OCR is required before reliable matching.",
            all_page_evidence([doc]),
        ) for doc in unreadable]

    findings: list[dict[str, Any]] = []

    def first_page(doc: dict[str, Any]) -> int:
        pages = doc.get("pages", [])
        return pages[0].get("page", 1) if pages else 1

    def evidence_for(doc: dict[str, Any], text_hint: str | None = None) -> list[dict[str, Any]]:
        pages = doc.get("pages", [])
        if text_hint:
            for page in pages:
                if text_hint.lower() in page.get("text", "").lower():
                    return [get_page_evidence(doc, page.get("page", 1))]
        return [get_page_evidence(doc, first_page(doc))]

    # Infer document roles from their content, not upload order.
    invoice_docs = [d for d in docs if re.search(r"\binvoice\b|\btax invoice\b", d.get("text", ""), re.I)]
    po_docs = [d for d in docs if re.search(r"\bpurchase\s+order\b|\bP\.?O\.?\s*(?:number|no\.?|#)?", d.get("text", ""), re.I)]
    invoice_doc = invoice_docs[0] if invoice_docs else docs[0]
    po_doc = next((d for d in po_docs if d is not invoice_doc), docs[1] if docs[1] is not invoice_doc else docs[0])

    def extract_reference(text: str, labels: str) -> str | None:
        m = re.search(rf"\b(?:{labels})\b\s*(?:number|no\.?|#|ID)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9/_.\-]{{2,}})", text, re.I)
        return m.group(1).strip(" .,:;") if m else None

    invoice_no = extract_reference(invoice_doc.get("text", ""), r"invoice(?:\s+no)?")
    po_no = extract_reference(po_doc.get("text", ""), r"purchase\s+order|P\.?O\.?|PO")
    if not invoice_no:
        findings.append(finding(
            "Invoice number not identified", "review",
            f"Could not confidently extract an invoice number from {invoice_doc.get('filename', 'invoice PDF')}. Check the document manually.",
            evidence_for(invoice_doc),
        ))
    if not po_no:
        findings.append(finding(
            "Purchase order number not identified", "review",
            f"Could not confidently extract a PO number from {po_doc.get('filename', 'purchase order PDF')}. Check the document manually.",
            evidence_for(po_doc),
        ))

    def extract_party(text: str) -> str | None:
        patterns = [
            r"(?:vendor|supplier)\s*(?:name)?\s*[:\-]\s*([^\n\r]{3,90})",
            r"(?:billed\s+by|sold\s+by|from)\s*[:\-]\s*([^\n\r]{3,90})",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.I)
            if m:
                return re.sub(r"\s+", " ", m.group(1)).strip(" .,:;")
        return None

    invoice_party = extract_party(invoice_doc.get("text", ""))
    po_party = extract_party(po_doc.get("text", ""))
    if invoice_party and po_party:
        norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
        if norm(invoice_party) != norm(po_party) and not (norm(invoice_party) in norm(po_party) or norm(po_party) in norm(invoice_party)):
            findings.append(finding(
                "Vendor names may not match", "high",
                f"Invoice vendor appears to be '{invoice_party}', while the PO vendor appears to be '{po_party}'. Verify that these documents belong together.",
                evidence_for(invoice_doc) + evidence_for(po_doc),
            ))

    # Extract labelled totals and compare only like-for-like fields.
    def extract_money(text: str, labels: str) -> float | None:
        pattern = rf"(?:{labels})\s*(?:\(\s*(?:INR|Rs\.?|₹)\s*\))?\s*[:=\-]?\s*(?:INR|Rs\.?|₹)?\s*([0-9][0-9,]*(?:\.\d{{1,2}})?)"
        matches = list(re.finditer(pattern, text, re.I))
        if not matches:
            return None
        try:
            return float(matches[-1].group(1).replace(",", ""))
        except ValueError:
            return None

    total_labels = r"(?:grand\s+total|total\s+amount|invoice\s+total|amount\s+due|net\s+payable|total\s+payable)"
    invoice_total = extract_money(invoice_doc.get("text", ""), total_labels)
    po_total = extract_money(po_doc.get("text", ""), r"(?:grand\s+total|total\s+amount|order\s+total|total\s+value|PO\s+value)")
    if invoice_total is not None and po_total is not None:
        if abs(invoice_total - po_total) > 0.01:
            findings.append(finding(
                "Invoice and PO totals differ", "high",
                f"Extracted invoice total is {invoice_total:,.2f}; extracted PO total is {po_total:,.2f}; difference is {invoice_total - po_total:,.2f}. Confirm taxes, freight, discounts, and approved changes before payment.",
                evidence_for(invoice_doc) + evidence_for(po_doc),
            ))
        else:
            findings.append(finding(
                "Invoice and PO totals match", "low",
                f"The extracted totals match at {invoice_total:,.2f}. This check does not independently validate every line item or tax calculation.",
                evidence_for(invoice_doc) + evidence_for(po_doc),
            ))
    else:
        findings.append(finding(
            "Total amount could not be compared", "review",
            "A clearly labelled total was not confidently extracted from both PDFs. Verify the invoice and PO totals manually; the matcher does not guess missing values.",
            evidence_for(invoice_doc) + evidence_for(po_doc),
        ))

    # Explainable line-item extraction. PDF libraries often return table cells
    # vertically (description, quantity, unit price, line total), not as rows.
    def extract_items(doc: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[tuple[str, float, float, float, int]] = set()
        number_re = re.compile(r"^(?:INR\s*|Rs\.?\s*|₹\s*)?([0-9][0-9,]*(?:\.\d{1,2})?)$", re.I)
        ignored = re.compile(
            r"^(description|quantity|unit price(?: \(.*?\))?|line total(?: \(.*?\))?|"
            r"subtotal.*|grand total.*|tax total.*|invoice total.*|amount due.*|gst.*|"
            r"purchase order.*|invoice number.*|invoice date.*|order date.*|vendor.*|customer.*|buyer.*|"
            r"fictional test document.*|page \d+)$", re.I
        )

        for page in doc.get("pages", []):
            lines = [re.sub(r"\s+", " ", raw).strip(" \t|•") for raw in page.get("text", "").splitlines()]
            lines = [line for line in lines if line]
            # First handle conventional single-line rows.
            for line in lines:
                if ignored.match(line):
                    continue
                m = re.match(
                    r"^(.*?)[\s|,;]+(\d+(?:\.\d+)?)\s+(?:x\s*)?(?:INR|Rs\.?|₹)?\s*([0-9][0-9,]*(?:\.\d{1,2})?)\s+(?:INR|Rs\.?|₹)?\s*([0-9][0-9,]*(?:\.\d{1,2})?)$",
                    line, re.I
                )
                if m:
                    try:
                        desc, qty, unit, total = m.group(1).strip(" -:|"), float(m.group(2)), float(m.group(3).replace(",", "")), float(m.group(4).replace(",", ""))
                    except ValueError:
                        continue
                    if desc and re.search(r"[A-Za-z]{2}", desc) and qty > 0:
                        key = (desc.lower(), qty, unit, total, page.get("page", 1))
                        if key not in seen:
                            items.append({"description": desc[:120], "quantity": qty, "unit_price": unit, "line_total": total, "page": page.get("page", 1), "line": line})
                            seen.add(key)

            # Then handle cell-per-line extraction, e.g. description / qty / unit / total.
            i = 0
            while i + 3 < len(lines):
                desc = lines[i].strip()
                if ignored.match(desc) or not re.search(r"[A-Za-z]{2}", desc):
                    i += 1
                    continue
                nums = []
                for candidate in lines[i + 1:i + 4]:
                    nm = number_re.match(candidate.replace(",", ","))
                    if not nm:
                        break
                    try:
                        nums.append(float(nm.group(1).replace(",", "")))
                    except ValueError:
                        break
                if len(nums) == 3 and nums[0] > 0:
                    qty, unit, total = nums
                    # Keep the line item even if its arithmetic is inconsistent.
                    # Otherwise a malformed invoice row (e.g. qty 12 x 800 = 8000)
                    # disappears and is incorrectly reported as an unmatched PO item.
                    key = (desc.lower(), qty, unit, total, page.get("page", 1))
                    if key not in seen:
                        items.append({"description": desc[:120], "quantity": qty, "unit_price": unit, "line_total": total, "page": page.get("page", 1), "line": f"{desc} | {qty:g} | {unit:g} | {total:g}"})
                        seen.add(key)
                    i += 4
                    continue
                i += 1
        return items

    invoice_items = extract_items(invoice_doc)
    po_items = extract_items(po_doc)

    # Validate arithmetic independently from invoice-vs-PO comparisons.
    # This catches internally inconsistent rows even when both documents' grand totals match.
    def add_line_arithmetic_findings(items: list[dict[str, Any]], doc: dict[str, Any], role: str) -> None:
        for item in items:
            calculated = item["quantity"] * item["unit_price"]
            if abs(calculated - item["line_total"]) > max(0.02, abs(item["line_total"]) * 0.0001):
                findings.append(finding(
                    f"{role} line-item calculation mismatch", "high",
                    f"'{item['description']}': quantity {item['quantity']:g} × unit price {item['unit_price']:,.2f} = {calculated:,.2f}, but the extracted line total is {item['line_total']:,.2f}. Check the source document.",
                    [get_page_evidence(doc, item["page"])],
                ))

    add_line_arithmetic_findings(invoice_items, invoice_doc, "Invoice")
    add_line_arithmetic_findings(po_items, po_doc, "Purchase order")

    # Reconcile invoice arithmetic and tax components when the PDF exposes
    # sufficiently clear labels. These are deterministic checks, not tax advice.
    def labelled_amount(text: str, labels: str) -> float | None:
        return extract_money(text, labels)

    def add_amount_finding(title: str, description: str, doc: dict[str, Any], severity: str = "review") -> None:
        findings.append(finding(title, severity, description, evidence_for(doc)))

    invoice_text = invoice_doc.get("text", "")
    po_text = po_doc.get("text", "")

    invoice_subtotal = labelled_amount(
        invoice_text, r"(?:(?:taxable\s+value)|(?:sub\s*total)|(?:subtotal)|(?:total\s+before\s+tax))"
    )
    invoice_discount = labelled_amount(
        invoice_text, r"(?:discount(?:\s+amount)?|less\s+discount)"
    )
    invoice_tax_total = labelled_amount(
        invoice_text, r"(?:(?:total\s+gst)|(?:gst\s+amount)|(?:total\s+tax)|(?:tax\s+amount)|(?:tax\s+total))"
    )

    # Sum extracted line totals only if line items were actually parsed.
    invoice_lines_sum = sum(item["line_total"] for item in invoice_items) if invoice_items else None
    po_lines_sum = sum(item["line_total"] for item in po_items) if po_items else None

    if invoice_subtotal is not None and invoice_lines_sum is not None:
        if abs(invoice_subtotal - invoice_lines_sum) > 0.02:
            add_amount_finding(
                "Invoice subtotal does not match line items",
                f"Extracted invoice line totals sum to {invoice_lines_sum:,.2f}, but the labelled subtotal/taxable value is {invoice_subtotal:,.2f}. Check omitted rows, discounts, freight, or extraction accuracy.",
                invoice_doc, "high"
            )
        else:
            add_amount_finding(
                "Invoice subtotal matches extracted line items",
                f"The extracted invoice line totals sum to {invoice_lines_sum:,.2f}, matching the labelled subtotal/taxable value.",
                invoice_doc, "low"
            )

    if invoice_subtotal is not None and invoice_discount is not None and invoice_discount > invoice_subtotal:
        add_amount_finding(
            "Discount exceeds invoice subtotal",
            f"The extracted discount ({invoice_discount:,.2f}) is greater than the subtotal ({invoice_subtotal:,.2f}). Verify labels and amounts.",
            invoice_doc, "high"
        )

    # Validate explicitly labelled GST components and rates. Do not assume GST
    # is always split as CGST/SGST; IGST is supported as a separate component.
    def extract_tax_component(text: str, component: str) -> tuple[float | None, float | None]:
        patterns = [
            rf"\b{component}\b\s*(?:\(\s*(\d+(?:\.\d+)?)\s*%\s*\))?\s*[:=\-]?\s*(?:INR|Rs\.?|₹)?\s*([0-9][0-9,]*(?:\.\d{{1,2}})?)",
            rf"\b{component}\b\s*(\d+(?:\.\d+)?)\s*%\s*(?:GST)?\s*[:=\-]?\s*(?:INR|Rs\.?|₹)?\s*([0-9][0-9,]*(?:\.\d{{1,2}})?)",
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.I))
            if matches:
                match = matches[-1]
                rate = float(match.group(1)) if match.group(1) else None
                try:
                    return rate, float(match.group(2).replace(",", ""))
                except ValueError:
                    continue
        return None, None

    tax_components = []
    for component in ("CGST", "SGST", "IGST", "GST"):
        rate, amount = extract_tax_component(invoice_text, component)
        if amount is not None:
            tax_components.append((component, rate, amount))
            if invoice_subtotal is not None and rate is not None:
                expected = invoice_subtotal * rate / 100.0
                if abs(expected - amount) > max(0.02, expected * 0.005):
                    add_amount_finding(
                        f"{component} amount does not match stated rate",
                        f"The extracted {component} rate is {rate:g}% on a subtotal/taxable value of {invoice_subtotal:,.2f}, which implies {expected:,.2f}; the extracted component amount is {amount:,.2f}. Verify taxable base, exemptions, rounding, and extraction.",
                        invoice_doc, "high"
                    )

    if any(name in {"CGST", "SGST", "IGST"} for name, _, _ in tax_components):
        tax_components = [component for component in tax_components if component[0] != "GST"]

    if len(tax_components) >= 2:
        component_sum = sum(component[2] for component in tax_components)
        if invoice_tax_total is not None and abs(component_sum - invoice_tax_total) > 0.02:
            add_amount_finding(
                "GST components do not match total tax",
                f"Extracted GST components sum to {component_sum:,.2f}, but the labelled total tax/GST amount is {invoice_tax_total:,.2f}. Verify tax breakdown and extraction.",
                invoice_doc, "high"
            )
        elif invoice_tax_total is None:
            add_amount_finding(
                "GST components extracted",
                f"Extracted tax components ({', '.join(f'{name} {amount:,.2f}' for name, _, amount in tax_components)}) sum to {component_sum:,.2f}. No clearly labelled total tax amount was found for reconciliation.",
                invoice_doc, "review"
            )

    # Reconcile subtotal + tax - discount to the grand total only when all
    # required values are confidently available. Other charges may still apply.
    if invoice_subtotal is not None and invoice_tax_total is not None and invoice_total is not None:
        expected_total = invoice_subtotal + invoice_tax_total - (invoice_discount or 0.0)
        if abs(expected_total - invoice_total) > 0.02:
            add_amount_finding(
                "Invoice subtotal, tax, discount, and grand total do not reconcile",
                f"Subtotal {invoice_subtotal:,.2f} + tax {invoice_tax_total:,.2f} - discount {(invoice_discount or 0.0):,.2f} = {expected_total:,.2f}, but the extracted grand total is {invoice_total:,.2f}. Check freight, other charges, rounding, and extraction.",
                invoice_doc, "high"
            )
        else:
            add_amount_finding(
                "Invoice grand total reconciles",
                f"Subtotal {invoice_subtotal:,.2f} + tax {invoice_tax_total:,.2f} - discount {(invoice_discount or 0.0):,.2f} reconciles to the extracted grand total {invoice_total:,.2f}. Other charges should still be reviewed if present.",
                invoice_doc, "low"
            )

    if po_lines_sum is not None:
        po_subtotal = labelled_amount(
            po_text, r"(?:(?:taxable\s+value)|(?:sub\s*total)|(?:subtotal)|(?:total\s+before\s+tax))"
        )
        if po_subtotal is not None and abs(po_subtotal - po_lines_sum) > 0.02:
            add_amount_finding(
                "Purchase order subtotal does not match line items",
                f"Extracted PO line totals sum to {po_lines_sum:,.2f}, but the labelled PO subtotal is {po_subtotal:,.2f}. Verify line items and extraction.",
                po_doc, "high"
            )

    if invoice_items and po_items:
        # First pass: normalized exact descriptions, then optional local LLM semantic suggestions.
        normalize = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
        unmatched_invoice = list(invoice_items)
        unmatched_po = list(po_items)
        pairs = []
        for ii in list(unmatched_invoice):
            hit = next((pp for pp in unmatched_po if normalize(ii["description"]) == normalize(pp["description"])), None)
            if hit:
                pairs.append((ii, hit, "exact"))
                unmatched_invoice.remove(ii); unmatched_po.remove(hit)

        ai_used = False
        if unmatched_invoice and unmatched_po:
            # AI can suggest description equivalence; numeric comparisons remain Python-only.
            try:
                prompt_data = {"invoice_items": [{"description": x["description"], "quantity": x["quantity"], "unit_price": x["unit_price"]} for x in unmatched_invoice],
                               "purchase_order_items": [{"description": x["description"], "quantity": x["quantity"], "unit_price": x["unit_price"]} for x in unmatched_po]}
                payload = {"model": __import__("os").environ.get("DOCUAUDIT_OLLAMA_MODEL", "qwen2.5:3b"), "stream": False,
                           "format": "json", "messages": [
                    {"role": "system", "content": "Match invoice line descriptions to semantically equivalent PO line descriptions. Do not invent matches. Return JSON only: {\"matches\":[{\"invoice_index\":0,\"po_index\":0,\"reason\":\"short reason\"}]}. Indices are zero-based. Only match when clearly equivalent; ignore quantities and prices for matching."},
                    {"role": "user", "content": json.dumps(prompt_data, ensure_ascii=False)}]}
                req = urllib.request.Request(__import__("os").environ.get("DOCUAUDIT_OLLAMA_URL", "http://localhost:11434/api/chat"), data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=float(__import__("os").environ.get("DOCUAUDIT_OLLAMA_TIMEOUT", "30"))) as response:
                    result = json.loads(response.read().decode("utf-8"))
                content = result.get("message", {}).get("content", "")
                proposed = json.loads(content).get("matches", [])
                used_i, used_p = set(), set()
                for item in proposed:
                    i, j = item.get("invoice_index"), item.get("po_index")
                    if isinstance(i, int) and isinstance(j, int) and 0 <= i < len(unmatched_invoice) and 0 <= j < len(unmatched_po) and i not in used_i and j not in used_p:
                        pairs.append((unmatched_invoice[i], unmatched_po[j], "AI-suggested"))
                        used_i.add(i); used_p.add(j); ai_used = True
                unmatched_invoice = [x for i, x in enumerate(unmatched_invoice) if i not in used_i]
                unmatched_po = [x for j, x in enumerate(unmatched_po) if j not in used_p]
            except Exception:
                # Optional AI assistance must never prevent deterministic checks.
                pass

        for inv_item, po_item, match_type in pairs:
            issues = []
            if abs(inv_item["quantity"] - po_item["quantity"]) > 1e-9:
                issues.append(f"quantity differs (invoice {inv_item['quantity']:g}, PO {po_item['quantity']:g})")
            if abs(inv_item["unit_price"] - po_item["unit_price"]) > 0.01:
                issues.append(f"unit price differs (invoice {inv_item['unit_price']:,.2f}, PO {po_item['unit_price']:,.2f})")
            if abs(inv_item["line_total"] - po_item["line_total"]) > 0.01:
                issues.append(f"line total differs (invoice {inv_item['line_total']:,.2f}, PO {po_item['line_total']:,.2f})")
            if issues:
                findings.append(finding(
                    "Invoice line-item mismatch", "high",
                    f"'{inv_item['description']}' matched to PO item '{po_item['description']}' ({match_type}); " + "; ".join(issues) + ". Verify the source lines before payment.",
                    [get_page_evidence(invoice_doc, inv_item["page"]), get_page_evidence(po_doc, po_item["page"])],
                ))
            else:
                findings.append(finding(
                    "Invoice line item matched", "low",
                    f"'{inv_item['description']}' matches PO item '{po_item['description']}' on quantity, unit price, and line total ({match_type} description match).",
                    [get_page_evidence(invoice_doc, inv_item["page"]), get_page_evidence(po_doc, po_item["page"])],
                ))
        for item in unmatched_invoice:
            findings.append(finding(
                "Invoice item not found on purchase order", "high",
                f"Invoice item '{item['description']}' was not confidently matched to a PO line item. Check whether it was authorized.",
                [get_page_evidence(invoice_doc, item["page"]), get_page_evidence(po_doc, first_page(po_doc))],
            ))
        for item in unmatched_po:
            findings.append(finding(
                "Purchase order item not billed", "review",
                f"PO item '{item['description']}' was not confidently matched to an invoice line item. It may be unbilled, described differently, or omitted from the invoice.",
                [get_page_evidence(po_doc, item["page"]), get_page_evidence(invoice_doc, first_page(invoice_doc))],
            ))
    else:
        findings.append(finding(
            "Line items could not be reliably extracted", "review",
            "The matcher could not identify comparable table-like line items in both PDFs. This may happen when tables are split across lines or the PDFs use unusual layouts. Verify line items manually; OCR is not included.",
            evidence_for(invoice_doc) + evidence_for(po_doc),
        ))

    # Overall reconciliation status: a matching subtotal/grand total must never
    # imply that the invoice is safe when any line-level or tax check failed.
    high_findings = [item for item in findings if item.get("severity") == "high"]
    if high_findings:
        issue_titles = list(dict.fromkeys(item.get("title", "Discrepancy") for item in high_findings))
        summary_text = (
            "Mismatch detected — manual review required before payment. "
            f"{len(high_findings)} high-severity issue(s) were detected: "
            + "; ".join(issue_titles[:6])
            + ("; additional issues are listed below." if len(issue_titles) > 6 else ".")
            + " Matching invoice/PO totals or subtotals do not override line-item, arithmetic, or tax discrepancies."
        )
        findings.insert(0, finding(
            "Invoice and PO reconciliation: mismatch detected", "high",
            summary_text,
            evidence_for(invoice_doc) + evidence_for(po_doc),
        ))
        # Qualify otherwise-positive aggregate checks to avoid a misleading green signal.
        for item in findings[1:]:
            title = item.get("title", "")
            if item.get("severity") == "low" and any(
                phrase in title.lower() for phrase in (
                    "invoice and po totals match",
                    "invoice subtotal matches",
                    "invoice grand total reconciles",
                )
            ):
                item["title"] = title + " (line-level issue still present)"
                item["description"] = item.get("description", "") + (
                    " Important: this aggregate check passed, but the overall reconciliation has a high-severity discrepancy. "
                    "Do not treat this result as approval to pay."
                )
    else:
        findings.insert(0, finding(
            "Invoice and PO reconciliation: review findings",
            "review",
            "No high-severity discrepancy was detected by the available checks. This is not an automatic approval: verify extraction quality, missing fields, tax treatment, and the source PDFs before payment.",
            evidence_for(invoice_doc) + evidence_for(po_doc),
        ))

    return findings


# ==================================================
# 8. MAIN AUDIT ENDPOINT
# ==================================================

@app.post("/api/audit")
async def audit(
    service: str = Form(...),
    files: list[UploadFile] = File(
        ...,
        description="Select one or more PDF files",
        media_type="application/pdf",
    ),
) -> dict[str, Any]:
    started_at = time.perf_counter()

    if service not in SERVICES:
        raise HTTPException(
            status_code=400,
            detail="Unknown audit service.",
        )

    if not files or len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Upload between 1 and {MAX_FILES} PDFs.",
        )

    if service == "invoice" and len(files) < 2:
        raise HTTPException(
            status_code=400,
            detail="Invoice matching requires at least two PDFs.",
        )

    try:
        docs = [
            await extract_pdf(upload)
            for upload in files
        ]
    finally:
        for upload in files:
            await upload.close()

    unreadable_files = [
        doc.get("filename", "document.pdf")
        for doc in docs
        if not doc.get("has_extractable_text", False)
    ]

    note = (
        "No selectable text was found in: "
        + ", ".join(unreadable_files)
        + ". OCR support is not included yet."
        if unreadable_files
        else None
    )

    ml_diagnostics = []

    if service == "document":
        findings = document_audit(docs)
        service_name = "AI Document Auditor"

    elif service == "financial":
        findings, ml_diagnostics = financial_audit(docs)
        service_name = "AI Financial Report Detective"

    else:
        findings = invoice_audit(docs)
        service_name = "AI Invoice and Purchase Order Matcher"

    documents_response = []

    for doc in docs:
        extracted_text = doc.get("text", "")

        documents_response.append({
            "filename": doc.get("filename", "document.pdf"),
            "page_count": doc.get("page_count", 0),
            "has_extractable_text": doc.get(
                "has_extractable_text",
                bool(extracted_text.strip()),
            ),
            "character_count": doc.get(
                "character_count",
                len(extracted_text),
            ),
            "extraction_status": doc.get(
                "extraction_status", "unknown"
            ),
            "likely_scanned_pages": doc.get(
                "likely_scanned_pages", []
            ),
            "page_diagnostics": [
                {
                    "page": page.get("page"),
                    "character_count": page.get(
                        "character_count",
                        len(page.get("text", "")),
                    ),
                    "word_count": page.get(
                        "word_count",
                        len(page.get("text", "").split()),
                    ),
                    "likely_scanned": page.get(
                        "likely_scanned", False
                    ),
                    "extraction_status": page.get(
                        "extraction_status",
                        "low_text" if page.get(
                            "likely_scanned", False
                        ) else "good",
                    ),
                }
                for page in doc.get("pages", [])
            ],
            "preview": doc.get(
                "preview", extracted_text[:1200]
            ),
        })

    elapsed_ms = round(
        (time.perf_counter() - started_at) * 1000,
        2,
    )

    response = {
        "service": service,
        "service_name": service_name,
        "documents": documents_response,
        "findings": findings,
        "note": note,
        "processing_time_ms": elapsed_ms,
    }

    # Return ML diagnostics only for the financial service.
    if service == "financial":
        response["ml_diagnostics"] = ml_diagnostics

    return response


# ==================================================
# 9. CUSTOM OPENAPI SCHEMA
# ==================================================

def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    operation = schema["paths"]["/api/audit"]["post"]

    request_body = (
        operation["requestBody"]["content"]
        ["multipart/form-data"]["schema"]
    )

    if "$ref" in request_body:
        ref_name = request_body["$ref"].split("/")[-1]
        request_schema = schema["components"]["schemas"][ref_name]
    else:
        request_schema = request_body

    request_schema["properties"]["files"] = {
        "type": "array",
        "title": "Files",
        "description": "Upload one or more PDF files.",
        "items": {
            "type": "string",
            "format": "binary",
        },
    }

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
