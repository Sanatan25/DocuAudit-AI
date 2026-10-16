<<<<<<< HEAD
# DocuAudit AI — MVP

A starter application with a React frontend and FastAPI backend. The home page asks users to choose one of three audit services before uploading PDFs.

## Workflows
1. AI Document Auditor — checks for basic text-based inconsistencies and missing common sections.
2. AI Financial Report Detective — extracts financial-looking values and calculates simple totals where possible; findings are prompts for human review, not financial advice.
3. AI Invoice and Purchase Order Matcher — compares extracted line items where recognizable; upload both documents.

## Requirements
- Python 3.10+
- Node.js 18+

## Start backend
```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API: http://127.0.0.1:8000
API docs: http://127.0.0.1:8000/docs

## Start frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```
Open the URL printed by Vite (usually http://localhost:5173).

## Notes
- This MVP processes text-based PDFs using PyMuPDF. Scanned PDFs need OCR, which is a planned next phase.
- Audit results are rule-based starter findings. They are not a substitute for professional legal, accounting, or procurement review.
- For invoice matching, upload one invoice and one purchase order. Delivery receipt support is planned.
- Do not upload sensitive documents to a public deployment until authentication, access controls, secure storage, retention policies and production security have been implemented.
=======
# DocuAudit-AI
>>>>>>> 8ba0dcfba250b7a08fda84e4fbf582df73d3c305
