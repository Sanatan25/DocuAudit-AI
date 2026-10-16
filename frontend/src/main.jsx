import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const services = [
  { id: 'document', number: '01', icon: '◈', title: 'AI Document Auditor', desc: 'Find missing sections, conflicting dates, and inconsistencies in business documents.', upload: 'Upload a contract, policy, or business document.' },
  { id: 'financial', number: '02', icon: '▥', title: 'AI Financial Report Detective', desc: 'Extract financial values and identify figures that need further verification.', upload: 'Upload a financial statement or annual report.' },
  { id: 'invoice', number: '03', icon: '⇄', title: 'AI Invoice & Purchase Order Matcher', desc: 'Compare invoice and purchase order details to surface potential mismatches.', upload: 'Upload both the invoice and purchase order PDFs.' },
]

function App() {
  const [service, setService] = useState(null)
  const [files, setFiles] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const selected = services.find(s => s.id === service)

  function reset() {
    setService(null); setFiles([]); setResult(null); setError('')
  }

  async function runAudit(e) {
    e.preventDefault()
    setError(''); setResult(null)
    if (!service) return setError('Choose an audit service first.')
    if (!files.length) return setError('Choose at least one PDF.')
    if (service === 'invoice' && files.length < 2) return setError('Please upload both an invoice and a purchase order.')
    const form = new FormData()
    form.append('service', service)
    files.forEach(file => form.append('files', file))
    setLoading(true)
    try {
      const response = await fetch(`${API}/api/audit`, { method: 'POST', body: form })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'The audit could not be completed.')
      setResult(data)
    } catch (err) {
      setError(`${err.message}. Check that the FastAPI backend is running at ${API}.`)
    } finally {
      setLoading(false)
    }
  }

  function downloadJSON() {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'docuaudit-results.json'; a.click()
    URL.revokeObjectURL(url)
  }

  return <div className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={reset} aria-label="DocuAudit AI home">
        <span className="brand-mark">D</span><span>DocuAudit<span className="brand-ai"> AI</span></span>
      </button>
      <span className="top-note"><span className="status-dot"></span> Document intelligence workspace</span>
    </header>

    <main>
      {!selected && !result && <section className="hero">
        <div className="eyebrow"><span className="spark">✦</span> DOCUMENT INTELLIGENCE, REIMAGINED</div>
        <h1>Make every document<br/><span className="gradient-text">work harder for you.</span></h1>
        <p className="hero-copy">Choose what you need to audit. DocuAudit AI will process your PDFs and present findings with evidence for review.</p>
        <div className="service-grid">
          {services.map(s => <button className="service-card" key={s.id} onClick={() => { setService(s.id); setFiles([]); setError('') }}>
            <div className="card-top"><span className="service-icon">{s.icon}</span><span className="card-number">{s.number}</span></div>
            <h2>{s.title}</h2><p>{s.desc}</p>
            <div className="card-action">Select service <span>↗</span></div>
          </button>)}
        </div>
        <div className="privacy-note"><span>⌑</span> Your documents are processed by this backend and are not saved to a database in this MVP.</div>
      </section>}

      {selected && !result && <section className="workspace">
        <button className="back-link" onClick={() => { setService(null); setFiles([]); setError('') }}>← Back to services</button>
        <div className="workspace-heading"><span className="service-icon large">{selected.icon}</span><div><div className="eyebrow">SELECTED WORKFLOW</div><h1>{selected.title}</h1><p>{selected.upload}</p></div></div>
        <form className="upload-panel" onSubmit={runAudit}>
          <label className="dropzone">
            <input type="file" accept=".pdf,application/pdf" multiple onChange={e => { setFiles(Array.from(e.target.files || [])); setError('') }}/>
            <span className="upload-icon">↑</span>
            <strong>Choose PDF file{service === 'invoice' ? 's' : ''}</strong>
            <span className="muted">{service === 'invoice' ? 'Select at least 2 PDFs: invoice and purchase order' : 'PDF only · Maximum 15 MB per file · Up to 3 files'}</span>
            <span className="browse-button">Browse files</span>
          </label>
          {files.length > 0 && <div className="file-list">{files.map((f, i) => <div className="file-row" key={`${f.name}-${i}`}><span className="pdf-badge">PDF</span><span className="file-name">{f.name}</span><span className="file-size">{(f.size / 1024 / 1024).toFixed(2)} MB</span><button type="button" className="remove-file" onClick={() => setFiles(old => old.filter((_, j) => i !== j))}>×</button></div>)}</div>}
          {error && <div className="error-message">{error}</div>}
          <button className="primary-button" type="submit" disabled={loading}>{loading ? <><span className="spinner"></span> Analyzing documents…</> : <>Analyze documents <span>→</span></>}</button>
          <p className="form-footnote">Review all findings against the original documents. This MVP uses basic extraction and heuristic checks.</p>
        </form>
      </section>}

      {result && <section className="results">
        <button className="back-link" onClick={reset}>← Start a new audit</button>
        <div className="results-title"><div><div className="eyebrow">AUDIT COMPLETE</div><h1>{result.service_name}</h1><p>Review the findings below and verify them against the original PDFs.</p></div><button className="secondary-button" onClick={downloadJSON}>↓ Export results JSON</button></div>
        <div className="summary-grid">
          <div className="summary-card"><span>Documents processed</span><strong>{result.documents.length}</strong></div>
          <div className="summary-card"><span>Findings to review</span><strong>{result.findings.length}</strong></div>
          <div className="summary-card"><span>Pages scanned</span><strong>{result.documents.reduce((n, d) => n + d.page_count, 0)}</strong></div>
        </div>
        {(result.note || result.notice) && (
  <div className="notice-box">
    ⓘ {result.note || result.notice}
  </div>
)}
        <div className="document-chips">{result.documents.map(d => <span className="doc-chip" key={d.filename}>▤ {d.filename} · {d.page_count} pages</span>)}</div>
        
<div className="extraction-section">
  <h2 className="section-heading">PDF Extraction Diagnostics</h2>

  {result.documents.map((doc) => (
    <article className="finding-card" key={doc.filename}>
      <h3>{doc.filename}</h3>

      <div className="summary-grid">
        <div className="summary-card">
          <span>Characters extracted</span>
          <strong>{doc.character_count ?? 0}</strong>
        </div>

        <div className="summary-card">
          <span>Extraction status</span>
          <strong>{doc.extraction_status ?? 'Unknown'}</strong>
        </div>

        <div className="summary-card">
          <span>Pages needing OCR review</span>
          <strong>{doc.likely_scanned_pages?.length ?? 0}</strong>
        </div>
      </div>

      {doc.likely_scanned_pages?.length > 0 && (
        <p>
          Pages with little extractable text:{" "}
          {doc.likely_scanned_pages.join(", ")}
        </p>
      )}

      <details>
        <summary>View page-by-page diagnostics</summary>
        <div className="page-diagnostics">
          {(doc.page_diagnostics || []).map((page) => (
            <div className="file-row" key={page.page}>
              <span>Page {page.page}</span>
              <span>{page.character_count} characters</span>
              <span>{page.word_count} words</span>
              <span>{page.extraction_status}</span>
            </div>
          ))}
        </div>
      </details>

      {doc.preview && (
        <details>
          <summary>Preview extracted PDF text</summary>
          <pre className="extraction-preview">{doc.preview}</pre>
        </details>
      )}
    </article>
  ))}
</div>

        <h2 className="section-heading">Findings & evidence</h2>
        {result.findings.length === 0 && <div className="empty-state">No findings were generated by the current checks. This does not guarantee that the documents are error-free.</div>}
        <div className="finding-list">{result.findings.map((f, i) => <article className="finding-card" key={i}>
          <div className="finding-head"><span className={`severity ${f.severity}`}>{f.severity}</span><span className="finding-index">FINDING {String(i + 1).padStart(2, '0')}</span></div>
          <h3>{f.title}</h3><p>{f.description}</p>
          {f.evidence?.length > 0 && <div className="evidence"><strong>Source evidence</strong>{f.evidence.map((ev, j) => <div className="evidence-row" key={j}><span className="evidence-page">{ev.filename}{ev.page ? ` · Page ${ev.page}` : ''}</span><blockquote>
  {ev.text?.trim() || ev.snippet?.trim() || 'No extractable text was available for this evidence.'}
</blockquote></div>)}</div>}
        </article>)}</div>
        <div className="disclaimer">{result.disclaimer}</div>
      </section>}
    </main>
    <footer><span>DocuAudit AI <span className="muted">· MVP</span></span><span>Built for evidence-led document review</span></footer>
  </div>
}

createRoot(document.getElementById('root')).render(<App />)
