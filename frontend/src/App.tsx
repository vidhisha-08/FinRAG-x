import { useState } from "react"
import { ask, type AskResponse } from "./api"
import "./App.css"

const EXAMPLE =
  "What was the percentage change in AES Corporation's proportional free cash flow between 2013 and 2014, as reported in the 2015 financial report?"

const CHECK_LABELS: Record<string, string> = {
  answered: "An answer was produced",
  citations_valid: "Cited sources are among the retrieved evidence",
  inputs_grounded: "Calculation inputs appear in the evidence",
  result_in_answer: "Computed result appears in the answer",
  answer_numbers_supported: "Numbers in the answer appear in the evidence",
}

const MODES: [string, string][] = [
  ["hybrid_rerank", "Hybrid + rerank"],
  ["hybrid", "Hybrid"],
  ["dense", "Dense only"],
  ["bm25", "BM25 only"],
]

function fmt(n: number, d = 4): string {
  return n.toLocaleString("en-US", { maximumFractionDigits: d }).replace("-", "\u2212")
}

function fmtArg(a: any): string {
  if (typeof a === "string" && a.startsWith("#")) return `result of step ${Number(a.slice(1)) + 1}`
  if (typeof a === "number") return fmt(a)
  return String(a)
}

function parseMeta(text: string) {
  const m = text.match(/^\[([^\]]*)\]\s*/)
  if (!m) return { company: "", year: "", page: "", body: text }
  const parts = m[1].split("|").map(s => s.trim())
  return {
    company: parts[0] || "",
    year: parts[1] || "",
    page: (parts[2] || "").replace(/^page\s*/i, ""),
    body: text.slice(m[0].length),
  }
}

function parseTable(body: string): string[][] {
  const rows: string[][] = []
  for (const line of body.split("\n")) {
    if (!line.trim().startsWith("|")) continue
    const cells = line.split("|").map(c => c.trim())
    cells.shift()
    if (cells.length && cells[cells.length - 1] === "") cells.pop()
    if (cells.length && cells.every(c => /^:?-{2,}:?$/.test(c))) continue
    rows.push(cells)
  }
  // drop the leading index column when its header cell is blank
  if (rows.length && rows[0][0] === "") rows.forEach(r => r.shift())
  return rows
}

const isNumeric = (c: string) => /^[\s$(\u2212-]*[\d,.]+\)?\s*%?$/.test(c)

function EvidenceTable({ body }: { body: string }) {
  const rows = parseTable(body)
  if (rows.length < 2) return <pre className="excerpt">{body}</pre>
  const [head, ...rest] = rows
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{head.map((h, i) => <th key={i}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rest.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (
                <td key={j} className={isNumeric(c) ? "num" : ""}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function App() {
  const [question, setQuestion] = useState("")
  const [mode, setMode] = useState("hybrid_rerank")
  const [data, setData] = useState<AskResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [open, setOpen] = useState<string | null>(null)

  async function submit() {
    if (!question.trim() || loading) return
    setLoading(true); setError(""); setData(null)
    try {
      const res = await ask(question, mode)
      setData(res)
      const cited = (res.answer + " " + ((res as any).raw_answer || "")).match(/D\d+_C\d+/g)
      setOpen((cited && cited[0]) || res.evidence[0]?.id || null)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  function jumpTo(id: string) {
    setOpen(id)
    setTimeout(() => {
      document.getElementById(`ev-${id}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" })
    }, 50)
  }

  const program: any = data ? (data as any).program : null
  const steps: any[] = program?.steps || []
  const result: number | null = data ? ((data as any).result ?? null) : null
  const isPercent = !!program?.percent
  const passed = !!data?.passed
  const rawAnswer: string = data ? (data as any).raw_answer || "" : ""
  const citedIds = new Set<string>(
    data ? ((data.answer + " " + rawAnswer).match(/D\d+_C\d+/g) || []) : []
  )

  const isChangePattern =
    steps.length === 2 && steps[0].op === "subtract" && steps[1].op === "divide" &&
    steps[1].args?.[0] === "#0" && isPercent

  function renderAnswer(text: string) {
    return text.split(/(\[\s*D\d+_C\d+\s*\])/g).map((part, i) => {
      const m = part.match(/D\d+_C\d+/)
      if (m && /^\[/.test(part)) {
        return (
          <button key={i} className="cite" onClick={() => jumpTo(m[0])}>{m[0]}</button>
        )
      }
      return <span key={i}>{part}</span>
    })
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-name">FinRAG-X</span>
          <span className="brand-sub">Financial Research &amp; Evidence</span>
        </div>
      </header>

      <main className="page">
        <section className="ask">
          <h2 className="section-title">Ask the report</h2>
          <textarea
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit() }}
            placeholder="Ask a question about a financial report…"
            aria-label="Question"
          />
          <div className="ask-row">
            <label className="mode">
              <span>Retrieval mode</span>
              <select value={mode} onChange={e => setMode(e.target.value)}>
                {MODES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </label>
            <button className="primary" onClick={submit} disabled={loading || !question.trim()}>
              {loading ? "Running…" : "Run analysis"}
            </button>
          </div>
          <p className="examples">
            Example:{" "}
            <button className="textlink" onClick={() => setQuestion(EXAMPLE)}>
              AES proportional free cash flow, 2013 to 2014
            </button>
          </p>
        </section>

        {loading && (
          <p className="status-line">Retrieving evidence, calculating and verifying the answer…</p>
        )}

        {error && (
          <div className="notice error" role="alert">
            <strong>Request failed.</strong> {error}
          </div>
        )}

        {!data && !loading && !error && (
          <p className="empty">
            Results appear here with the calculation, the retrieved evidence and the verification checks.
          </p>
        )}

        {data && (
          <div className="workspace">
            <div className="col-main">
              <section className={`block answer ${passed ? "" : "withheld"}`}>
                <h3 className="label">Answer</h3>
                {passed ? (
                  <p className="answer-text">{renderAnswer(data.answer)}</p>
                ) : (
                  <>
                    <p className="answer-text">{data.answer}</p>
                    <p className="note">
                      The system did not release an answer because one or more verification checks failed.
                    </p>
                    {rawAnswer && (
                      <details className="draft">
                        <summary>Show withheld draft (unverified)</summary>
                        <p>{renderAnswer(rawAnswer)}</p>
                      </details>
                    )}
                  </>
                )}
              </section>

              {steps.length > 0 && result !== null && (
                <section className="block">
                  <h3 className="label">Calculation</h3>
                  <ol className="steps">
                    {steps.map((s, i) => (
                      <li key={i}>
                        <span className="op">{s.op}</span>
                        <span className="args num">({(s.args || []).map(fmtArg).join(", ")})</span>
                      </li>
                    ))}
                  </ol>
                  {isChangePattern && (
                    <p className="formula">
                      Formula as executed:{" "}
                      <span className="num">
                        ({fmt(steps[0].args[0])} − {fmt(steps[0].args[1])}) / {fmt(steps[1].args[1])} × 100
                      </span>
                    </p>
                  )}
                  <div className="result-row">
                    <span>Result</span>
                    <span className={`result num ${result < 0 ? "neg" : ""}`}>
                      {fmt(result, 2)}{isPercent ? "%" : ""}
                    </span>
                  </div>
                </section>
              )}

              <section className="block">
                <h3 className="label">Verification</h3>
                <p className={`verdict ${passed ? "ok" : "fail"}`}>
                  {passed ? "All checks passed" : "Not released: one or more checks failed"}
                </p>
                <ul className="checks">
                  {Object.entries(data.checks).map(([k, v]) => (
                    <li key={k} className={v ? "pass" : "fail"}>
                      <span className="mark" role="img" aria-label={v ? "Passed" : "Failed"}>{v ? "✓" : "✗"}</span>
                      <span>{CHECK_LABELS[k] || k.replace(/_/g, " ")}</span>
                    </li>
                  ))}
                </ul>
                <p className="meta-line">
                  Evidence coverage: <span className="num">{data.evidence_coverage}%</span>
                  {" "}of numbers in the draft trace to the evidence or the computed result.
                </p>
                <p className="meta-line">
                  Retrieval mode: {MODES.find(m => m[0] === mode)?.[1] || mode}.
                  These checks test grounding, not correctness.
                </p>
              </section>
            </div>

            <aside className="col-evidence">
              <h3 className="label">Evidence ({data.evidence.length})</h3>
              {data.evidence.map(e => {
                const meta = parseMeta(e.text)
                const isOpen = open === e.id
                const truncated = e.text.length >= 790
                return (
                  <article key={e.id} id={`ev-${e.id}`} className={`ev ${isOpen ? "open" : ""}`}>
                    <button className="ev-head" onClick={() => setOpen(isOpen ? null : e.id)} aria-expanded={isOpen}>
                      <span className="ev-id">{e.id}</span>
                      <span className="ev-type">{e.type}</span>
                      {citedIds.has(e.id) && <span className="ev-cited">cited</span>}
                      <span className="ev-meta">
                        {[meta.company, meta.year, meta.page && `p. ${meta.page}`].filter(Boolean).join(" · ")}
                      </span>
                    </button>
                    {isOpen && (
                      <div className="ev-body">
                        {e.type === "table"
                          ? <EvidenceTable body={meta.body} />
                          : <p className="excerpt">{meta.body}</p>}
                        {truncated && <p className="trunc">Excerpt truncated by the backend.</p>}
                      </div>
                    )}
                  </article>
                )
              })}
            </aside>
          </div>
        )}
      </main>
    </div>
  )
}