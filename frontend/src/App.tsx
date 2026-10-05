import { useState } from "react"
import { ask, type AskResponse } from "./api"
import "./App.css"

export default function App() {
  const [question, setQuestion] = useState("")
  const [mode, setMode] = useState("dense")
  const [data, setData] = useState<AskResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [open, setOpen] = useState<string | null>(null)

  async function submit() {
    if (!question.trim()) return
    setLoading(true); setError(""); setData(null)
    try { setData(await ask(question, mode)) }
    catch (e) { setError((e as Error).message) }
    finally { setLoading(false) }
  }

  return (
    <div className="page">
      <h1>FinRAG-X</h1>
      <textarea value={question} onChange={e => setQuestion(e.target.value)}
                placeholder="Ask a question about a financial report..." />
      <div className="row">
        <select value={mode} onChange={e => setMode(e.target.value)}>
          <option value="dense">Dense only</option>
          <option value="bm25">BM25 only</option>
          <option value="hybrid">Hybrid</option>
          <option value="hybrid_rerank">Hybrid + rerank</option>
        </select>
        <button onClick={submit} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {error && <div className="card err">{error}</div>}

      {data && (<>
        <div className={`card ${data.passed ? "ok" : "warn"}`}>
          <h2>Answer</h2>
          <p>{data.answer}</p>
          <span className="badge">Evidence Coverage: {data.evidence_coverage}%</span>
        </div>

        {data.program && data.program.op !== "none" && (
          <div className="card">
            <h2>Calculation</h2>
            <p>Operation: <b>{data.program.op}</b></p>
            <p>{data.program.a_label || "a"} = {data.program.a} | {data.program.b_label || "b"} = {data.program.b}</p>
            <p>Result: <b>{data.result}</b> {data.program.unit}</p>
          </div>
        )}

        <div className="card">
          <h2>Checks</h2>
          {Object.entries(data.checks).map(([k, v]) => (
            <div key={k} className={v ? "pass" : "fail"}>
              {v ? "✔" : "✘"} {k.replace(/_/g, " ")}
            </div>
          ))}
        </div>

        <div className="card">
          <h2>Sources</h2>
          {data.evidence.map(e => (
            <div key={e.id}>
              <button className="link" onClick={() => setOpen(open === e.id ? null : e.id)}>
                [{e.id}] ({e.type})
              </button>
              {open === e.id && <pre className="evidence">{e.text}</pre>}
            </div>
          ))}
        </div>
      </>)}
    </div>
  )
}