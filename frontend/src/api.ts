export interface Evidence { id: string; type: string; text: string }
export interface Program {
  op: string; a?: number; b?: number
  a_label?: string; b_label?: string; unit?: string
}
export interface AskResponse {
  question: string; answer: string; passed: boolean
  program: Program | null; result: number | null
  evidence: Evidence[]; checks: Record<string, boolean>
  evidence_coverage: number
}

export async function ask(question: string, mode: string): Promise<AskResponse> {
  const res = await fetch("http://localhost:8000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode }),
  })
  if (!res.ok) throw new Error(`Server error ${res.status}`)
  return res.json()
}