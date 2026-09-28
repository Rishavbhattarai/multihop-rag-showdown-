import { useEffect, useState } from 'react'
import { getEval } from '../api'
import type { EvalResults, Metrics } from '../types'

const GROUPS = [['all', 'All'], ['2hop', '2-hop'], ['3hop', '3-hop'], ['4hop', '4-hop']] as const
// categorical slots 1-4 in fixed order; ablations run the same 5-paragraph budget without the triple list
const PIPES = [
  ['vector', 'Vector RAG', 'var(--vector)'],
  ['graph', 'GraphRAG', 'var(--graph)'],
  ['graph_paras', 'Graph paragraphs only', 'var(--graph-paras)'],
  ['hybrid', 'Hybrid (vector + graph)', 'var(--hybrid)'],
] as const
const pct = (x: number) => `${Math.round(x * 100)}%`

const ROWS: [keyof Metrics, string, (x: number) => string, boolean][] = [
  ['judge_acc', 'Accuracy', pct, true],
  ['em', 'EM', pct, true],
  ['f1', 'F1', (x) => x.toFixed(2), true],
  ['support_recall', 'Support recall', pct, true],
  ['full_support', 'All gold found', pct, true],
  ['latency_p50_s', 'Latency p50', (x) => `${x.toFixed(1)}s`, false],
  ['latency_p95_s', 'Latency p95', (x) => `${x.toFixed(1)}s`, false],
  ['tokens_per_query', 'Tokens / query', (x) => x.toLocaleString(), false],
]

function AccuracyChart({ res, pipes }: { res: EvalResults; pipes: readonly (typeof PIPES)[number][] }) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null)
  return (
    <div className="chart" onMouseLeave={() => setTip(null)}>
      <div className="bars">
        {GROUPS.map(([g]) => (
          <div className="bar-group" key={g}>
            {pipes.map(([p, label, color]) => {
              const m = res.summary[p]?.[g]
              if (!m) return null
              return (
                <div
                  key={p}
                  className="bar"
                  style={{ height: `${m.judge_acc * 100}%`, background: color }}
                  onMouseMove={(e) => {
                    const box = (e.currentTarget.closest('.chart') as HTMLElement).getBoundingClientRect()
                    setTip({ x: e.clientX - box.left, y: e.clientY - box.top, text: `${label}: ${pct(m.judge_acc)} correct (${Math.round(m.judge_acc * m.n)}/${m.n})` })
                  }}
                >
                  <span className="val">{pct(m.judge_acc)}</span>
                </div>
              )
            })}
          </div>
        ))}
      </div>
      <div className="bar-labels">
        {GROUPS.map(([g, label]) => <div key={g}>{label} <span className="mono">n={res.summary.vector?.[g]?.n ?? 0}</span></div>)}
      </div>
      {tip && <div className="tooltip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>}
    </div>
  )
}

export function EvalPage({ onPick }: { onPick: (q: string) => void }) {
  const [res, setRes] = useState<EvalResults | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { getEval().then(setRes).catch((e) => setErr(String(e.message ?? e))) }, [])

  if (err) return <div className="error">{err}</div>
  if (!res) return <div className="loading">Loading evaluation…</div>

  const pipes = PIPES.filter(([p]) => res.summary[p])
  const byQid = new Map<string, { question: string; gold: string; preds: Record<string, string> }>()
  for (const r of res.runs) {
    const row = byQid.get(r.qid) ?? { question: r.question, gold: r.gold, preds: {} }
    row.preds[r.pipeline] = r.pred
    byQid.set(r.qid, row)
  }
  const h = res.head_to_head
  const winList = (ids: string[]) =>
    ids.length ? ids.map((id) => {
      const r = byQid.get(id)!
      return (
        <div className="win-item" key={id}>
          <button onClick={() => onPick(r.question)}>
            <div className="q">{r.question}</div>
            <div className="a">gold <b>{r.gold}</b> · vector “{r.preds.vector}” · graph “{r.preds.graph}”</div>
          </button>
        </div>
      )
    }) : <p className="empty">None.</p>

  return (
    <div>
      <div className="tiles">
        <div className="tile"><div className="num">{h.graph_only}</div><div className="lbl">only GraphRAG correct</div></div>
        <div className="tile"><div className="num">{h.vector_only}</div><div className="lbl">only Vector RAG correct</div></div>
        <div className="tile"><div className="num">{h.both}</div><div className="lbl">both correct</div></div>
        <div className="tile"><div className="num">{h.neither}</div><div className="lbl">neither correct</div></div>
      </div>

      <div className="card section">
        <h3>Accuracy by question hops</h3>
        <div className="legend" style={{ marginBottom: 28 }}>
          {pipes.map(([p, label, color]) => <span key={p}><i className="dot" style={{ background: color, borderRadius: 3 }} /> {label}</span>)}
        </div>
        <AccuracyChart res={res} pipes={pipes} />
      </div>

      <div className="card section">
        <h3>All metrics · {res.config.n_eval} held-out questions</h3>
        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr><th>Configuration</th>{ROWS.map(([key, label]) => <th key={key}>{label}</th>)}</tr>
            </thead>
            <tbody>
              {pipes.map(([p, label, color]) => (
                <tr key={p}>
                  <td><i className="dot" style={{ background: color, borderRadius: 3, marginRight: 6 }} />{label}</td>
                  {ROWS.map(([key, , fmt, higherBetter]) => {
                    const vals = pipes.map(([q]) => res.summary[q].all[key])
                    const best = higherBetter ? Math.max(...vals) : Math.min(...vals)
                    const x = res.summary[p].all[key]
                    return <td key={key} className={x === best ? 'win' : ''}>{fmt(x)}</td>
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <h3 style={{ marginTop: 20 }}>Accuracy by hops (table view of the chart)</h3>
        <table>
          <thead><tr><th>Configuration</th>{GROUPS.map(([g, l]) => <th key={g}>{l}</th>)}</tr></thead>
          <tbody>
            {pipes.map(([p, label]) => (
              <tr key={p}>
                <td>{label}</td>
                {GROUPS.map(([g]) => {
                  const m = res.summary[p][g]
                  return <td key={g}>{m ? `${pct(m.judge_acc)} (${Math.round(m.judge_acc * m.n)}/${m.n})` : '–'}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note">
          {res.config.dataset} · {res.config.llm} + {res.config.embed} via Ollama · top-{res.config.top_k} paragraphs for every
          configuration · up to {res.config.max_hops} hops · bold = best. Accuracy is graded by the same local model (EM first,
          LLM judge only when EM fails). With ~50 questions, gaps under ~10 points are within noise.
        </p>
      </div>

      <div className="wins">
        <div className="card"><h3>Where GraphRAG wins</h3>{winList(res.wins.graph_only)}</div>
        <div className="card"><h3>Where Vector RAG wins</h3>{winList(res.wins.vector_only)}</div>
      </div>
    </div>
  )
}
