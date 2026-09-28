import { useEffect, useState } from 'react'
import { getDemoQuestions, runQuery } from './api'
import { AnswerCard } from './components/AnswerCard'
import { ChunkList } from './components/ChunkList'
import { EvalPage } from './components/EvalPage'
import { GraphView } from './components/GraphView'
import type { DemoQuestion, QueryResponse } from './types'

type Tab = 'compare' | 'eval'

export default function App() {
  const [tab, setTab] = useState<Tab>('compare')
  const [demos, setDemos] = useState<DemoQuestion[]>([])
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { getDemoQuestions().then(setDemos).catch(() => setDemos([])) }, [])

  useEffect(() => {
    if (!loading) return
    const t0 = Date.now()
    const id = setInterval(() => setElapsed((Date.now() - t0) / 1000), 200)
    return () => clearInterval(id)
  }, [loading])

  async function ask(q: string) {
    if (!q.trim() || loading) return
    setQuestion(q)
    setTab('compare')
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await runQuery(q))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const gold = result?.gold

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>GraphRAG vs Vector RAG</h1>
          <p>Same question, same local model, same 5-paragraph budget. Only the retrieval differs.</p>
        </div>
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'compare'} onClick={() => setTab('compare')}>Compare</button>
          <button role="tab" aria-selected={tab === 'eval'} onClick={() => setTab('eval')}>Evaluation</button>
        </div>
      </header>

      {tab === 'eval' ? (
        <EvalPage onPick={ask} />
      ) : (
        <>
          <form className="query" onSubmit={(e) => { e.preventDefault(); ask(question) }}>
            <div className="query-row">
              <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a multi-hop question…" />
              <button type="submit" disabled={loading || !question.trim()}>{loading ? 'Running…' : 'Run both'}</button>
            </div>
            {demos.length > 0 && (
              <select value="" onChange={(e) => e.target.value && ask(e.target.value)}>
                <option value="">Try a demo question from MuSiQue…</option>
                {demos.map((d) => <option key={d.question} value={d.question}>[{d.type}] {d.question}</option>)}
              </select>
            )}
            {gold && <div className="gold-line">Gold answer: <b>{gold.answer}</b> · {gold.type} question · {gold.supporting_ids.length} supporting paragraphs</div>}
          </form>

          {error && <div className="error">{error}</div>}
          {loading && <div className="loading">Running both pipelines on the local model… {elapsed.toFixed(0)}s</div>}

          {result && (
            <div className="columns">
              <section className="column">
                <div className="col-head"><i className="swatch" style={{ background: 'var(--vector)' }} /> Vector RAG <span className="sub">top-5 by cosine similarity</span></div>
                <AnswerCard {...result.vector} gold={gold} />
                <div className="card">
                  <h3>Retrieved paragraphs</h3>
                  <ChunkList chunks={result.vector.chunks} goldIds={gold?.supporting_ids} color="var(--vector)" scoreLabel="cosine similarity" />
                </div>
              </section>
              <section className="column">
                <div className="col-head"><i className="swatch" style={{ background: 'var(--graph)' }} /> GraphRAG <span className="sub">entity linking → beam traversal</span></div>
                <AnswerCard {...result.graph} gold={gold} />
                <GraphView result={result.graph} />
                <div className="card">
                  <h3>Graph-selected paragraphs</h3>
                  <ChunkList chunks={result.graph.chunks} goldIds={gold?.supporting_ids} color="var(--graph)" scoreLabel="best edge score" />
                </div>
              </section>
            </div>
          )}
        </>
      )}
    </div>
  )
}
