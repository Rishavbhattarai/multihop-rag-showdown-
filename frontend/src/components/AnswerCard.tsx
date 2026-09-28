import type { Gold, Grade, Usage } from '../types'

type Props = { answer: string; reasoning: string; usage: Usage; grade?: Grade; gold?: Gold }

export function AnswerCard({ answer, reasoning, usage, grade, gold }: Props) {
  return (
    <div className="card">
      <h3>Answer</h3>
      <div className="answer-text">{answer}</div>
      {grade && gold && (
        <div className="badges">
          <span className={`badge ${grade.judge ? 'good' : 'bad'}`}>{grade.judge ? '✓ Correct' : '✗ Incorrect'}</span>
          <span className="badge">EM {grade.em.toFixed(0)}</span>
          <span className="badge">F1 {grade.f1.toFixed(2)}</span>
          <span className="badge" title="Share of the gold supporting paragraphs this pipeline retrieved">
            Support {Math.round(grade.support_recall * gold.supporting_ids.length)}/{gold.supporting_ids.length}
          </span>
        </div>
      )}
      <div className="stats">
        <span><b>{usage.total_s.toFixed(1)}s</b> total</span>
        <span><b>{usage.retrieval_s.toFixed(2)}s</b> retrieval</span>
        <span><b>{(usage.prompt_tokens + usage.completion_tokens).toLocaleString()}</b> tokens</span>
        <span><b>$0</b> local</span>
      </div>
      <details>
        <summary>Model reasoning</summary>
        <pre>{reasoning}</pre>
      </details>
    </div>
  )
}
