import { useState } from 'react'
import type { Chunk } from '../types'

type Props = { chunks: Chunk[]; goldIds?: string[]; color: string; scoreLabel: string }

export function ChunkList({ chunks, goldIds, color, scoreLabel }: Props) {
  const [open, setOpen] = useState<string | null>(null)
  if (!chunks.length) return <p className="empty">No paragraphs retrieved.</p>
  const max = Math.max(...chunks.map((c) => c.score), 1e-6)
  return (
    <div className="chunks">
      {chunks.map((c, i) => (
        <div className="chunk" key={c.id}>
          <div className="chunk-head">
            <span className="mono id">#{i + 1}</span>
            <span className="title" title={c.title}>{c.title}</span>
            {goldIds?.includes(c.id) && <span className="gold-tag" title="Gold supporting paragraph">GOLD</span>}
            <span className="mono id" title={scoreLabel}>{c.score.toFixed(3)}</span>
          </div>
          <div className="scorebar"><span style={{ width: `${(c.score / max) * 100}%`, background: color }} /></div>
          <p className={open === c.id ? 'open' : ''} onClick={() => setOpen(open === c.id ? null : c.id)}>
            {c.text}
          </p>
        </div>
      ))}
    </div>
  )
}
