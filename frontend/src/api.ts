import type { DemoQuestion, EvalResults, QueryResponse } from './types'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export const getDemoQuestions = () => fetch('/api/demo-questions').then(json<DemoQuestion[]>)

export const getEval = () => fetch('/api/eval').then(json<EvalResults>)

export const runQuery = (question: string) =>
  fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  }).then(json<QueryResponse>)
