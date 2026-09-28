export type Chunk = { id: string; title: string; text: string; score: number }

export type Usage = {
  prompt_tokens: number
  completion_tokens: number
  latency_s: number
  retrieval_s: number
  total_s: number
}

export type Grade = { em: number; f1: number; judge: boolean; support_recall: number }

export type VectorResult = {
  pipeline: 'vector'
  answer: string
  reasoning: string
  chunks: Chunk[]
  retrieved_ids: string[]
  usage: Usage
  grade?: Grade
}

export type GraphNode = { id: string; hop: number; seed: boolean }
export type GraphEdge = { source: string; target: string; rel: string; doc: string; score: number; hop: number }

export type GraphResult = Omit<VectorResult, 'pipeline'> & {
  pipeline: 'graph'
  mentions: string[]
  seeds: { node: string; mention: string }[]
  triples: string[]
  subgraph: { nodes: GraphNode[]; edges: GraphEdge[] }
}

export type Gold = { answer: string; aliases: string[]; supporting_ids: string[]; type: string }

export type QueryResponse = { vector: VectorResult; graph: GraphResult; gold?: Gold }

export type DemoQuestion = { question: string; type: string; answer: string }

export type Metrics = {
  n: number
  em: number
  f1: number
  judge_acc: number
  support_recall: number
  full_support: number
  latency_p50_s: number
  latency_p95_s: number
  tokens_per_query: number
}

export type EvalRun = {
  qid: string
  pipeline: string
  type: string
  split: string
  question: string
  gold: string
  pred: string
  judge: boolean
  support_recall: number
}

export type EvalResults = {
  config: { llm: string; embed: string; top_k: number; max_hops: number; dataset: string; n_eval: number }
  summary: Record<string, Record<string, Metrics>>
  head_to_head: Record<'vector_only' | 'graph_only' | 'both' | 'neither', number>
  wins: Record<'vector_only' | 'graph_only' | 'both' | 'neither', string[]>
  runs: EvalRun[]
}
