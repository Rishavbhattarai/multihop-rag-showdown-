import cytoscape from 'cytoscape'
import fcose from 'cytoscape-fcose'
import { useEffect, useRef, useState } from 'react'
import type { GraphResult } from '../types'

cytoscape.use(fcose)

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9 ]/g, '').replace(/\b(the|a|an)\b/g, '').replace(/\s+/g, ' ').trim()

const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

/** Node that best matches the predicted answer: exact normalized match, else containment. */
function answerNode(cy: cytoscape.Core, answer: string) {
  const a = norm(answer)
  if (!a || a === 'unknown') return null
  const nodes = cy.nodes().toArray()
  return (
    nodes.find((n) => norm(n.id()) === a) ??
    nodes.find((n) => norm(n.id()).includes(a) || (a.length > 3 && a.includes(norm(n.id())))) ??
    null
  )
}

/** Shortest undirected path from any seed to the answer node, rendered with true edge directions. */
function findPath(cy: cytoscape.Core, answer: string) {
  const goal = answerNode(cy, answer)
  if (!goal) return null
  let best: cytoscape.SearchAStarResult | null = null
  cy.nodes('[?seed]').forEach((seed) => {
    const r = cy.elements().aStar({ root: seed, goal, directed: false })
    if (r.found && (!best || r.distance < best.distance)) best = r
  })
  if (!best) return null
  const path = (best as cytoscape.SearchAStarResult).path
  let text = `(${path[0].id()})`
  for (let i = 1; i < path.length; i += 2) {
    const edge = path[i]
    const next = path[i + 1]
    const forward = edge.data('target') === next.id()
    text += forward ? `-[${edge.data('rel')}]->(${next.id()})` : `<-[${edge.data('rel')}]-(${next.id()})`
  }
  return { elements: path, text }
}

export function GraphView({ result }: { result: GraphResult }) {
  const ref = useRef<HTMLDivElement>(null)
  const [pathText, setPathText] = useState<string | null>(null)

  useEffect(() => {
    if (!ref.current) return
    const graph = cssVar('--graph')
    const surface0 = cssVar('--surface-0')
    const surface2 = cssVar('--surface-2')
    const border = cssVar('--text-muted')
    const text = cssVar('--text-primary')
    const muted = cssVar('--text-secondary')

    const cy = cytoscape({
      container: ref.current,
      elements: [
        ...result.subgraph.nodes.map((n) => ({ data: { id: n.id, hop: n.hop, seed: n.seed } })),
        ...result.subgraph.edges.map((e, i) => ({ data: { id: `e${i}`, ...e } })),
      ],
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(id)', 'font-size': 11, color: text, 'font-family': 'Inter, sans-serif',
            'text-valign': 'bottom', 'text-margin-y': 4, 'text-wrap': 'wrap', 'text-max-width': '110px',
            'text-background-color': surface0, 'text-background-opacity': 0.85, 'text-background-padding': '2px',
            width: 14, height: 14, 'background-color': surface2, 'border-width': 1.5, 'border-color': border,
          },
        },
        { selector: 'node[?seed]', style: { 'background-color': graph, 'border-color': graph, width: 22, height: 22, 'font-weight': 700 } },
        {
          selector: 'edge',
          style: {
            width: 1.2, 'line-color': border, 'target-arrow-color': border, 'target-arrow-shape': 'triangle',
            'arrow-scale': 0.7, 'curve-style': 'bezier', 'font-size': 9, color: muted,
            'text-rotation': 'autorotate', 'text-background-color': surface0, 'text-background-opacity': 0.9,
            'text-background-padding': '1px', opacity: 0.7,
          },
        },
        // relation labels only where they're read: the answer path and the hovered neighbourhood
        { selector: 'edge.path, edge.hovered', style: { label: 'data(rel)', opacity: 1 } },
        { selector: 'edge.path', style: { 'line-color': graph, 'target-arrow-color': graph, width: 3, 'z-index': 10, color: text, 'font-weight': 600 } },
        { selector: 'node.path', style: { 'border-color': graph, 'border-width': 3 } },
        { selector: '.faded', style: { opacity: 0.2 } },
      ],
      layout: {
        name: 'fcose', animate: false, randomize: true, nodeRepulsion: () => 20000,
        idealEdgeLength: () => 140, nodeSeparation: 120, padding: 30, numIter: 5000,
      } as cytoscape.LayoutOptions,
      minZoom: 0.3,
      maxZoom: 2.5,
    })

    const found = findPath(cy, result.answer)
    found?.elements.addClass('path')
    setPathText(found?.text ?? null)

    // hover a node: fade everything outside its neighbourhood
    cy.on('mouseover', 'node', (evt) => {
      cy.elements().addClass('faded')
      evt.target.closedNeighborhood().removeClass('faded')
      evt.target.connectedEdges().addClass('hovered')
    })
    cy.on('mouseout', 'node', () => cy.elements().removeClass('faded hovered'))

    return () => cy.destroy()
  }, [result])

  return (
    <div className="card">
      <h3>Traversed subgraph</h3>
      <div className="seeds">
        Seeds:
        {result.seeds.length ? (
          result.seeds.map((s) => <span className="seed-chip" key={s.node} title={`from “${s.mention}”`}>{s.node}</span>)
        ) : (
          <span className="empty">no question entities matched the graph</span>
        )}
      </div>
      {result.subgraph.nodes.length ? <div className="graph-canvas" ref={ref} /> : <p className="empty">Empty subgraph.</p>}
      <div className="legend">
        <span><i className="dot" style={{ background: 'var(--graph)' }} /> seed entity</span>
        <span><i className="dot" style={{ background: 'var(--surface-2)', border: '1.5px solid var(--text-muted)' }} /> reached by traversal</span>
        <span><i className="dot" style={{ background: 'transparent', border: '3px solid var(--graph)' }} /> path to answer</span>
      </div>
      <div className="path">
        {pathText ? <>Answer path: <code>{pathText}</code></> : 'No graph path to the answer node (answer came from paragraph text).'}
      </div>
    </div>
  )
}
