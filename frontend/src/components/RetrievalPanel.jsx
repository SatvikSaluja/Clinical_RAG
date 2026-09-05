import React, { useState } from "react";

function EvidenceItem({ ev, index }) {
  return (
    <div className="evidence-item">
      <div className="ev-title">{index + 1}. {ev.title}</div>
      <div className="ev-meta">
        {ev.section} {ev.publication_year ? `· ${ev.publication_year}` : ""}
        {ev.pmid ? ` · PMID: ${ev.pmid}` : ""}
        {ev.citation_id ? ` · [${ev.citation_id}]` : ""}
      </div>
      <div className="ev-text">{ev.text}</div>
      <div className="ev-scores">
        {ev.bm25_score != null && <>BM25: {ev.bm25_score.toFixed(2)} </>}
        {ev.dense_score != null && <>Dense: {ev.dense_score.toFixed(3)} </>}
        {ev.hybrid_score != null && <>Hybrid: {ev.hybrid_score.toFixed(3)} </>}
        {ev.rerank_score != null && <>Rerank: {ev.rerank_score.toFixed(3)}</>}
      </div>
    </div>
  );
}

export default function RetrievalPanel({ result }) {
  const [tab, setTab] = useState("reranked");

  if (!result) {
    return (
      <div className="panel">
        <h2>Retrieved Evidence</h2>
        <div className="empty-state">Ask a question to see retrieval results.</div>
      </div>
    );
  }

  const tabs = {
    reranked: result.reranked,
    fused: result.fused,
    bm25: result.retrieved_bm25,
    dense: result.retrieved_dense,
  };

  return (
    <div className="panel">
      <h2>Retrieved Evidence</h2>
      <div className="tabs-mini">
        {Object.keys(tabs).map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t} ({tabs[t].length})
          </button>
        ))}
      </div>
      {tabs[tab].length === 0 && <div className="empty-state">No results for this stage.</div>}
      {tabs[tab].map((ev, i) => <EvidenceItem key={ev.chunk_id + i} ev={ev} index={i} />)}
    </div>
  );
}
