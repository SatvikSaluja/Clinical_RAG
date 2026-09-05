import React from "react";

function verdictClass(verdict) {
  if (verdict === "SUPPORTED") return "supported";
  if (verdict === "CONTRADICTED") return "contradicted";
  return "unsupported";
}

function ClaimRow({ claim }) {
  const cls = claim.verdict ? verdictClass(claim.verdict) : "unsupported";
  return (
    <div className={`claim-row ${cls}`}>
      <span className="marker">{claim.display_marker}</span>
      {claim.text}
      {claim.citation_ids.length > 0 && (
        <span style={{ marginLeft: 6 }}>
          {claim.citation_ids.map((c) => (
            <span key={c} className="citation-chip">{c}</span>
          ))}
        </span>
      )}
    </div>
  );
}

export default function AnswerPanel({ result, loading }) {
  if (loading) {
    return (
      <div className="panel">
        <h2>Answer</h2>
        <div className="loading-note">
          Running retrieval &rarr; reranking &rarr; generation &rarr; verification...
          This can take up to a minute or two with a local CPU model.
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="panel">
        <h2>Answer</h2>
        <div className="empty-state">No answer yet.</div>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>Answer</h2>
      <div className="answer-text">{result.final_answer}</div>

      <h3 style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 16 }}>
        Claim verification ({result.claims.length} claims)
      </h3>
      {result.claims.length === 0 && (
        <div className="empty-state">No factual claims were extracted from this answer.</div>
      )}
      {result.claims.map((c, i) => <ClaimRow key={i} claim={c} />)}

      <h3 style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 16 }}>
        Patient-context fields used
      </h3>
      {result.patient_context_fields.length === 0 && (
        <div className="empty-state">None selected as relevant.</div>
      )}
      {result.patient_context_fields.map((f, i) => (
        <div className="field-row" key={i}>
          <span>{f.text}</span>
          <span style={{ color: "var(--text-dim)" }}>{f.similarity.toFixed(2)}</span>
        </div>
      ))}

      <div style={{ marginTop: 14, fontSize: 11, color: "var(--text-dim)" }}>
        Provider: {result.llm_provider} ({result.llm_model}) · Reliability gate: {result.reliability_gate_mode}
        <br />
        Latency - retrieval: {result.latency_retrieval_s.toFixed(2)}s · generation:{" "}
        {result.latency_generation_s.toFixed(2)}s · verification: {result.latency_verification_s.toFixed(2)}s
      </div>

      <div style={{ marginTop: 10, fontSize: 11, color: "var(--warn)" }}>{result.disclaimer}</div>
    </div>
  );
}
