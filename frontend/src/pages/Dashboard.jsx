import React, { useEffect, useState } from "react";
import { getDashboard } from "../api/client.js";

const TARGETS = {
  recall_at_5: 0.81,
  citation_accuracy: 0.85,
  supported_claim_rate: 0.82,
  hallucination_rate: 0.13, // ceiling, not floor
};

function pct(v) {
  if (v === null || v === undefined) return "N/A";
  return `${(v * 100).toFixed(1)}%`;
}

function MetricTile({ label, value, target, isCeiling }) {
  return (
    <div className="metric-tile">
      <div className="value">{pct(value)}</div>
      <div className="label">{label}</div>
      {target != null && (
        <div className="target">
          Target: {isCeiling ? "≤" : ""}{pct(target)}
        </div>
      )}
    </div>
  );
}

function ResultsTable({ title, table }) {
  const rows = Object.entries(table || {});
  if (rows.length === 0) return null;
  return (
    <>
      <h3>{title}</h3>
      <div style={{ overflowX: "auto" }}>
        <table className="results-table">
          <thead>
            <tr>
              <th>System</th>
              <th>Recall@5</th>
              <th>Recall@10</th>
              <th>Citation Acc.</th>
              <th>Supported Claims</th>
              <th>Hallucination</th>
              <th>Queries</th>
              <th>Avg Gen. Latency (s)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, m]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{pct(m.recall_at_5)}</td>
                <td>{pct(m.recall_at_10)}</td>
                <td>{pct(m.citation_accuracy)}</td>
                <td>{pct(m.supported_claim_rate)}</td>
                <td>{pct(m.hallucination_rate)}</td>
                <td>{m.num_queries}</td>
                <td>{m.avg_latency_generation_s?.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getDashboard().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="panel"><div className="empty-state">Error loading dashboard: {error}</div></div>;
  if (!data) return <div className="panel"><div className="loading-note">Loading dashboard...</div></div>;

  const full = data.results?.full_system?.summary;
  const baselines = data.results?.baselines_summary;
  const ablation = data.results?.ablation_summary;

  return (
    <div style={{ padding: "20px" }}>
      <div className="panel" style={{ marginBottom: 20 }}>
        <h2>Research Dashboard - Final System</h2>
        {!full && (
          <div className="empty-state">
            No evaluation results yet. Run <code>python -m backend.evaluation.baselines</code> and{" "}
            <code>python -m backend.evaluation.ablation</code> from the backend to populate this dashboard.
          </div>
        )}
        {full && (
          <div className="metric-grid">
            <MetricTile label="Recall@5" value={full.recall_at_5} target={TARGETS.recall_at_5} />
            <MetricTile label="Recall@10" value={full.recall_at_10} />
            <MetricTile label="Citation Accuracy" value={full.citation_accuracy} target={TARGETS.citation_accuracy} />
            <MetricTile label="Supported Claim Rate" value={full.supported_claim_rate} target={TARGETS.supported_claim_rate} />
            <MetricTile label="Hallucination Rate" value={full.hallucination_rate} target={TARGETS.hallucination_rate} isCeiling />
            <div className="metric-tile">
              <div className="value">{full.num_queries}</div>
              <div className="label">Evaluated Queries</div>
            </div>
            <div className="metric-tile">
              <div className="value">{full.avg_latency_retrieval_s?.toFixed(2)}s</div>
              <div className="label">Avg Retrieval Latency</div>
            </div>
            <div className="metric-tile">
              <div className="value">{full.avg_latency_generation_s?.toFixed(2)}s</div>
              <div className="label">Avg Generation Latency</div>
            </div>
            <div className="metric-tile">
              <div className="value">{full.avg_latency_verification_s?.toFixed(2)}s</div>
              <div className="label">Avg Verification Latency</div>
            </div>
          </div>
        )}
      </div>

      <div className="panel">
        <ResultsTable title="Baseline Comparison" table={baselines} />
        <ResultsTable title="Ablation Study" table={ablation} />
        {!baselines && !ablation && (
          <div className="empty-state">No baseline/ablation results yet - see RESULTS.md for how to generate them.</div>
        )}
      </div>
    </div>
  );
}
