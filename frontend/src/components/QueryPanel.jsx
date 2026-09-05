import React, { useState } from "react";

const EXAMPLE_QUESTIONS = [
  "What could explain my recent HbA1c result?",
  "What are common side effects of my medications?",
  "How does my kidney function relate to my other conditions?",
  "What do my thyroid labs mean?",
];

export default function QueryPanel({ onSubmit, loading, disabled }) {
  const [question, setQuestion] = useState("");

  return (
    <div className="panel">
      <h2>Ask a Question</h2>
      <textarea
        className="question-input"
        placeholder="e.g. What factors could explain my recent glucose levels?"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />
      <div>
        <button
          className="run-button"
          disabled={disabled || loading || !question.trim()}
          onClick={() => onSubmit(question.trim())}
        >
          {loading ? "Running pipeline..." : "Ask"}
        </button>
      </div>

      <div style={{ marginTop: 14 }}>
        <h3 style={{ fontSize: 12, color: "var(--text-dim)" }}>Example questions</h3>
        {EXAMPLE_QUESTIONS.map((q) => (
          <div key={q} className="example-q" onClick={() => setQuestion(q)}>{q}</div>
        ))}
      </div>
    </div>
  );
}
