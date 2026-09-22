import React, { useState } from "react";

const EXAMPLE_QUESTIONS = [
  "What could explain my recent HbA1c result?",
  "What are common side effects of my medications?",
  "How does my kidney function relate to my other conditions?",
  "What do my thyroid labs mean?",
];

export default function QueryPanel({ onSubmit, loading, disabled }) {
  const [question, setQuestion] = useState("");

  function submit() {
    if (!disabled && !loading && question.trim()) onSubmit(question.trim());
  }

  return (
    <div className="query-bar">
      <div className="query-bar-row">
        <textarea
          className="question-input"
          placeholder="Ask about your labs, medications, or history — e.g. “What factors could explain my recent glucose levels?”"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={1}
        />
        <button className="run-button" disabled={disabled || loading || !question.trim()} onClick={submit}>
          {loading ? "Running…" : "Ask"}
        </button>
      </div>
      <div className="example-q-row">
        {EXAMPLE_QUESTIONS.map((q) => (
          <button key={q} className="example-chip" onClick={() => setQuestion(q)}>{q}</button>
        ))}
      </div>
    </div>
  );
}
