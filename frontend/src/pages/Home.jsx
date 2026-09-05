import React, { useEffect, useState } from "react";
import { listPatients, getPatient, runQuery } from "../api/client.js";
import PatientPanel from "../components/PatientPanel.jsx";
import QueryPanel from "../components/QueryPanel.jsx";
import RetrievalPanel from "../components/RetrievalPanel.jsx";
import AnswerPanel from "../components/AnswerPanel.jsx";

export default function Home() {
  const [patients, setPatients] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [patient, setPatient] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    listPatients().then((list) => {
      setPatients(list);
      if (list.length > 0) setSelectedId(list[0].patient_id);
    });
  }, []);

  useEffect(() => {
    if (selectedId) {
      getPatient(selectedId).then(setPatient);
      setResult(null);
    }
  }, [selectedId]);

  async function handleSubmit(question) {
    setLoading(true);
    setError(null);
    try {
      const res = await runQuery(selectedId, question);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="main-grid">
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <PatientPanel
          patients={patients}
          selectedId={selectedId}
          onSelect={setSelectedId}
          patient={patient}
        />
        <QueryPanel onSubmit={handleSubmit} loading={loading} disabled={!selectedId} />
        {error && <div className="panel"><div className="empty-state" style={{ color: "var(--bad)" }}>{error}</div></div>}
      </div>

      <RetrievalPanel result={result} />
      <AnswerPanel result={result} loading={loading} />
    </div>
  );
}
