import React from "react";

function LabRow({ lab }) {
  return (
    <div className="field-row">
      <span>{lab.test}: {lab.value} {lab.unit}</span>
      <span className={`status-badge status-${lab.status}`}>{lab.status}</span>
    </div>
  );
}

export default function PatientPanel({ patients, selectedId, onSelect, patient }) {
  return (
    <div className="panel">
      <h2>Patient</h2>
      <select
        className="patient-select"
        value={selectedId || ""}
        onChange={(e) => onSelect(e.target.value)}
      >
        <option value="" disabled>Select a synthetic patient...</option>
        {patients.map((p) => (
          <option key={p.patient_id} value={p.patient_id}>{p.name} ({p.age}{p.sex[0]})</option>
        ))}
      </select>

      {!patient && <div className="empty-state">No patient selected.</div>}

      {patient && (
        <>
          <div className="field-group">
            <h3>Laboratory Results ({patient.labs.length})</h3>
            {patient.labs.map((lab, i) => <LabRow key={i} lab={lab} />)}
          </div>

          <div className="field-group">
            <h3>Medications ({patient.medications.length})</h3>
            {patient.medications.map((m, i) => (
              <div className="field-row" key={i}>
                <span>{m.name} - {m.dose}, {m.frequency}</span>
              </div>
            ))}
          </div>

          <div className="field-group">
            <h3>Medical History ({patient.history.length})</h3>
            {patient.history.map((h, i) => (
              <div className="field-row" key={i}>
                <span>{h.date}: {h.event}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
