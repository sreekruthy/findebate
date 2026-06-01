import { ArrowRight, Search } from "lucide-react";

export default function SearchBar({ mode, question, setQuestion, companies, onSubmit, pipelineAvailable, busy }) {
  if (mode === "precomputed") {
    return (
      <section className="search-panel">
        <div className="section-heading">
          <h2>Select a Company from the Paper Dataset</h2>
          <span>{companies.length} reports loaded</span>
        </div>
        <div className="company-grid">
          {companies.map((company) => (
            <button
              className="company-card"
              key={company.filename}
              onClick={() => onSubmit({ ticker: company.ticker })}
              disabled={busy}
            >
              <span className="ticker-badge">{company.ticker}</span>
              <strong>{company.company}</strong>
              <small>{company.quarter || company.label}</small>
            </button>
          ))}
        </div>
      </section>
    );
  }

  return (
    <section className="search-panel">
      <div className="section-heading">
        <h2>AI-Powered Financial Analysis</h2>
        <span>{pipelineAvailable ? "Live pipeline ready" : "Live pipeline unavailable"}</span>
      </div>
      <form
        className="query-form"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit({ question });
        }}
      >
        <div className="query-input">
          <Search size={20} />
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Enter a company name or earnings question"
            rows={3}
            disabled={busy || !pipelineAvailable}
          />
        </div>
        <button className="primary-action" disabled={busy || !pipelineAvailable || !question.trim()}>
          Analyze <ArrowRight size={18} />
        </button>
      </form>
      <div className="warning-banner">Live mode runs the full 5-stage pipeline. Estimated time: 3-5 minutes.</div>
    </section>
  );
}
