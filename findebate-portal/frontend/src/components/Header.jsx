import { Activity, Database, FlaskConical, Zap } from "lucide-react";

export default function Header({ mode, setMode, pipelineAvailable, disabled }) {
  return (
    <header className="header">
      <div className="brand">
        <div className="brand-mark"><Zap size={22} /></div>
        <div>
          <h1>FinDebate</h1>
          <span>Research Preview</span>
        </div>
      </div>
      <div className="mode-wrap">
        <div className="mode-copy">
          {mode === "precomputed" ? (
            <>
              <Database size={15} /> Instant results from paper dataset
            </>
          ) : (
            <>
              <Activity size={15} /> Full pipeline ~3-5 minutes
            </>
          )}
        </div>
        <div className="segmented">
          <button
            className={mode === "precomputed" ? "active precomputed" : ""}
            onClick={() => setMode("precomputed")}
            disabled={disabled}
          >
            Precomputed
          </button>
          <button
            className={mode === "live" ? "active live" : ""}
            onClick={() => pipelineAvailable && setMode("live")}
            disabled={disabled || !pipelineAvailable}
            title={pipelineAvailable ? "" : "Live mode needs ChromaDB, agent code, and API keys"}
          >
            <FlaskConical size={15} /> Live
          </button>
        </div>
      </div>
    </header>
  );
}
