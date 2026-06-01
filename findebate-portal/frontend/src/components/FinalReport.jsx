import { Copy, Download } from "lucide-react";

function textOf(value) {
  if (!value) return "";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.join("\n");
  return Object.entries(value).map(([key, val]) => `${key}: ${textOf(val)}`).join("\n");
}

function convictionValue(value = "") {
  const match = String(value).match(/\d+/);
  return match ? Number(match[0]) : 75;
}

export default function FinalReport({ result }) {
  const recs = result.investment_recommendations || {};
  const reportText = [
    result.executive_summary,
    textOf(recs),
    result.risk_reward,
    result.conclusion,
  ].join("\n\n");

  const download = () => {
    const blob = new Blob([reportText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${result.source_file || result.ticker}_report.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="final-report">
      <header>
        <h2>Final Investment Report</h2>
        <div className="icon-actions">
          <button onClick={() => navigator.clipboard?.writeText(reportText)} title="Copy report"><Copy size={18} /></button>
          <button onClick={download} title="Download report"><Download size={18} /></button>
        </div>
      </header>
      <article className="summary-block">
        <h3>Executive Summary</h3>
        <p>{result.executive_summary}</p>
      </article>
      <div className="horizon-grid">
        {[
          ["1 Day", recs.one_day],
          ["1 Week", recs.one_week],
          ["1 Month", recs.one_month],
        ].map(([label, rec]) => (
          <article className="horizon-card" key={label}>
            <span>{label}</span>
            <strong>{rec?.position || "NEUTRAL"}</strong>
            <div className="bar"><i style={{ width: `${convictionValue(rec?.conviction || result.conviction)}%` }} /></div>
            <p>{rec?.rationale || rec?.expected_direction || "Recommendation rationale unavailable."}</p>
          </article>
        ))}
      </div>
      <article className="report-section">
        <h3>Risk-Reward</h3>
        <div className="risk-bar"><i /><b /></div>
        <p>{result.risk_reward}</p>
      </article>
      <article className="report-section">
        <h3>Investment Conclusion</h3>
        <p>{result.conclusion}</p>
      </article>
      <footer>Powered by FinDebate Safe Debate Protocol</footer>
    </section>
  );
}
