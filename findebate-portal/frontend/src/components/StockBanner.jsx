import { ArrowDown, ArrowRight, ArrowUp } from "lucide-react";

function stanceClass(stance = "") {
  const value = stance.toUpperCase();
  if (value.includes("BULL")) return "bullish";
  if (value.includes("BEAR")) return "bearish";
  return "neutral";
}

function Icon({ stance }) {
  const type = stanceClass(stance);
  if (type === "bullish") return <ArrowUp size={28} />;
  if (type === "bearish") return <ArrowDown size={28} />;
  return <ArrowRight size={28} />;
}

export default function StockBanner({ result, ticker }) {
  const recs = result.investment_recommendations || {};
  return (
    <section className="stock-banner">
      <div>
        <span className="eyebrow">{result.source_file}</span>
        <h2>{result.company || ticker} <small>{ticker || result.ticker}</small></h2>
        <p>Earnings Analysis Report</p>
      </div>
      <div className={`stance ${stanceClass(result.stance)}`}>
        <Icon stance={result.stance} />
        <strong>{result.stance}</strong>
        <span>{result.conviction} conviction</span>
      </div>
      <div className="recommendation-pills">
        {[
          ["1-Day", recs.one_day],
          ["1-Week", recs.one_week],
          ["1-Month", recs.one_month],
        ].map(([label, rec]) => (
          <div className={`rec-pill ${(rec?.position || "neutral").toLowerCase()}`} key={label}>
            <span>{label}</span>
            <strong>{rec?.position || "NEUTRAL"}</strong>
            <small>{rec?.conviction || result.conviction}</small>
          </div>
        ))}
      </div>
    </section>
  );
}
