import { Bar, BarChart, PolarAngleAxis, PolarGrid, Radar, RadarChart, ReferenceLine, ResponsiveContainer, XAxis, YAxis } from "recharts";

const LABELS = {
  readability: "Readability",
  linguistic_abstractness: "Abstractness",
  coherence: "Coherence",
  financial_key_point_coverage: "Key Points",
  background_context_adequacy: "Context",
  management_sentiment_conveyance: "Sentiment",
  future_outlook_analysis: "Outlook",
  factual_accuracy: "Accuracy",
};

export default function ScoresPanel({ scores }) {
  const data = Object.entries(LABELS).map(([key, label]) => ({
    key,
    label,
    score: Number(scores[key] || 0),
    fill: Number(scores[key] || 0) < 2.5 ? "#ef4444" : Number(scores[key] || 0) < 3 ? "#f59e0b" : "#22c55e",
  }));
  const avg = Number(scores.avg_overall || 0);
  const diff = avg - 3.71;

  return (
    <section className="scores-panel">
      <h2>LLM Judge Evaluation - 8 Dimensions</h2>
      <div className="score-layout">
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={data}>
            <PolarGrid stroke="rgba(255,255,255,0.12)" />
            <PolarAngleAxis dataKey="label" tick={{ fill: "#94a3b8", fontSize: 11 }} />
            <Radar dataKey="score" stroke="#14b8a6" fill="#14b8a6" fillOpacity={0.35} />
          </RadarChart>
        </ResponsiveContainer>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
            <XAxis type="number" domain={[0, 4]} stroke="#64748b" />
            <YAxis type="category" dataKey="label" width={84} stroke="#94a3b8" tick={{ fontSize: 11 }} />
            <ReferenceLine x={3} stroke="#f59e0b" strokeDasharray="4 4" />
            <Bar dataKey="score" radius={[0, 6, 6, 0]} fill="#14b8a6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="score-ring" style={{ "--score": avg / 4 }}>
        <svg viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="48" />
          <circle className="progress" cx="60" cy="60" r="48" />
        </svg>
        <div>
          <strong>{avg.toFixed(2)} / 4.00</strong>
          <span>Paper average: 3.71 | This analysis: {avg.toFixed(2)} {diff >= 0 ? "above" : "below"}</span>
        </div>
      </div>
    </section>
  );
}
