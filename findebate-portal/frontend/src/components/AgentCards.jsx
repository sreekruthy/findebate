import { BarChart3, BriefcaseBusiness, Gauge, MessageSquareText, TrendingUp } from "lucide-react";
import { useState } from "react";

const AGENTS = [
  ["earnings", "Earnings", BriefcaseBusiness],
  ["market", "Market", TrendingUp],
  ["sentiment", "Sentiment", MessageSquareText],
  ["valuation", "Valuation", BarChart3],
  ["risk", "Risk", Gauge],
];

export default function AgentCards({ outputs }) {
  const [open, setOpen] = useState({});
  return (
    <section className="agent-grid">
      {AGENTS.map(([key, label, Icon]) => {
        const text = String(outputs[key] || "No agent output attached to this result.");
        const expanded = open[key];
        return (
          <article className={`agent-card ${key}`} key={key}>
            <header>
              <Icon size={20} />
              <strong>{label}</strong>
            </header>
            <p>{expanded ? text : `${text.slice(0, 200)}${text.length > 200 ? "..." : ""}`}</p>
            {text.length > 200 && (
              <button onClick={() => setOpen((current) => ({ ...current, [key]: !expanded }))}>
                {expanded ? "Show less" : "Show more"}
              </button>
            )}
          </article>
        );
      })}
    </section>
  );
}
