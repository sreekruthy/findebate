import { Check, CircleDot, Loader2 } from "lucide-react";

const STAGES = [
  { id: "rag", label: "RAG Retrieval" },
  { id: "agents", label: "5 Analyst Agents" },
  { id: "synthesis", label: "Report Synthesis" },
  { id: "debate", label: "Safe Debate" },
  { id: "judge", label: "LLM Judge" },
];

const AGENTS = ["earnings", "market", "sentiment", "valuation", "risk"];

export default function PipelineProgress({ events, streaming }) {
  const statusFor = (stage) => {
    const stageEvents = events.filter((event) => event.stage === stage);
    const laterDone = (laterStages) => laterStages.some((later) => events.some((event) => event.stage === later && event.status === "done"));
    if (stageEvents.some((event) => event.status === "done" || event.status === "all_done")) return "done";
    if (stage === "agents" && laterDone(["synthesis", "debate", "judge", "complete"])) return "done";
    if (stage === "synthesis" && laterDone(["debate", "judge", "complete"])) return "done";
    if (stage === "debate" && laterDone(["judge", "complete"])) return "done";
    if (stage === "judge" && laterDone(["complete"])) return "done";
    if (stageEvents.some((event) => event.status === "running")) return "running";
    return "idle";
  };

  const summaryFor = (stage) => {
    const done = [...events].reverse().find((event) => event.stage === stage && event.status === "done");
    if (!done) return "";
    if (stage === "rag") return `${done.data?.n_chunks || 0} chunks`;
    if (stage === "synthesis") return done.data?.stance || "";
    if (stage === "debate") return `Conviction ${done.data?.conviction || ""}`;
    if (stage === "judge") return `${done.data?.scores?.avg_overall || ""} / 4`;
    return "";
  };

  const completedAgents = new Set(events.filter((event) => event.stage === "agents" && event.agent).map((event) => event.agent));
  const agentsInferredDone = statusFor("agents") === "done";

  return (
    <section className={`pipeline ${streaming ? "is-streaming" : ""}`}>
      {STAGES.map((stage, index) => {
        const status = statusFor(stage.id);
        return (
          <div className={`stage-card ${status}`} key={stage.id} style={{ animationDelay: `${index * 90}ms` }}>
            <div className="stage-icon">
              {status === "done" ? <Check size={18} /> : status === "running" ? <Loader2 size={18} /> : <CircleDot size={18} />}
            </div>
            <strong>{stage.label}</strong>
            <span>{summaryFor(stage.id) || (status === "running" ? "Running" : status === "done" ? "Complete" : "Waiting")}</span>
            {stage.id === "agents" && (
              <div className="agent-dots">
                {AGENTS.map((agent) => (
                  <i className={completedAgents.has(agent) || agentsInferredDone ? "lit" : ""} key={agent}>{agent[0].toUpperCase()}</i>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </section>
  );
}
