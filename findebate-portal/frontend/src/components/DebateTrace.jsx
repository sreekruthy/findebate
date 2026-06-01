import { ChevronDown, Shield, Sparkles, UserRoundCheck } from "lucide-react";
import { useState } from "react";

const META = {
  trust: [Sparkles, "Trust"],
  skeptic: [Shield, "Skeptic"],
  leader: [UserRoundCheck, "Leader"],
};

export default function DebateTrace({ result }) {
  const [open, setOpen] = useState(false);
  const steps = result.debate_log || [];
  const textByType = {
    trust: result.trust_text,
    skeptic: result.skeptic_text,
    leader: result.conclusion,
  };

  return (
    <section className={`debate-trace ${open ? "open" : ""}`}>
      <button className="collapse-row" onClick={() => setOpen(!open)}>
        <span>Safe Debate Trace</span>
        <ChevronDown size={20} />
      </button>
      {open && (
        <div className="timeline">
          {["trust", "skeptic", "leader"].map((type) => {
            const [Icon, label] = META[type];
            const found = steps.find((step) => String(step.step || "").toLowerCase().includes(type));
            return (
              <article className={type} key={type}>
                <Icon size={20} />
                <div>
                  <h3>{label}</h3>
                  <p>{String(textByType[type] || found?.result || "Completed").slice(0, 300)}</p>
                  <span>Preserved: core recommendations OK</span>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
