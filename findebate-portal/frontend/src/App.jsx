import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Header from "./components/Header.jsx";
import SearchBar from "./components/SearchBar.jsx";
import PipelineProgress from "./components/PipelineProgress.jsx";
import StockBanner from "./components/StockBanner.jsx";
import AgentCards from "./components/AgentCards.jsx";
import FinalReport from "./components/FinalReport.jsx";
import ScoresPanel from "./components/ScoresPanel.jsx";
import DebateTrace from "./components/DebateTrace.jsx";

export default function App() {
  const [mode, setMode] = useState("precomputed");
  const [question, setQuestion] = useState("");
  const [jobId, setJobId] = useState(null);
  const [stage, setStage] = useState("idle");
  const [pipelineEvents, setPipelineEvents] = useState([]);
  const [result, setResult] = useState(null);
  const [companies, setCompanies] = useState([]);
  const [pipelineAvailable, setPipelineAvailable] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/companies").then((res) => res.json()).then(setCompanies).catch(() => setCompanies([]));
    fetch("/api/config")
      .then((res) => res.json())
      .then((data) => setPipelineAvailable(Boolean(data.pipeline_available)))
      .catch(() => setPipelineAvailable(false));
  }, []);

  useEffect(() => {
    if (!jobId) return undefined;
    const events = new EventSource(`/api/stream/${jobId}`);
    events.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      setPipelineEvents((current) => [...current, payload]);
      if (payload.stage === "complete") {
        setResult(payload.data);
        setStage("complete");
        events.close();
      }
      if (payload.stage === "error") {
        setError(payload.data?.message || "Analysis failed");
        setStage("idle");
        events.close();
      }
    };
    events.onerror = () => {
      events.close();
    };
    return () => events.close();
  }, [jobId]);

  useEffect(() => {
    if (!jobId || stage !== "streaming") return undefined;
    const timer = window.setInterval(async () => {
      const response = await fetch(`/api/results/${jobId}`).catch(() => null);
      if (!response || response.status !== 200) return;
      const data = await response.json();
      if (data?.error) {
        setError(data.error);
        setStage("idle");
        return;
      }
      setResult(data);
      setStage("complete");
    }, 1500);
    return () => window.clearInterval(timer);
  }, [jobId, stage]);

  const handleSubmit = async ({ ticker, question: submittedQuestion }) => {
    setError("");
    setResult(null);
    setPipelineEvents([]);
    setStage("streaming");
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode,
        ticker: ticker || null,
        question: submittedQuestion || question,
      }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      setError(detail.detail || "Unable to start analysis");
      setStage("idle");
      return;
    }
    const data = await response.json();
    setJobId(data.job_id);
  };

  const activeTicker = useMemo(() => {
    if (result?.ticker) return result.ticker;
    const rag = pipelineEvents.find((event) => event.stage === "rag" && event.status === "done");
    return rag?.data?.top_sources?.[0]?.split("_")?.[0] || "";
  }, [pipelineEvents, result]);

  return (
    <main className="app-shell">
      <Header
        mode={mode}
        setMode={setMode}
        pipelineAvailable={pipelineAvailable}
        disabled={stage === "streaming"}
      />
      <SearchBar
        mode={mode}
        question={question}
        setQuestion={setQuestion}
        companies={companies}
        onSubmit={handleSubmit}
        pipelineAvailable={pipelineAvailable}
        busy={stage === "streaming"}
      />
      {error && <div className="error-banner">{error}</div>}
      <PipelineProgress events={pipelineEvents} streaming={stage === "streaming"} />
      <AnimatePresence>
        {result && (
          <motion.section
            className="result-stack"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            <StockBanner result={result} ticker={activeTicker} />
            <AgentCards outputs={result.agent_outputs || {}} />
            <FinalReport result={result} />
            <ScoresPanel scores={result.scores || {}} />
            <DebateTrace result={result} />
          </motion.section>
        )}
      </AnimatePresence>
    </main>
  );
}
