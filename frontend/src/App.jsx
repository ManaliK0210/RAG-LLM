import { useState } from "react";

import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import ModeSelector from "./components/ModeSelector";
import SourcePanel from "./components/SourcePanel";
import AgentSteps from "./components/AgentSteps";
import MemoryPanel from "./components/MemoryPanel";
import SettingsPanel from "./components/SettingsPanel";

import {
  generateText,
  queryRAG,
  runAgent,
} from "./services/api";

function App() {
  const [mode, setMode] =
    useState("rag");

  const [messages, setMessages] =
    useState([]);

  const [loading, setLoading] =
    useState(false);

  const [sources, setSources] =
    useState([]);

  const [agentSteps, setAgentSteps] =
    useState(null);

  const [error, setError] =
    useState("");

  const [settings, setSettings] =
    useState({
      maxNewTokens: 50,
      temperature: 1.0,
      topK: 20,
    });

  async function handleSubmit(
    prompt
  ) {
    setLoading(true);
    setError("");
    setSources([]);
    setAgentSteps(null);

    setMessages((current) => [
      ...current,
      {
        role: "user",
        content: prompt,
      },
    ]);

    try {
      let response;
      let answer;

      if (mode === "generate") {
        response =
          await generateText({
            prompt,
            maxNewTokens:
              settings.maxNewTokens,
            temperature:
              settings.temperature,
            topK: settings.topK,
          });

        answer = response.text;
      } else if (mode === "rag") {
        response =
          await queryRAG(prompt);

        answer = response.answer;

        setSources(
          response.sources || []
        );
      } else {
        response =
          await runAgent(prompt);

        answer = response.answer;

        setAgentSteps(
          response.steps
        );
      }

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: answer,
        },
      ]);
    } catch (err) {
      setError(
        err.message ||
          "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="brand">
            RAGForge
          </div>

          <div className="tagline">
            Generative AI · RAG · Agents
          </div>
        </div>

        <div className="status-pill">
          ● API Ready
        </div>
      </header>

      <main className="workspace">
        <section className="chat-workspace">
          <ModeSelector
            mode={mode}
            onChange={setMode}
          />

          <div className="chat-header">
            <div>
              <h1>
                {mode === "generate"
                  ? "LLM Generation"
                  : mode === "rag"
                    ? "Knowledge Assistant"
                    : "Agentic Assistant"}
              </h1>

              <p>
                {mode === "generate"
                  ? "Direct language-model generation."
                  : mode === "rag"
                    ? "Answers grounded in retrieved knowledge."
                    : "LLM-driven tool use and reasoning."}
              </p>
            </div>
          </div>

          <div className="messages">
            {messages.length === 0 && (
              <div className="empty-state">
                <div className="empty-icon">
                  ✦
                </div>

                <h2>
                  Ask RAGForge
                </h2>

                <p>
                  Try a question using
                  Generate, RAG, or Agent mode.
                </p>
              </div>
            )}

            {messages.map(
              (message, index) => (
                <ChatMessage
                  key={index}
                  role={
                    message.role
                  }
                  content={
                    message.content
                  }
                />
              )
            )}

            {loading && (
              <div className="typing">
                RAGForge is thinking...
              </div>
            )}
          </div>

          {error && (
            <div className="error-box">
              {error}
            </div>
          )}

          <SourcePanel
            sources={sources}
          />

          <AgentSteps
            steps={agentSteps}
          />

          <ChatInput
            onSubmit={
              handleSubmit
            }
            loading={loading}
          />
        </section>

        <aside className="sidebar">
          <SettingsPanel
            settings={settings}
            onChange={setSettings}
          />

          <MemoryPanel />
        </aside>
      </main>
    </div>
  );
}

export default App;