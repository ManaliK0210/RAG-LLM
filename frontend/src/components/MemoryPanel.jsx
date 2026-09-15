import { useState } from "react";

import {
  clearMemory,
  recallMemory,
  remember,
} from "../services/api";

function MemoryPanel() {
  const [content, setContent] =
    useState("");

  const [query, setQuery] =
    useState("");

  const [memories, setMemories] =
    useState([]);

  const [message, setMessage] =
    useState("");

  const [loading, setLoading] =
    useState(false);

  async function handleRemember() {
    if (!content.trim()) {
      return;
    }

    setLoading(true);
    setMessage("");

    try {
      await remember({
        content,
      });

      setContent("");

      setMessage(
        "Memory saved successfully."
      );
    } catch (error) {
      setMessage(
        error.message
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleRecall() {
    if (!query.trim()) {
      return;
    }

    setLoading(true);
    setMessage("");

    try {
      const response =
        await recallMemory({
          query,
          topK: 5,
        });

      setMemories(
        response.memories
      );
    } catch (error) {
      setMessage(
        error.message
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleClear() {
    setLoading(true);

    try {
      await clearMemory();

      setMemories([]);

      setMessage(
        "Memory cleared."
      );
    } catch (error) {
      setMessage(
        error.message
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="memory-panel">
      <div className="panel-heading">
        <h2>
          Memory
        </h2>

        <span>
          Long-term context
        </span>
      </div>

      <textarea
        value={content}
        onChange={(event) =>
          setContent(
            event.target.value
          )
        }
        placeholder="Store something..."
        rows={3}
      />

      <button
        onClick={
          handleRemember
        }
        disabled={
          loading ||
          !content.trim()
        }
      >
        Save Memory
      </button>

      <div className="memory-divider" />

      <input
        value={query}
        onChange={(event) =>
          setQuery(
            event.target.value
          )
        }
        placeholder="Recall memory..."
      />

      <button
        onClick={handleRecall}
        disabled={
          loading ||
          !query.trim()
        }
      >
        Recall
      </button>

      {memories.length > 0 && (
        <div className="memory-results">
          {memories.map(
            (memory, index) => (
              <div
                className="memory-card"
                key={index}
              >
                <strong>
                  {
                    memory.memory_type
                  }
                </strong>

                <p>
                  {
                    memory.content
                  }
                </p>
              </div>
            )
          )}
        </div>
      )}

      <button
        className="danger-button"
        onClick={
          handleClear
        }
        disabled={loading}
      >
        Clear Memory
      </button>

      {message && (
        <p className="memory-message">
          {message}
        </p>
      )}
    </section>
  );
}

export default MemoryPanel;