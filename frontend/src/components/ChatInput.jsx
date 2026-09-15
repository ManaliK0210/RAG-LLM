import { useState } from "react";

function ChatInput({
  onSubmit,
  loading,
}) {
  const [value, setValue] =
    useState("");

  function handleSubmit(event) {
    event.preventDefault();

    const trimmed =
      value.trim();

    if (!trimmed || loading) {
      return;
    }

    onSubmit(trimmed);
    setValue("");
  }

  return (
    <form
      className="chat-input"
      onSubmit={handleSubmit}
    >
      <textarea
        value={value}
        onChange={(event) =>
          setValue(event.target.value)
        }
        placeholder="Ask RAGForge something..."
        rows={3}
        disabled={loading}
      />

      <button
        type="submit"
        disabled={
          loading ||
          !value.trim()
        }
      >
        {loading
          ? "Thinking..."
          : "Send"}
      </button>
    </form>
  );
}

export default ChatInput;