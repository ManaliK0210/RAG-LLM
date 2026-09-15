function ModeSelector({
  mode,
  onChange,
}) {
  const modes = [
    {
      id: "generate",
      label: "Generate",
      description:
        "Direct LLM generation",
    },
    {
      id: "rag",
      label: "RAG",
      description:
        "Knowledge-grounded answers",
    },
    {
      id: "agent",
      label: "Agent",
      description:
        "Tool-using reasoning",
    },
  ];

  return (
    <div className="mode-selector">
      {modes.map((item) => (
        <button
          key={item.id}
          className={
            mode === item.id
              ? "mode-button active"
              : "mode-button"
          }
          onClick={() =>
            onChange(item.id)
          }
        >
          <strong>
            {item.label}
          </strong>

          <span>
            {item.description}
          </span>
        </button>
      ))}
    </div>
  );
}

export default ModeSelector;