function AgentSteps({
  steps,
}) {
  if (
    steps === null ||
    steps === undefined
  ) {
    return null;
  }

  return (
    <section className="result-panel">
      <div className="panel-title">
        Agent Execution
      </div>

      <div className="agent-stat">
        <span>
          Reasoning steps
        </span>

        <strong>
          {steps}
        </strong>
      </div>
    </section>
  );
}

export default AgentSteps;