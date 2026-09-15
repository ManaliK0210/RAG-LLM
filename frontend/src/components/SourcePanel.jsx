function SourcePanel({
  sources = [],
}) {
  if (!sources.length) {
    return null;
  }

  return (
    <section className="result-panel">
      <div className="panel-title">
        Sources
      </div>

      <ul className="source-list">
        {sources.map(
          (source, index) => (
            <li
              key={`${source}-${index}`}
            >
              {source}
            </li>
          )
        )}
      </ul>
    </section>
  );
}

export default SourcePanel;