function SettingsPanel({
  settings,
  onChange,
}) {
  return (
    <section className="settings-panel">
      <div className="panel-heading">
        <h2>
          Generation
        </h2>

        <span>
          Model controls
        </span>
      </div>

      <label>
        Max new tokens

        <input
          type="number"
          min="1"
          max="512"
          value={
            settings.maxNewTokens
          }
          onChange={(event) =>
            onChange({
              ...settings,
              maxNewTokens:
                Number(
                  event.target.value
                ),
            })
          }
        />
      </label>

      <label>
        Temperature

        <input
          type="number"
          min="0.01"
          step="0.1"
          value={
            settings.temperature
          }
          onChange={(event) =>
            onChange({
              ...settings,
              temperature:
                Number(
                  event.target.value
                ),
            })
          }
        />
      </label>

      <label>
        Top-K

        <input
          type="number"
          min="1"
          value={
            settings.topK
          }
          onChange={(event) =>
            onChange({
              ...settings,
              topK:
                Number(
                  event.target.value
                ),
            })
          }
        />
      </label>
    </section>
  );
}

export default SettingsPanel;