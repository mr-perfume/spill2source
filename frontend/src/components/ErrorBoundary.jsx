import React from "react";

/**
 * A rendering error anywhere in the map layers would otherwise throw away the
 * whole React tree, leaving a flat background with no clue what went wrong.
 * This catches it, keeps the rest of the page alive, and shows the message and
 * stack so the failure is diagnosable instead of invisible.
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null, info: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    this.setState({ info });
    console.error(`[${this.props.label || "component"}] render failed`, error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <div className="flex h-full w-full items-center justify-center p-6">
        <div className="max-w-[560px] rounded-sm border border-alert/60 bg-hull p-4">
          <h2 className="text-[13px] font-medium text-alert">
            {this.props.label || "This panel"} could not render
          </h2>
          <p className="mt-1 text-[12px] leading-snug text-chart/85">
            The rest of the page still works. Copy the message below when reporting it.
          </p>
          <pre className="scroll-thin mt-3 max-h-52 overflow-auto rounded-sm bg-abyss p-2.5 font-mono text-[10px] leading-relaxed text-chart/75">
{String(this.state.error?.stack || this.state.error)}
{this.state.info?.componentStack || ""}
          </pre>
          <button
            type="button"
            onClick={() => this.setState({ error: null, info: null })}
            className="mt-3 rounded-sm border border-rule px-3 py-1.5 text-[12px] text-chart
              transition hover:border-posterior hover:text-posterior focus:outline-none
              focus-visible:ring-2 focus-visible:ring-posterior"
          >
            Try rendering again
          </button>
        </div>
      </div>
    );
  }
}
