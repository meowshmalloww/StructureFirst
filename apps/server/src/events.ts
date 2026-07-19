import type { PipelineEvent } from "@structurefirst/contracts";

type Listener = (event: PipelineEvent) => void;

export class CaseEventHub {
  private readonly listeners = new Map<string, Set<Listener>>();

  publish(event: PipelineEvent): void {
    for (const listener of this.listeners.get(event.caseId) ?? []) {
      listener(event);
    }
  }

  subscribe(caseId: string, listener: Listener): () => void {
    const existing = this.listeners.get(caseId) ?? new Set<Listener>();
    existing.add(listener);
    this.listeners.set(caseId, existing);
    return () => {
      existing.delete(listener);
      if (existing.size === 0) this.listeners.delete(caseId);
    };
  }
}
