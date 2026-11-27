export const NAV_GROUPS = [
  {
    title: 'MAIN MENU',
    items: [
      { id: 'dashboard', label: 'Dashboard' },
      {
        id: 'agents',
        label: 'Agents',
        subs: ['All agents', 'Versions', 'Prompts and configs', 'Settings'],
      },
      { id: 'traces', label: 'Traces', subs: ['Runs', 'Spans', 'Errors', 'Replay'] },
      { id: 'metrics', label: 'Metrics', subs: ['Latency', 'Token usage', 'Cost'] },
    ],
  }]