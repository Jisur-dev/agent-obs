import { useState } from 'react';
import { NAV_GROUPS } from '../data/nav.js';


function Chevron({ open }) {
  return (
    <svg
      aria-hidden="true"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={open ? 'M6 9l6 6 6-6' : 'M9 6l6 6-6 6'} />
    </svg>
  );
}

export default function Aside({ activeId, onNavigate }) {
  const [openIds, setOpenIds] = useState(['']);

  const toggle = (id) =>
    setOpenIds((ids) =>
      ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]
    );

  return (
    <nav className="sidebar" aria-label="Main menu">
      <div className="brand">
        <div className="brand__logo">
          <svg aria-hidden="true" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" />
          </svg>
        </div>
        <span className="brand__name">Jisur</span>
      </div>

      {NAV_GROUPS.map((group) => (
        <div className="nav-group" key={group.title}>
          <div className="nav-group__title">{group.title}</div>

          {group.items.map((item) => {
            const expandable = Boolean(item.subs);
            const open = openIds.includes(item.id);
            const active = activeId === item.id;

            return (
  <div key={item.id}>
    <button
      type="button"
      className={`nav-item${active ? ' nav-item--active' : ''}`}
      aria-expanded={expandable ? open : undefined}
      onClick={() => (expandable ? toggle(item.id) : onNavigate(item.id))}
    >
      <span className="nav-item__dot" aria-hidden="true" />
      <span className="nav-item__label">{item.label}</span>
      {expandable && <Chevron open={open} />}
    </button>

    {expandable && open && (
      <div className="nav-sub">
        {item.subs.map((sub) => (
          <button
            type="button"
            className="nav-sub__item"
            key={sub}
            onClick={() => onNavigate(item.id)}
          >
            {sub}
          </button>
        ))}
      </div>
    )}
  </div>
);
          })}
        </div>
      ))}
    </nav>
  );
}