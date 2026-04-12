export default function CoffeeFuturesSite() {
  const contracts = [
    { month: "May 2026", symbol: "KCK26", price: "193.40", change: "+2.15", pct: "+1.12%", volume: "28,410", openInterest: "112,883" },
    { month: "Jul 2026", symbol: "KCN26", price: "196.10", change: "+1.80", pct: "+0.93%", volume: "19,204", openInterest: "97,551" },
    { month: "Sep 2026", symbol: "KCU26", price: "198.75", change: "+1.55", pct: "+0.79%", volume: "14,902", openInterest: "85,114" },
    { month: "Dec 2026", symbol: "KCZ26", price: "201.90", change: "+1.25", pct: "+0.62%", volume: "11,088", openInterest: "73,904" },
  ];

  const news = [
    {
      title: "Brazil weather risk supports arabica curve",
      source: "Market Brief",
      time: "2h ago",
      summary: "Dryness concerns in key producing regions are keeping nearby contracts firm relative to deferred months.",
    },
    {
      title: "Certified stocks remain in focus",
      source: "Commodities Desk",
      time: "5h ago",
      summary: "Traders continue watching exchange inventories for signals on tightness in deliverable supply.",
    },
    {
      title: "Roaster hedging picks up into summer",
      source: "Trade Note",
      time: "Today",
      summary: "Commercial hedging activity has increased as buyers manage margin volatility into upcoming shipment windows.",
    },
  ];

  const curve = [193.4, 196.1, 198.75, 201.9];
  const min = Math.min(...curve);
  const max = Math.max(...curve);

  const points = curve
    .map((value, index) => {
      const x = 40 + index * 170;
      const normalized = (value - min) / (max - min || 1);
      const y = 170 - normalized * 100;
      return `${x},${y}`;
    })
    .join(" ");

  const stats = [
    { label: "Front Month", value: "193.40", sub: "US¢/lb" },
    { label: "Curve Shape", value: "Contango", sub: "Deferred > spot" },
    { label: "Daily Volume", value: "73.6K", sub: "Across shown contracts" },
    { label: "Open Interest", value: "369.4K", sub: "Across shown contracts" },
  ];

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50">
      <div className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
        <header className="mb-8 flex flex-col gap-6 rounded-3xl border border-white/10 bg-white/5 p-8 shadow-2xl backdrop-blur md:flex-row md:items-end md:justify-between">
          <div className="max-w-3xl">
            <div className="mb-3 inline-flex items-center rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-sm text-emerald-200">
              ICE Arabica Coffee Futures Dashboard
            </div>
            <h1 className="text-4xl font-semibold tracking-tight md:text-5xl">
              Coffee futures, term structure, and market context in one place.
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-zinc-300 md:text-lg">
              A Vercel-ready dashboard for tracking arabica coffee contracts, market structure,
              inventories, and trade headlines. Replace the sample values with your own live API
              or data feed.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 md:w-[420px]">
            {stats.map((stat) => (
              <div key={stat.label} className="rounded-2xl border border-white/10 bg-black/20 p-4">
                <div className="text-xs uppercase tracking-[0.18em] text-zinc-400">{stat.label}</div>
                <div className="mt-2 text-2xl font-semibold">{stat.value}</div>
                <div className="mt-1 text-sm text-zinc-400">{stat.sub}</div>
              </div>
            ))}
          </div>
        </header>

        <main className="grid gap-6 lg:grid-cols-12">
          <section className="lg:col-span-8 space-y-6">
            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl">
              <div className="mb-5 flex items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-semibold">Forward curve</h2>
                  <p className="mt-1 text-sm text-zinc-400">
                    Sample settlement curve for nearby ICE coffee contracts.
                  </p>
                </div>
                <button className="rounded-xl border border-white/10 bg-white/10 px-4 py-2 text-sm transition hover:bg-white/15">
                  Connect live feed
                </button>
              </div>

              <div className="overflow-hidden rounded-2xl border border-white/10 bg-zinc-950/80 p-4">
                <svg viewBox="0 0 560 220" className="h-64 w-full">
                  {[0, 1, 2, 3].map((i) => (
                    <line
                      key={`v-${i}`}
                      x1={40 + i * 170}
                      y1="20"
                      x2={40 + i * 170}
                      y2="180"
                      className="stroke-white/10"
                      strokeWidth="1"
                    />
                  ))}
                  {[40, 80, 120, 160].map((y) => (
                    <line
                      key={`h-${y}`}
                      x1="40"
                      y1={y}
                      x2="550"
                      y2={y}
                      className="stroke-white/10"
                      strokeWidth="1"
                    />
                  ))}
                  <polyline
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="4"
                    className="text-emerald-400"
                    points={points}
                  />
                  {curve.map((value, index) => {
                    const x = 40 + index * 170;
                    const normalized = (value - min) / (max - min || 1);
                    const y = 170 - normalized * 100;
                    return (
                      <g key={value}>
                        <circle cx={x} cy={y} r="6" className="fill-emerald-400" />
                        <text x={x} y={200} textAnchor="middle" className="fill-zinc-400 text-[11px]">
                          {contracts[index].month}
                        </text>
                        <text x={x} y={y - 12} textAnchor="middle" className="fill-zinc-200 text-[11px]">
                          {value.toFixed(2)}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-semibold">Contracts</h2>
                  <p className="mt-1 text-sm text-zinc-400">
                    Track front-month and deferred pricing, liquidity, and positioning.
                  </p>
                </div>
                <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-2 text-sm text-zinc-300">
                  Delayed demo data
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                {contracts.map((contract) => (
                  <article
                    key={contract.symbol}
                    className="rounded-2xl border border-white/10 bg-zinc-950/70 p-5 transition hover:-translate-y-0.5 hover:bg-zinc-900"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="text-sm uppercase tracking-[0.18em] text-zinc-400">
                          {contract.symbol}
                        </div>
                        <h3 className="mt-1 text-xl font-semibold">{contract.month}</h3>
                      </div>
                      <div className="rounded-full bg-emerald-400/10 px-3 py-1 text-sm text-emerald-300">
                        {contract.pct}
                      </div>
                    </div>
                    <div className="mt-5 flex items-end justify-between">
                      <div>
                        <div className="text-4xl font-semibold">{contract.price}</div>
                        <div className="mt-1 text-sm text-zinc-400">US¢/lb settlement</div>
                      </div>
                      <div className="text-right text-sm">
                        <div className="text-emerald-300">{contract.change} today</div>
                        <div className="mt-1 text-zinc-400">Volume {contract.volume}</div>
                        <div className="text-zinc-400">OI {contract.openInterest}</div>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          </section>

          <aside className="lg:col-span-4 space-y-6">
            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl">
              <h2 className="text-2xl font-semibold">What matters today</h2>
              <div className="mt-5 space-y-4">
                {news.map((item) => (
                  <article key={item.title} className="rounded-2xl border border-white/10 bg-zinc-950/70 p-4">
                    <div className="flex items-center justify-between gap-3 text-xs uppercase tracking-[0.16em] text-zinc-400">
                      <span>{item.source}</span>
                      <span>{item.time}</span>
                    </div>
                    <h3 className="mt-2 text-base font-medium text-zinc-100">{item.title}</h3>
                    <p className="mt-2 text-sm leading-6 text-zinc-400">{item.summary}</p>
                  </article>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border border-white/10 bg-white/5 p-6 shadow-xl">
              <h2 className="text-2xl font-semibold">Build notes</h2>
              <div className="mt-4 space-y-3 text-sm leading-6 text-zinc-300">
                <p>
                  This page is designed to drop into a Next.js app on Vercel as a landing page or dashboard route.
                </p>
                <p>
                  For production, replace the hard-coded arrays with data from your preferred commodity market source and cache results in a server route.
                </p>
                <p>
                  Good additions: basis tables, COT positioning, certified stocks, Brazil weather tiles, roaster hedge calculators, and exportable watchlists.
                </p>
              </div>
            </div>

            <div className="rounded-3xl border border-emerald-400/20 bg-emerald-400/10 p-6 shadow-xl">
              <h2 className="text-xl font-semibold text-emerald-100">Suggested API shape</h2>
              <pre className="mt-4 overflow-x-auto rounded-2xl bg-black/30 p-4 text-xs leading-6 text-emerald-50">
{`GET /api/coffee-futures
{
  "updatedAt": "2026-04-12T14:30:00Z",
  "contracts": [
    {
      "symbol": "KCK26",
      "month": "May 2026",
      "settlement": 193.40,
      "change": 2.15,
      "pctChange": 1.12,
      "volume": 28410,
      "openInterest": 112883
    }
  ]
}`}
              </pre>
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
}
