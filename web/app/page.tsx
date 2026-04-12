export default function CoffeeFuturesSite() {
  const contracts = [
    { month: "May 2026", symbol: "KCK26", price: "193.40", change: "+2.15", pct: "+1.12%", volume: "28.4K", openInterest: "112.9K" },
    { month: "Jul 2026", symbol: "KCN26", price: "196.10", change: "+1.80", pct: "+0.93%", volume: "19.2K", openInterest: "97.6K" },
    { month: "Sep 2026", symbol: "KCU26", price: "198.75", change: "+1.55", pct: "+0.79%", volume: "14.9K", openInterest: "85.1K" },
    { month: "Dec 2026", symbol: "KCZ26", price: "201.90", change: "+1.25", pct: "+0.62%", volume: "11.1K", openInterest: "73.9K" },
  ];

  const headlines = [
    { title: "Brazil weather risk supports nearby strength", source: "Market brief", time: "2h" },
    { title: "Certified stocks remain in focus", source: "Commodities desk", time: "5h" },
    { title: "Roaster hedging ticks up into summer", source: "Trade note", time: "Today" },
  ];

  const curve = [193.4, 196.1, 198.75, 201.9];
  const min = Math.min(...curve);
  const max = Math.max(...curve);

  const points = curve
    .map((value, index) => {
      const x = 28 + index * 96;
      const normalized = (value - min) / (max - min || 1);
      const y = 104 - normalized * 44;
      return `${x},${y}`;
    })
    .join(" ");

  const stats = [
    { label: "Front", value: "193.40", sub: "US¢/lb" },
    { label: "Shape", value: "Contango", sub: "Deferred > spot" },
    { label: "Vol", value: "73.6K", sub: "Shown" },
    { label: "OI", value: "369.4K", sub: "Shown" },
  ];

  return (
    <div className="min-h-screen bg-black text-zinc-100">
      <div className="mx-auto max-w-6xl px-4 py-4 sm:px-6 lg:px-8">
        <div className="grid gap-4 lg:grid-cols-[1.7fr_1fr]">
          <section className="space-y-4 rounded-[28px] border border-white/10 bg-zinc-950 p-4 sm:p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div className="max-w-2xl">
                <div className="inline-flex rounded-full border border-emerald-500/20 bg-emerald-500/8 px-2.5 py-1 text-[11px] font-medium tracking-wide text-emerald-300">
                  ICE Arabica Coffee Futures
                </div>
                <h1 className="mt-3 max-w-xl text-2xl font-semibold tracking-tight text-white sm:text-3xl">
                  Coffee futures, simplified.
                </h1>
                <p className="mt-2 max-w-lg text-sm leading-6 text-zinc-400">
                  A tighter, cleaner dashboard for curve shape, key contracts, and market context.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2 sm:w-[320px]">
                {stats.map((stat) => (
                  <div key={stat.label} className="rounded-2xl border border-white/8 bg-white/[0.03] p-3">
                    <div className="text-[10px] uppercase tracking-[0.18em] text-zinc-500">{stat.label}</div>
                    <div className="mt-1 text-lg font-semibold text-white">{stat.value}</div>
                    <div className="text-xs text-zinc-500">{stat.sub}</div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
              <div className="rounded-3xl border border-white/8 bg-white/[0.02] p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <h2 className="text-base font-medium text-white">Forward curve</h2>
                    <p className="text-xs text-zinc-500">Nearby ICE contracts</p>
                  </div>
                  <button className="rounded-full border border-white/10 px-3 py-1.5 text-xs text-zinc-300 transition hover:bg-white/5">
                    Live feed
                  </button>
                </div>

                <div className="rounded-2xl border border-white/8 bg-black p-3">
                  <svg viewBox="0 0 320 128" className="h-32 w-full">
                    {[28, 124, 220, 316].map((x) => (
                      <line key={x} x1={x} y1="20" x2={x} y2="108" stroke="currentColor" strokeWidth="1" className="text-white/8" />
                    ))}
                    {[32, 56, 80, 104].map((y) => (
                      <line key={y} x1="28" y1={y} x2="316" y2={y} stroke="currentColor" strokeWidth="1" className="text-white/8" />
                    ))}
                    <polyline fill="none" stroke="currentColor" strokeWidth="2.5" className="text-emerald-400" points={points} />
                    {curve.map((value, index) => {
                      const x = 28 + index * 96;
                      const normalized = (value - min) / (max - min || 1);
                      const y = 104 - normalized * 44;
                      return (
                        <g key={value}>
                          <circle cx={x} cy={y} r="3.5" className="fill-emerald-400" />
                          <text x={x} y={118} textAnchor="middle" className="fill-zinc-500 text-[9px]">
                            {contracts[index].month.split(" ")[0]}
                          </text>
                          <text x={x} y={y - 8} textAnchor="middle" className="fill-zinc-300 text-[9px]">
                            {value.toFixed(1)}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                </div>
              </div>

              <div className="rounded-3xl border border-white/8 bg-white/[0.02] p-4">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-base font-medium text-white">Today</h2>
                  <span className="text-xs text-zinc-500">3 notes</span>
                </div>
                <div className="space-y-2.5">
                  {headlines.map((item) => (
                    <article key={item.title} className="rounded-2xl border border-white/8 bg-black/50 p-3">
                      <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.16em] text-zinc-500">
                        <span>{item.source}</span>
                        <span>{item.time}</span>
                      </div>
                      <h3 className="mt-1.5 text-sm font-medium leading-5 text-zinc-100">{item.title}</h3>
                    </article>
                  ))}
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-white/8 bg-white/[0.02] p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-medium text-white">Contracts</h2>
                  <p className="text-xs text-zinc-500">Compact contract cards</p>
                </div>
                <div className="rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400">Delayed demo</div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {contracts.map((contract) => (
                  <article key={contract.symbol} className="rounded-2xl border border-white/8 bg-black/55 p-3.5">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="text-[10px] uppercase tracking-[0.18em] text-zinc-500">{contract.symbol}</div>
                        <h3 className="mt-1 text-sm font-medium text-white">{contract.month}</h3>
                      </div>
                      <div className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-300">{contract.pct}</div>
                    </div>

                    <div className="mt-4 text-3xl font-semibold tracking-tight text-white">{contract.price}</div>
                    <div className="mt-1 text-xs text-zinc-500">US¢/lb settlement</div>

                    <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <div className="text-zinc-500">Change</div>
                        <div className="text-emerald-300">{contract.change}</div>
                      </div>
                      <div>
                        <div className="text-zinc-500">Volume</div>
                        <div className="text-zinc-300">{contract.volume}</div>
                      </div>
                      <div className="col-span-2">
                        <div className="text-zinc-500">Open interest</div>
                        <div className="text-zinc-300">{contract.openInterest}</div>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          </section>

          <aside className="space-y-4">
            <div className="rounded-[28px] border border-white/10 bg-zinc-950 p-4 sm:p-5">
              <h2 className="text-base font-medium text-white">Notes</h2>
              <div className="mt-3 space-y-2 text-sm leading-6 text-zinc-400">
                <p>Reduced headline size, tighter spacing, and fewer words.</p>
                <p>Moved the news section higher so everything important lands above the fold on larger screens.</p>
                <p>Compressed contract cards and chart height to cut scrolling.</p>
              </div>
            </div>

            <div className="rounded-[28px] border border-emerald-500/15 bg-emerald-500/[0.06] p-4 sm:p-5">
              <h2 className="text-base font-medium text-emerald-100">API shape</h2>
              <pre className="mt-3 overflow-x-auto rounded-2xl bg-black/30 p-3 text-[11px] leading-5 text-emerald-50">
{`GET /api/coffee-futures
{
  "updatedAt": "2026-04-12T14:30:00Z",
  "contracts": [{
    "symbol": "KCK26",
    "month": "May 2026",
    "settlement": 193.40
  }]
}`}
              </pre>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
