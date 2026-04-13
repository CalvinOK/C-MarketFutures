import Image from "next/image";

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
    { label: "Front", value: "193.40", sub: "US¢/lb", featured: true },
    { label: "Shape", value: "Contango", sub: "Deferred > spot" },
    { label: "Vol", value: "73.6K", sub: "Shown" },
    { label: "OI", value: "369.4K", sub: "Shown" },
  ];

  return (
    <div className="min-h-screen bg-[var(--page-bg)] text-[var(--ink)]">
      <div className="mx-auto max-w-6xl px-4 py-5 sm:px-6 lg:px-8">
        <section className="space-y-4 rounded-[28px] border border-[var(--line)] bg-[var(--panel)] p-4 shadow-[0_24px_80px_rgba(32,44,102,0.08)] sm:p-5">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div className="max-w-3xl flex-1">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <h1 className="mt-3 max-w-xl text-2xl font-semibold tracking-tight text-[var(--bond-blue)] sm:text-3xl">
                    Great Lakes Coffee Futures
                  </h1>
                  <p className="mt-2 max-w-lg text-sm leading-6 text-[var(--muted)]">
                    ICE Arabia Coffee Futures
                  </p>
                </div>

                <div className="flex h-16 shrink-0 items-center rounded-2xl border border-[var(--line)] bg-white px-4 shadow-[0_8px_24px_rgba(32,44,102,0.06)] sm:h-20 sm:px-5">
                  <Image
                    src="/bond-logo-navy.png"
                    alt="Bond Consulting"
                    width={320}
                    height={96}
                    className="h-10 w-auto object-contain sm:h-12"
                    priority
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
            <div className="rounded-3xl border border-[var(--line)] bg-[var(--baby-blue)]/25 p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-medium text-[var(--bond-blue)]">Forward curve</h2>
                  <p className="text-xs text-[var(--muted)]">Nearby ICE contracts</p>
                </div>
                <button className="rounded-full border border-[var(--line-strong)] bg-white px-3 py-1.5 text-xs font-medium text-[var(--bond-blue)] transition hover:bg-[var(--baby-blue)]/40">
                  Live feed
                </button>
              </div>

              <div className="rounded-2xl border border-[var(--line)] bg-white p-3">
                <svg viewBox="0 0 320 128" className="h-32 w-full">
                  {[28, 124, 220, 316].map((x) => (
                    <line key={x} x1={x} y1="20" x2={x} y2="108" stroke="currentColor" strokeWidth="1" className="text-[var(--gray)]/60" />
                  ))}
                  {[32, 56, 80, 104].map((y) => (
                    <line key={y} x1="28" y1={y} x2="316" y2={y} stroke="currentColor" strokeWidth="1" className="text-[var(--gray)]/60" />
                  ))}
                  <polyline fill="none" stroke="currentColor" strokeWidth="2.5" className="text-[var(--bond-blue)]" points={points} />
                  {curve.map((value, index) => {
                    const x = 28 + index * 96;
                    const normalized = (value - min) / (max - min || 1);
                    const y = 104 - normalized * 44;
                    return (
                      <g key={value}>
                        <circle cx={x} cy={y} r="3.5" className="fill-[var(--bond-blue)]" />
                        <text x={x} y={118} textAnchor="middle" className="fill-[var(--muted)] text-[9px]">
                          {contracts[index].month.split(" ")[0]}
                        </text>
                        <text x={x} y={y - 8} textAnchor="middle" className="fill-[var(--ink)] text-[9px]">
                          {value.toFixed(1)}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            </div>

            <div className="rounded-3xl border border-[var(--line)] bg-white p-4">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-base font-medium text-[var(--bond-blue)]">Today</h2>
                <span className="text-xs text-[var(--muted)]">3 notes</span>
              </div>
              <div className="space-y-2.5">
                {headlines.map((item, index) => (
                  <article
                    key={item.title}
                    className={`rounded-2xl border p-3 ${
                      index === 0
                        ? "border-[var(--bond-blue)]/12 bg-[var(--bond-blue)]/5"
                        : "border-[var(--line)] bg-[var(--page-bg)]"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">
                      <span>{item.source}</span>
                      <span>{item.time}</span>
                    </div>
                    <h3 className="mt-1.5 text-sm font-medium leading-5 text-[var(--ink)]">{item.title}</h3>
                  </article>
                ))}
              </div>
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-[1.25fr_0.75fr]">
            <div className="rounded-3xl border border-[var(--line)] bg-white p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-medium text-[var(--bond-blue)]">Contracts</h2>
                  <p className="text-xs text-[var(--muted)]">Compact contract cards</p>
                </div>
                <div className="rounded-full border border-[var(--line-strong)] bg-[var(--prasad-purple)]/18 px-3 py-1 text-xs font-medium text-[var(--bond-blue)]">
                  Delayed demo
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {contracts.map((contract, index) => (
                  <article
                    key={contract.symbol}
                    className={`rounded-2xl border p-3.5 ${
                      index === 0
                        ? "border-[var(--bond-blue)]/16 bg-[var(--bond-blue)] text-white shadow-[0_14px_30px_rgba(32,44,102,0.18)]"
                        : "border-[var(--line)] bg-[var(--page-bg)]"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div
                          className={`text-[10px] uppercase tracking-[0.18em] ${
                            index === 0 ? "text-white/70" : "text-[var(--muted)]"
                          }`}
                        >
                          {contract.symbol}
                        </div>
                        <h3 className={`mt-1 text-sm font-medium ${index === 0 ? "text-white" : "text-[var(--ink)]"}`}>
                          {contract.month}
                        </h3>
                      </div>
                      <div
                        className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
                          index === 0
                            ? "bg-white/14 text-white"
                            : "bg-[var(--what-it-do-blue)]/20 text-[var(--bond-blue)]"
                        }`}
                      >
                        {contract.pct}
                      </div>
                    </div>

                    <div className={`mt-4 text-3xl font-semibold tracking-tight ${index === 0 ? "text-white" : "text-[var(--bond-blue)]"}`}>
                      {contract.price}
                    </div>
                    <div className={`mt-1 text-xs ${index === 0 ? "text-white/70" : "text-[var(--muted)]"}`}>US¢/lb settlement</div>

                    <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <div className={index === 0 ? "text-white/65" : "text-[var(--muted)]"}>Change</div>
                        <div className={index === 0 ? "text-white" : "text-[var(--bond-blue)]"}>{contract.change}</div>
                      </div>
                      <div>
                        <div className={index === 0 ? "text-white/65" : "text-[var(--muted)]"}>Volume</div>
                        <div className={index === 0 ? "text-white/85" : "text-[var(--ink)]"}>{contract.volume}</div>
                      </div>
                      <div className="col-span-2">
                        <div className={index === 0 ? "text-white/65" : "text-[var(--muted)]"}>Open interest</div>
                        <div className={index === 0 ? "text-white/85" : "text-[var(--ink)]"}>{contract.openInterest}</div>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border border-[var(--line)] bg-[linear-gradient(180deg,rgba(123,159,188,0.18),rgba(197,174,203,0.14))] p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-medium text-[var(--bond-blue)]">Market snapshot</h2>
                  <p className="text-xs text-[var(--muted)]">Core futures metrics at a glance</p>
                </div>
                <div className="rounded-full border border-[var(--line-strong)] bg-white px-3 py-1 text-xs font-medium text-[var(--bond-blue)]">
                  Live summary
                </div>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3">
                {stats.map((stat) => (
                  <div
                    key={stat.label}
                    className={`rounded-2xl border p-4 ${
                      stat.featured
                        ? "border-[var(--bond-blue)]/20 bg-[var(--bond-blue)] text-white shadow-[0_14px_30px_rgba(32,44,102,0.14)]"
                        : "border-[var(--line)] bg-white/72 backdrop-blur-sm"
                    }`}
                  >
                    <div
                      className={`text-[10px] uppercase tracking-[0.18em] ${
                        stat.featured ? "text-white/70" : "text-[var(--muted)]"
                      }`}
                    >
                      {stat.label}
                    </div>
                    <div className={`mt-2 text-2xl font-semibold tracking-tight ${stat.featured ? "text-white" : "text-[var(--bond-blue)]"}`}>
                      {stat.value}
                    </div>
                    <div className={`mt-1 text-xs ${stat.featured ? "text-white/70" : "text-[var(--muted)]"}`}>{stat.sub}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
