"use client";

import Image from "next/image";
import { useEffect, useMemo, useState } from "react";
import bondLogoNavy from "../public/bond-logo-navy.png";

type HistoryRow = {
  date: string;
  price: number;
};

type WeeklyPathRow = {
  asOfDate: string;
  stepWeek: number;
  date: string;
  predictedWeeklyLogReturn: number;
  projectedPrice: number;
  anchorWeeklyLogReturn: number;
  raw1wLogReturn: number;
};

type ForecastBandRow = {
  date: Date;
  projectedPrice: number;
  upper: number;
  lower: number;
};

// ─── Live API types ───────────────────────────────────────────────────────────

type ContractApiRow = {
  symbol: string;
  expiry_date: string;
  last_price: number;
  price_change: number;
  price_change_pct: number;
  volume: number;
  open_interest: number;
  captured_at: string;
};

type NewsApiItem = {
  category: string;
  text: string;
  source: string;
  url?: string;
  timestamp: string;
};

type ProjectedSpotApiResponse = {
  format: "projected-spot-csv.v1";
  files: {
    history: string;
    forecast: string;
  };
  asOfDate: string | null;
  historyCsv: string;
  forecastCsv: string;
};

type SucafinaBriefApiItem = {
  headline: string;
  source_report: string;
  report_url: string;
  market_bias: string;
  key_takeaways: string[];
  generated_at?: string;
};

type TodayFeedItem =
  | {
      kind: "news";
      title: string;
      source: string;
      date: string | null;
      url: string | null;
    }
  | {
      kind: "sucafina";
      title: string;
      source: string;
      date: string | null;
      url: string | null;
      summary: string;
    };

type SnapshotData = {
  frontPrice: number;
  curveShape: "Contango" | "Backwardation";
  totalVolume: number;
  totalOpenInterest: number;
  frontSymbol: string;
  asOf: string;
};

// ─── Display helpers ──────────────────────────────────────────────────────────

const MONTH_CODE_MAP: Record<string, string> = {
  F: "Jan", G: "Feb", H: "Mar", J: "Apr", K: "May", M: "Jun",
  N: "Jul", Q: "Aug", U: "Sep", V: "Oct", X: "Nov", Z: "Dec",
};

function symbolToMonth(symbol: string): string {
  // Handles KC month code and 1-digit/2-digit year suffixes (e.g., KCM6 or KCM26).
  const code = symbol[2];
  const rawYear = symbol.slice(4).trim();

  let fullYear: string;
  if (/^\d{2}$/.test(rawYear)) {
    fullYear = `20${rawYear}`;
  } else if (/^\d$/.test(rawYear)) {
    fullYear = `202${rawYear}`;
  } else if (/^\d{4}$/.test(rawYear)) {
    fullYear = rawYear;
  } else {
    fullYear = rawYear || "N/A";
  }

  return `${MONTH_CODE_MAP[code] ?? code} ${fullYear}`;
}

function formatK(value: number | null | undefined): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return "N/A";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toString();
}

function formatNewsDate(timestamp: string): string | null {
  if (!timestamp) return null;
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return null;

  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(date);
}

async function fetchJsonFromApi<T>(apiPath: string): Promise<T> {
  const apiResponse = await fetch(apiPath, { cache: "no-store" });
  if (apiResponse.ok) {
    return apiResponse.json() as Promise<T>;
  }

  let detail = "";
  try {
    const bodyText = await apiResponse.text();
    detail = bodyText.slice(0, 240).trim();
  } catch {
    detail = "";
  }

  throw new Error(
    detail
      ? `Failed to load ${apiPath} (${apiResponse.status}): ${detail}`
      : `Failed to load ${apiPath} (${apiResponse.status})`,
  );
}

function extractAsOfDateFromCsv(csvText: string): string | null {
  const lines = csvText.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return null;

  const header = lines[0].split(",").map((value) => value.trim().toLowerCase());
  const asOfIndex = header.indexOf("as_of_date");
  if (asOfIndex < 0) return null;

  const firstRow = lines[1].split(",").map((value) => value.trim());
  return firstRow[asOfIndex] || null;
}

async function fetchProjectedSpotWithFallback(): Promise<ProjectedSpotApiResponse> {
  const apiResponse = await fetch("/api/projected-spot", { cache: "no-store" });
  if (apiResponse.ok) {
    return apiResponse.json() as Promise<ProjectedSpotApiResponse>;
  }

  const [historyResponse, forecastResponse] = await Promise.all([
    fetch("/data/coffee_xgb_proj4_history.csv", { cache: "no-store" }),
    fetch("/data/coffee_xgb_proj4_rolling_path.csv", { cache: "no-store" }),
  ]);

  if (!historyResponse.ok || !forecastResponse.ok) {
    throw new Error(`Failed to load projected spot API (${apiResponse.status})`);
  }

  const [historyCsv, forecastCsv] = await Promise.all([
    historyResponse.text(),
    forecastResponse.text(),
  ]);

  return {
    format: "projected-spot-csv.v1",
    files: {
      history: "coffee_xgb_proj4_history.csv",
      forecast: "coffee_xgb_proj4_rolling_path.csv",
    },
    asOfDate: extractAsOfDateFromCsv(forecastCsv),
    historyCsv,
    forecastCsv,
  };
}

type ChartPoint = {
  date: Date;
  x: number;
  y: number;
  label: string;
  projectedPrice: number;
};

type MonthTick = {
  key: string;
  x: number;
  label: string;
};

type YAxisTick = {
  value: number;
  y: number;
  label: string;
};

function parseHistoryCsv(csvText: string): HistoryRow[] {
  const lines = csvText.trim().split(/\r?\n/);
  const rows = lines.slice(1);

  return rows
    .map((line) => line.split(","))
    .filter((parts) => parts.length >= 2)
    .map((parts) => ({
      date: parts[0],
      price: Number(parts[1]),
    }))
    .filter((row) => Number.isFinite(row.price));
}

function parseWeeklyPathCsv(csvText: string): WeeklyPathRow[] {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) {
    return [];
  }

  const header = lines[0].split(",").map((value) => value.trim().toLowerCase());
  const byName = (name: string) => header.indexOf(name.toLowerCase());
  const dateIdx = Math.max(byName("date"), byName("Date"));
  const asOfIdx = Math.max(byName("as_of_date"), byName("asofdate"));
  const stepIdx = Math.max(byName("step_week"), byName("step"));
  const projectedIdx = Math.max(byName("projected_price"), byName("forecast"));
  const predictedIdx = byName("predicted_weekly_log_return");
  const anchorIdx = byName("anchor_weekly_log_return");
  const rawIdx = byName("raw_1w_log_return");

  const rows = lines.slice(1);
  const parsed = rows
    .map((line) => line.split(","))
    .filter((parts) => parts.length >= 2)
    .map((parts) => {
      const date = dateIdx >= 0 ? parts[dateIdx] : parts[2] ?? parts[0];
      const stepWeek =
        stepIdx >= 0 ? Number(parts[stepIdx]) : Number(parts[1] ?? parts[4]);
      const projectedPrice =
        projectedIdx >= 0 ? Number(parts[projectedIdx]) : Number(parts[4]);

      return {
        asOfDate:
          asOfIdx >= 0
            ? parts[asOfIdx]
            : date,
        stepWeek,
        date,
        projectedPrice,
        predictedWeeklyLogReturn:
          predictedIdx >= 0 ? Number(parts[predictedIdx]) : Number.NaN,
        anchorWeeklyLogReturn:
          anchorIdx >= 0 ? Number(parts[anchorIdx]) : 0,
        raw1wLogReturn:
          rawIdx >= 0 ? Number(parts[rawIdx]) : 0,
      };
    })
    .filter(
      (row) =>
        Number.isFinite(row.stepWeek) &&
        Number.isFinite(row.projectedPrice) &&
        row.date,
    );

  if (parsed.length === 0) {
    return [];
  }

  for (let index = 0; index < parsed.length; index += 1) {
    if (!Number.isFinite(parsed[index].predictedWeeklyLogReturn)) {
      if (index === 0) {
        parsed[index].predictedWeeklyLogReturn = 0;
      } else {
        const previous = parsed[index - 1].projectedPrice;
        const current = parsed[index].projectedPrice;
        parsed[index].predictedWeeklyLogReturn =
          previous > 0 && current > 0 ? Math.log(current / previous) : 0;
      }
    }
  }

  const inferredAsOfDate = parsed[0].asOfDate || parsed[0].date;
  return parsed.map((row) => ({
    ...row,
    asOfDate: row.asOfDate || inferredAsOfDate,
  }));
}

function parseLocalDate(dateStr: string): Date {
  const [year, month, day] = dateStr.split("-").map(Number);
  return new Date(year, month - 1, day);
}

function stddev(values: number[]): number {
  const finiteValues = values.filter((value) => Number.isFinite(value));

  if (finiteValues.length < 2) {
    return 0.03;
  }

  const mean =
    finiteValues.reduce((sum, value) => sum + value, 0) / finiteValues.length;

  const variance =
    finiteValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
    (finiteValues.length - 1);

  return Math.sqrt(Math.max(variance, 0));
}

function buildPolyline(points: ChartPoint[]): string {
  return points.map((point) => `${point.x},${point.y}`).join(" ");
}

function niceStep(rawStep: number): number {
  if (rawStep <= 0) return 1;

  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const residual = rawStep / magnitude;

  if (residual <= 1) return 1 * magnitude;
  if (residual <= 2) return 2 * magnitude;
  if (residual <= 5) return 5 * magnitude;
  return 10 * magnitude;
}

function buildYAxisTicks(minValue: number, maxValue: number, count = 5): number[] {
  const range = maxValue - minValue || 1;
  const step = niceStep(range / (count - 1));
  const start = Math.floor(minValue / step) * step;
  const end = Math.ceil(maxValue / step) * step;

  const ticks: number[] = [];
  for (let value = start; value <= end + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(4)));
  }

  return ticks;
}

export default function CoffeeFuturesSite() {
  const glossaryTerms = [
    {
      term: "OI (Open Interest)",
      definition:
        "The total number of outstanding futures contracts that are still open and not yet settled.",
    },
    {
      term: "Volume",
      definition:
        "How many contracts traded during the session. Higher volume usually means better liquidity.",
    },
    {
      term: "Backwardation",
      definition:
        "A market shape where near-dated contracts trade above deferred contracts.",
    },
    {
      term: "Contango",
      definition:
        "A market shape where deferred contracts trade above near-dated contracts.",
    },
    {
      term: "Front Month",
      definition:
        "The nearest contract month that is currently the most actively traded.",
    },
    {
      term: "Deferred",
      definition:
        "Contract months that expire later than the front month.",
    },
    {
      term: "Settlement",
      definition:
        "The official end-of-day price used by the exchange for margining and valuation.",
    },
  ];

  const staticContracts = [
    {
      month: "May 2026",
      symbol: "KCK26",
      price: "193.40",
      change: "+2.15",
      pct: "+1.12%",
      volume: "28.4K",
      openInterest: "112.9K",
    },
    {
      month: "Jul 2026",
      symbol: "KCN26",
      price: "196.10",
      change: "+1.80",
      pct: "+0.93%",
      volume: "19.2K",
      openInterest: "97.6K",
    },
    {
      month: "Sep 2026",
      symbol: "KCU26",
      price: "198.75",
      change: "+1.55",
      pct: "+0.79%",
      volume: "14.9K",
      openInterest: "85.1K",
    },
    {
      month: "Dec 2026",
      symbol: "KCZ26",
      price: "201.90",
      change: "+1.25",
      pct: "+0.62%",
      volume: "11.1K",
      openInterest: "73.9K",
    },
  ];

  const staticHeadlines = [
    {
      title: "Brazil weather risk supports nearby strength",
      source: "Market Brief",
      date: null,
    },
    {
      title: "Certified stocks remain in focus",
      source: "Commodities Desk",
      date: null,
    },
    {
      title: "Roaster hedging ticks up into summer",
      source: "Trade Note",
      date: null,
    },
  ];

  const staticSucafinaSummary: TodayFeedItem = {
    kind: "sucafina",
    title: "Sucafina market report",
    source: "Sucafina Market Report",
    date: null,
    url: "https://sucafina.com/na/lp/market-report",
    summary:
      "The latest Sucafina report points to a bearish tilt, with nearby support around 290 USc/lb and downside risk toward 275.",
  };

  const staticStats = [
    { label: "Front", value: "193.40", sub: "US¢/lb", featured: true },
    { label: "Shape", value: "Contango", sub: "Deferred > spot" },
    { label: "Vol", value: "73.6K", sub: "Shown" },
    { label: "OI", value: "369.4K", sub: "Shown" },
  ];

  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [forecastPath, setForecastPath] = useState<WeeklyPathRow[]>([]);
  const [dataError, setDataError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [hoveredHistoryIndex, setHoveredHistoryIndex] = useState<number | null>(null);

  // Live data from backend API
  const [liveContracts, setLiveContracts] = useState<ContractApiRow[] | null>(null);
  const [contractsUnavailable, setContractsUnavailable] = useState(false);
  const [contractPage, setContractPage] = useState(0);
  const [liveNews, setLiveNews] = useState<NewsApiItem[] | null>(null);
  const [liveSucafinaBrief, setLiveSucafinaBrief] = useState<SucafinaBriefApiItem | null>(null);
  const [liveSnapshot, setLiveSnapshot] = useState<SnapshotData | null>(null);
  const [isGlossaryOpen, setIsGlossaryOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadChartData() {
      try {
        const payload = await fetchProjectedSpotWithFallback();
        const historyText = payload.historyCsv ?? "";
        const pathText = payload.forecastCsv ?? "";

        const historyRows = parseHistoryCsv(historyText);
        const forecastRows = parseWeeklyPathCsv(pathText);

        if (!cancelled) {
          setHistory(historyRows);
          setForecastPath(forecastRows);
          setDataError(
            historyRows.length > 0 && forecastRows.length > 0
              ? null
              : "Projected spot API returned empty CSV data.",
          );
        }
      } catch (error) {
        if (!cancelled) {
          setDataError(
            error instanceof Error
              ? error.message
              : "Failed to load chart data.",
          );
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    loadChartData();

    return () => {
      cancelled = true;
    };
  }, []);

  // Fetch contracts/news/snapshot/brief from API routes.
  useEffect(() => {
    let cancelled = false;
    let firstLoad = true;

    async function loadLiveData() {
      const refreshQuery = firstLoad ? "?run=true" : "";
      firstLoad = false;

      const [contractsRes, newsRes, snapshotRes, briefRes] = await Promise.allSettled([
        fetchJsonFromApi<ContractApiRow[]>(`/api/contracts${refreshQuery}`),
        fetchJsonFromApi<NewsApiItem[]>("/api/news"),
        fetchJsonFromApi<SnapshotData>("/api/snapshot"),
        fetchJsonFromApi<SucafinaBriefApiItem>("/api/brief"),
      ]);

      if (cancelled) return;

      if (contractsRes.status === "fulfilled") {
        const raw = contractsRes.value as ContractApiRow[] | { data: ContractApiRow[] };
        const data = Array.isArray(raw) ? raw : (raw as { data: ContractApiRow[] }).data;
        if (Array.isArray(data) && data.length > 0) {
          setLiveContracts(data);
          setContractsUnavailable(false);
        } else {
          setLiveContracts(null);
          setContractsUnavailable(true);
        }
      } else {
        setLiveContracts(null);
        setContractsUnavailable(true);
      }

      if (newsRes.status === "fulfilled") {
        const raw = newsRes.value as NewsApiItem[] | { data: NewsApiItem[] };
        const data = Array.isArray(raw) ? raw : (raw as { data: NewsApiItem[] }).data;
        if (Array.isArray(data) && data.length > 0) setLiveNews(data);
      }

      if (snapshotRes.status === "fulfilled") {
        const data = snapshotRes.value;
        if (data?.frontPrice) {
          setLiveSnapshot(data);
        }
      }

      if (briefRes.status === "fulfilled") {
        const data = briefRes.value;
        if (data?.headline && data?.source_report) {
          setLiveSucafinaBrief(data);
        }
      }
    }

    loadLiveData();
    // Refresh every 5 minutes to pick up newly generated files
    const interval = setInterval(loadLiveData, 5 * 60 * 1000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  // Derive display data: live when available, static otherwise
  const displayContracts = liveContracts
    ? liveContracts.map((c) => ({
        month: symbolToMonth(c.symbol),
        symbol: c.symbol,
        price: Number(c.last_price).toFixed(2),
        change:
          (Number(c.price_change) >= 0 ? "+" : "") +
          Number(c.price_change).toFixed(2),
        pct:
          (Number(c.price_change_pct) >= 0 ? "+" : "") +
          Number(c.price_change_pct).toFixed(2) +
          "%",
        volume: formatK(Number(c.volume)),
        openInterest: formatK(Number(c.open_interest)),
      }))
    : contractsUnavailable
      ? staticContracts.map((contract) => ({
          ...contract,
          month: "N/A",
          symbol: "N/A",
          price: "N/A",
          change: "N/A",
          pct: "N/A",
          volume: "N/A",
          openInterest: "N/A",
        }))
      : staticContracts;

  const contractsPerPage = 4;
  const totalContractPages = Math.ceil(displayContracts.length / contractsPerPage);
  const pagedContracts = displayContracts.slice(
    contractPage * contractsPerPage,
    contractPage * contractsPerPage + contractsPerPage,
  );

  const displayHeadlines = liveNews
    ? liveNews.map((n) => ({
        title: n.text,
        source: n.category,
        date: formatNewsDate(n.timestamp),
        url: n.url || null,
      }))
    : staticHeadlines.map((h) => ({ ...h, url: null }));

  const isLiveData = Boolean(
    liveContracts && liveContracts.length > 0 &&
    liveNews && liveNews.length > 0 &&
    liveSnapshot && liveSnapshot.frontPrice,
  );

  const sucafinaSummaryText = liveSucafinaBrief
    ? `${liveSucafinaBrief.market_bias}. ${liveSucafinaBrief.key_takeaways[0] ?? ""}`.trim()
    : staticSucafinaSummary.summary;

  const displayTodayItems: TodayFeedItem[] = [
    ...displayHeadlines.slice(0, 2).map((item) => ({
      kind: "news" as const,
      title: item.title,
      source: item.source,
      date: item.date,
      url: item.url,
    })),
    liveSucafinaBrief
      ? {
          kind: "sucafina" as const,
          title: liveSucafinaBrief.headline,
          source: liveSucafinaBrief.source_report,
          date: formatNewsDate(liveSucafinaBrief.generated_at ?? ""),
          url: liveSucafinaBrief.report_url,
          summary: sucafinaSummaryText,
        }
      : staticSucafinaSummary,
  ];

  const displayStats = liveSnapshot
    ? [
        {
          label: "Front",
          value: Number(liveSnapshot.frontPrice).toFixed(2),
          sub: "US¢/lb",
          featured: true,
        },
        {
          label: "Shape",
          value: liveSnapshot.curveShape,
          sub:
            liveSnapshot.curveShape === "Contango"
              ? "Deferred > spot"
              : "Spot > deferred",
          featured: false,
        },
        {
          label: "Vol",
          value: formatK(liveSnapshot.totalVolume),
          sub: "Shown",
          featured: false,
        },
        {
          label: "OI",
          value: formatK(liveSnapshot.totalOpenInterest),
          sub: "Shown",
          featured: false,
        },
      ]
    : staticStats;

  const visibleHistory = useMemo(() => {
    if (history.length === 0) {
      return [] as HistoryRow[];
    }

    const latestHistoryDate = parseLocalDate(history[history.length - 1].date);
    const cutoffDate = new Date(latestHistoryDate);
    cutoffDate.setFullYear(cutoffDate.getFullYear() - 1);

    const filteredHistory = history.filter((row) => parseLocalDate(row.date) >= cutoffDate);
    return filteredHistory.length > 0 ? filteredHistory : history;
  }, [history]);

  const chart = useMemo(() => {
    const width = 920;
    const height = 360;

    const left = 72;
    const right = 18;
    const top = 14;
    const bottom = 52;

    if (visibleHistory.length === 0 || forecastPath.length === 0) {
      return null;
    }

    const historyDates = visibleHistory.map((row) => parseLocalDate(row.date));
    const forecastDates = forecastPath.map((row) => parseLocalDate(row.date));
    const allDates = [...historyDates, ...forecastDates];

    const monthKeys = Array.from(
      new Set(
        allDates.map((date) => `${date.getFullYear()}-${date.getMonth()}`),
      ),
    );
    const monthKeyToIndex = new Map(
      monthKeys.map((key, index) => [key, index] as const),
    );

    const msPerWeek = 7 * 24 * 60 * 60 * 1000;
    const asOfDate = parseLocalDate(forecastPath[0].asOfDate);
    const currentPrice = visibleHistory[visibleHistory.length - 1].price;
    const sigmaWeekly = stddev(
      forecastPath.map((row) => row.predictedWeeklyLogReturn),
    );

    const forecastBands: ForecastBandRow[] = forecastPath.map((row) => {
      const forecastDate = parseLocalDate(row.date);
      const weeksElapsed = Math.max(
        (forecastDate.getTime() - asOfDate.getTime()) / msPerWeek,
        0,
      );
      const coneHalf =
        currentPrice * (Math.exp(sigmaWeekly * Math.sqrt(weeksElapsed)) - 1);

      return {
        date: forecastDate,
        projectedPrice: row.projectedPrice,
        upper: row.projectedPrice + coneHalf,
        lower: Math.max(row.projectedPrice - coneHalf, 1),
      };
    });

    const priceValues = [
      ...visibleHistory.map((row) => row.price),
      ...forecastPath.map((row) => row.projectedPrice),
      ...forecastBands.flatMap((band) => [band.upper, band.lower]),
    ];

    const minPrice = Math.min(...priceValues);
    const maxPrice = Math.max(...priceValues);
    const padding = (maxPrice - minPrice || 1) * 0.08;
    const paddedMin = minPrice - padding;
    const paddedMax = maxPrice + padding;
    const priceRange = paddedMax - paddedMin || 1;

    const innerWidth = width - left - right;
    const innerHeight = height - top - bottom;

    const toMonthX = (date: Date) => {
      const monthKey = `${date.getFullYear()}-${date.getMonth()}`;
      const monthIndex = monthKeyToIndex.get(monthKey) ?? 0;
      const daysInMonth = new Date(date.getFullYear(), date.getMonth() + 1, 0).getDate();
      const monthFraction =
        (date.getDate() - 1 +
          date.getHours() / 24 +
          date.getMinutes() / (24 * 60) +
          date.getSeconds() / (24 * 60 * 60)) /
        Math.max(daysInMonth, 1);

      return left + ((monthIndex + monthFraction) / Math.max(monthKeys.length, 1)) * innerWidth;
    };

    const toY = (price: number) =>
      top + (1 - (price - paddedMin) / priceRange) * innerHeight;

    const historyPoints: ChartPoint[] = visibleHistory.map((row) => {
      const date = parseLocalDate(row.date);

      return {
        date,
        x: toMonthX(date),
        y: toY(row.price),
        label: date.toLocaleDateString("en-US", { month: "short" }),
        projectedPrice: row.price,
      };
    });

    const forecastPoints: ChartPoint[] = forecastPath.map((row) => {
      const date = parseLocalDate(row.date);

      return {
        date,
        x: toMonthX(date),
        y: toY(row.projectedPrice),
        label: date.toLocaleDateString("en-US", { month: "short" }),
        projectedPrice: row.projectedPrice,
      };
    });

    const bandPolygon = [
      ...forecastBands.map((band) => `${toMonthX(band.date)},${toY(band.upper)}`),
      ...forecastBands
        .slice()
        .reverse()
        .map((band) => `${toMonthX(band.date)},${toY(band.lower)}`),
    ].join(" ");

    const monthTicks: MonthTick[] = monthKeys.map((key, index) => {
      const [yearString, monthString] = key.split("-");
      const year = Number(yearString);
      const month = Number(monthString);
      const date = new Date(year, month, 1);

      return {
        key,
        x: left + (((index + 0.5) / Math.max(monthKeys.length, 1)) * innerWidth),
        label: date.toLocaleDateString("en-US", { month: "short" }),
      };
    });

    const filteredTicks: MonthTick[] = [];
    let lastAcceptedX = -Infinity;
    const minimumGap = 46;

    monthTicks.forEach((tick, index) => {
      const isLast = index === monthTicks.length - 1;
      if (tick.x - lastAcceptedX >= minimumGap || isLast) {
        filteredTicks.push(tick);
        lastAcceptedX = tick.x;
      }
    });

    const yAxisValues = buildYAxisTicks(paddedMin, paddedMax, 5);
    const yAxisTicks: YAxisTick[] = yAxisValues.map((value) => ({
      value,
      y: toY(value),
      label: value.toFixed(0),
    }));

    const futureProjectionValue = forecastPath[forecastPath.length - 1].projectedPrice;
    const futureProjectionY = toY(futureProjectionValue);

    // Calculate mean prices
    // 1-month rolling mean: use last 30 days of history (~20 trading days)
    const historyMeanPrice =
      visibleHistory.length > 0
        ? visibleHistory
            .slice(-30)
            .reduce((sum, row) => sum + row.price, 0) /
          Math.min(30, visibleHistory.length)
        : 0;
    const historyMeanY = toY(historyMeanPrice);

    const forecastMeanPrice =
      forecastPath.length > 0
        ? forecastPath.reduce((sum, row) => sum + row.projectedPrice, 0) /
          forecastPath.length
        : 0;
    const forecastMeanY = toY(forecastMeanPrice);

    return {
      width,
      height,
      left,
      right,
      top,
      bottom,
      historyPolyline: buildPolyline(historyPoints),
      forecastPolyline: buildPolyline(forecastPoints),
      bandPolygon,
      historyPoints,
      forecastPoints,
      futureProjectionY,
      futureProjectionValue,
      historyMeanY,
      historyMeanPrice,
      forecastMeanY,
      forecastMeanPrice,
      monthTicks: filteredTicks,
      yAxisTicks,
      dividerX: historyPoints[historyPoints.length - 1].x,
      plotBottom: height - bottom,
      plotRight: width - right,
      toMonthX,
    };
  }, [visibleHistory, forecastPath]);

  const hoveredPoint = chart
    ? hoveredHistoryIndex !== null
      ? chart.historyPoints[hoveredHistoryIndex]
      : hoveredIndex !== null
        ? chart.forecastPoints[hoveredIndex]
        : null
    : null;

  const chartDownloadCsv = useMemo(() => {
    if (visibleHistory.length === 0 && forecastPath.length === 0) {
      return "";
    }

    const lines = [
      "series,date,price,asOfDate,stepWeek,predictedWeeklyLogReturn,anchorWeeklyLogReturn,raw1wLogReturn",
      ...visibleHistory.map(
        (row) =>
          `history,${row.date},${row.price.toFixed(6)},,,,,`,
      ),
      ...forecastPath.map(
        (row) =>
          `forecast,${row.date},${row.projectedPrice.toFixed(6)},${row.asOfDate},${row.stepWeek},${row.predictedWeeklyLogReturn.toFixed(8)},${row.anchorWeeklyLogReturn.toFixed(8)},${row.raw1wLogReturn.toFixed(8)}`,
      ),
    ];

    return lines.join("\n");
  }, [visibleHistory, forecastPath]);

  const handleDownloadChartData = () => {
    if (!chartDownloadCsv) {
      return;
    }

    const blob = new Blob([chartDownloadCsv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const timestamp = new Date().toISOString().slice(0, 10);
    anchor.href = url;
    anchor.download = `coffee_chart_data_${timestamp}.csv`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const downloadApiJson = async (apiPath: string, filePrefix: string) => {
    const response = await fetch(apiPath, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load ${apiPath} (${response.status})`);
    }

    const payload = await response.json();
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    const timestamp = new Date().toISOString().slice(0, 10);
    anchor.href = url;
    anchor.download = `${filePrefix}_${timestamp}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const handleDownloadLiveContracts = async () => {
    try {
      await downloadApiJson("/api/contracts", "contracts_live");
    } catch (error) {
      setDataError(error instanceof Error ? error.message : "Failed to download live contracts.");
    }
  };

  const handleDownloadLiveSnapshot = async () => {
    try {
      await downloadApiJson("/api/snapshot", "snapshot_live");
    } catch (error) {
      setDataError(error instanceof Error ? error.message : "Failed to download live snapshot.");
    }
  };

  const todayHeaderDate = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date());

  return (
    <div className="min-h-screen bg-[var(--page-bg)] text-[var(--ink)]">
      <div className="mx-auto max-w-6xl px-4 py-5 sm:px-6 lg:px-8">
        <section className="space-y-4 rounded-[28px] border border-[var(--line)] bg-[var(--panel)] p-4 shadow-[0_24px_80px_rgba(32,44,102,0.08)] sm:p-5">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div className="max-w-3xl flex-1">
              <div className="flex flex-col gap-4">
                <div>
                  <h1 className="mt-3 max-w-xl text-2xl font-semibold tracking-tight text-[var(--bond-blue)] sm:text-3xl">
                    Great Lakes Coffee Futures
                  </h1>
                  <p className="mt-2 max-w-lg text-sm leading-6 text-[var(--muted)]">
                    ICE Arabia Coffee Futures
                  </p>
                </div>
              </div>
            </div>

            <div className="flex h-16 shrink-0 items-center rounded-2xl border border-[var(--line)] bg-white px-4 shadow-[0_8px_24px_rgba(32,44,102,0.06)] sm:h-20 sm:px-5">
              <Image
                src={bondLogoNavy}
                alt="Bond Consulting"
                width={320}
                height={96}
                className="h-10 w-auto object-contain sm:h-12"
                priority
              />
            </div>
          </div>

          <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
            <div className="rounded-3xl border border-[var(--line)] bg-[var(--baby-blue)]/25 p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-medium text-[var(--bond-blue)]">
                    Price Projection
                  </h2>
                </div>

                <button
                  onClick={handleDownloadChartData}
                  disabled={!chartDownloadCsv}
                  className="rounded-full border border-[var(--line-strong)] bg-white px-3 py-1.5 text-xs font-medium text-[var(--bond-blue)] transition hover:bg-[var(--baby-blue)]/40 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Download chart CSV
                </button>
              </div>

              <div className="rounded-2xl border border-[var(--line)] bg-white p-2">
                <div className="relative">
                  {chart && hoveredPoint && (
                    <div
                      className="pointer-events-none absolute z-10 rounded-xl border border-[var(--bond-blue)]/15 bg-[var(--baby-blue)]/85 px-3 py-2 text-xs shadow-[0_10px_20px_rgba(123,159,188,0.18)] backdrop-blur-sm"
                      style={{
                        left: `${(hoveredPoint.x / chart.width) * 100}%`,
                        top: 8,
                        transform: "translateX(-50%)",
                      }}
                    >
                      <div className="font-medium text-[var(--bond-blue)]">
                        {hoveredPoint.date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                      </div>
                      <div className="mt-0.5 text-[var(--muted)]">
                        {hoveredPoint.projectedPrice.toFixed(2)} US¢/lb
                      </div>
                    </div>
                  )}

                  {!chart ? (
                    <div className="flex h-[320px] items-center justify-center rounded-2xl border border-dashed border-[var(--line)] text-sm text-[var(--muted)]">
                      {isLoading
                        ? "Loading chart data..."
                        : dataError ?? "No chart data available."}
                    </div>
                  ) : (
                    <svg
                      viewBox={`0 0 ${chart.width} ${chart.height}`}
                      className="block h-auto w-full"
                      preserveAspectRatio="xMidYMid meet"
                    >
                      {chart.yAxisTicks.map((tick) => (
                        <g key={`y-${tick.value}`}>
                          <line
                            x1={chart.left}
                            y1={tick.y}
                            x2={chart.plotRight}
                            y2={tick.y}
                            stroke="rgba(163, 171, 186, 0.45)"
                            strokeWidth="1.2"
                          />
                          <text
                            x={chart.left - 10}
                            y={tick.y + 4}
                            textAnchor="end"
                            fontSize="13"
                            fontWeight="500"
                            fill="var(--muted)"
                          >
                            {tick.label}
                          </text>
                        </g>
                      ))}

                      <line
                        x1={chart.left}
                        y1={chart.top}
                        x2={chart.left}
                        y2={chart.plotBottom}
                        stroke="rgba(32,44,102,0.18)"
                        strokeWidth="1.2"
                      />

                      <rect
                        x={chart.left}
                        y={chart.top}
                        width={chart.plotRight - chart.left}
                        height={chart.plotBottom - chart.top}
                        fill="transparent"
                        onMouseEnter={() => {
                          setHoveredIndex(null);
                          setHoveredHistoryIndex(null);
                        }}
                        onMouseLeave={() => {
                          setHoveredIndex(null);
                          setHoveredHistoryIndex(null);
                        }}
                      />

                      <polygon
                        points={chart.bandPolygon}
                        fill="rgba(200, 215, 227, 0.9)"
                      />

                      <line
                        x1={chart.dividerX}
                        y1={chart.top}
                        x2={chart.dividerX}
                        y2={chart.plotBottom}
                        stroke="currentColor"
                        strokeDasharray="6 6"
                        strokeWidth="1.4"
                        className="text-[var(--bond-blue)]/35"
                      />

                      <polyline
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="4.2"
                        strokeLinejoin="round"
                        strokeLinecap="round"
                        className="text-[var(--bond-blue)]"
                        points={chart.historyPolyline}
                      />

                      <polyline
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="3.8"
                        strokeDasharray="8 6"
                        strokeLinejoin="round"
                        strokeLinecap="round"
                        className="text-[var(--gameday-blue)]"
                        points={chart.forecastPolyline}
                      />

                      <line
                        x1={chart.left}
                        y1={chart.futureProjectionY}
                        x2={chart.plotRight}
                        y2={chart.futureProjectionY}
                        stroke="#059669"
                        strokeWidth="1.5"
                        strokeDasharray="6 4"
                        opacity="0.8"
                      />

                      <line
                        x1={chart.left}
                        y1={chart.historyMeanY}
                        x2={chart.dividerX}
                        y2={chart.historyMeanY}
                        stroke="rgba(32, 44, 102, 0.5)"
                        strokeWidth="1.2"
                        strokeDasharray="3 3"
                        opacity="0.7"
                      />

                      <line
                        x1={chart.dividerX}
                        y1={chart.forecastMeanY}
                        x2={chart.plotRight}
                        y2={chart.forecastMeanY}
                        stroke="rgba(123, 159, 188, 0.5)"
                        strokeWidth="1.2"
                        strokeDasharray="3 3"
                        opacity="0.7"
                      />

                      {chart.historyPoints.map((point, index) => {
                        const isActive = hoveredHistoryIndex === index;

                        return (
                          <g key={`h-${point.date.toISOString()}`}>
                            {isActive && (
                              <circle
                                cx={point.x}
                                cy={point.y}
                                r={5}
                                fill="rgba(32, 44, 102, 1)"
                                stroke="white"
                                strokeWidth={1.6}
                              />
                            )}
                            <circle
                              cx={point.x}
                              cy={point.y}
                              r={10}
                              fill="transparent"
                              onMouseEnter={() => {
                                setHoveredHistoryIndex(index);
                                setHoveredIndex(null);
                              }}
                            />
                          </g>
                        );
                      })}

                      {chart.forecastPoints.map((point, index) => {
                        const isActive = hoveredIndex === index;

                        return (
                          <g key={point.date.toISOString()}>
                            <circle
                              cx={point.x}
                              cy={point.y}
                              r={isActive ? 5.5 : 3.5}
                              fill={
                                isActive
                                  ? "rgba(123, 159, 188, 1)"
                                  : "rgba(123, 159, 188, 0.95)"
                              }
                              stroke={
                                isActive
                                  ? "rgba(32, 44, 102, 1)"
                                  : "transparent"
                              }
                              strokeWidth={isActive ? 1.6 : 0}
                            />
                            <circle
                              cx={point.x}
                              cy={point.y}
                              r={12}
                              fill="transparent"
                              onMouseEnter={() => {
                                setHoveredIndex(index);
                                setHoveredHistoryIndex(null);
                              }}
                            />
                          </g>
                        );
                      })}

                      {chart.monthTicks.map((tick) => (
                        <text
                          key={tick.key}
                          x={tick.x}
                          y={chart.height - 14}
                          textAnchor="middle"
                          fontSize="14"
                          fontWeight="500"
                          fill="var(--muted)"
                        >
                          {tick.label}
                        </text>
                      ))}

                      <text
                        x={16}
                        y={chart.top - 2}
                        fontSize="12"
                        fontWeight="600"
                        fill="var(--muted)"
                      >
                        US¢/lb
                      </text>

                    </svg>
                  )}
                </div>

                <div className="mt-3 flex flex-wrap gap-3 text-[12px] text-[var(--muted)]">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-[var(--bond-blue)]" />
                    Historical coffee_c series
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-[var(--gameday-blue)]" />
                    Recursive weekly forecast path
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="inline-block h-[2px] w-5 rounded-full bg-[#059669]" style={{ borderTop: "2px dashed #059669" }} />
                    Future projection ({chart?.futureProjectionValue.toFixed(2)})
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="inline-block h-[2px] w-5" style={{ borderTop: "1px dashed rgba(32, 44, 102, 0.5)" }} />
                    1-mo mean ({chart?.historyMeanPrice.toFixed(2)})
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="inline-block h-[2px] w-5" style={{ borderTop: "1px dashed rgba(123, 159, 188, 0.5)" }} />
                    Forecast mean ({chart?.forecastMeanPrice.toFixed(2)})
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-[var(--line)] bg-white p-4">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-base font-medium text-[var(--bond-blue)]">
                  Today ({todayHeaderDate})
                </h2>
                <span className="text-xs text-[var(--muted)]">3 notes</span>
              </div>
              <div className="space-y-2.5">
                {displayTodayItems.map((item, index) => {
                  const isSucafina = item.kind === "sucafina";
                  const inner = isSucafina ? (
                    <>
                      <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">
                        <span>{item.source}</span>
                        {item.date ? <span>{item.date}</span> : null}
                      </div>
                      <h3 className="mt-1.5 text-sm font-medium leading-5 text-[var(--ink)]">
                        {item.title}
                      </h3>
                      <p className="mt-1.5 text-sm leading-5 text-[var(--muted)]">
                        {item.summary}
                      </p>
                    </>
                  ) : (
                    <>
                      <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.16em] text-[var(--muted)]">
                        <span>{item.source}</span>
                        {item.date ? <span>{item.date}</span> : null}
                      </div>
                      <h3 className="mt-1.5 text-sm font-medium leading-5 text-[var(--ink)]">
                        {item.title}
                      </h3>
                    </>
                  );

                  const articleClass = isSucafina
                    ? "rounded-2xl border border-[var(--bond-blue)]/14 bg-[var(--baby-blue)]/18 p-3"
                    : `rounded-2xl border p-3 ${
                        index === 0
                          ? "border-[var(--bond-blue)]/12 bg-[var(--bond-blue)]/5"
                          : "border-[var(--line)] bg-[var(--page-bg)]"
                      }`;

                  const href = item.url;

                  return href ? (
                    <a
                      key={item.kind === "sucafina" ? `sucafina-${item.title}` : item.title}
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className={`block ${articleClass} transition hover:border-[var(--bond-blue)]/30 hover:bg-[var(--baby-blue)]/20`}
                    >
                      {inner}
                    </a>
                  ) : (
                    <article
                      key={item.kind === "sucafina" ? `sucafina-${item.title}` : item.title}
                      className={articleClass}
                    >
                      {inner}
                    </article>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-[1.25fr_0.75fr]">
            <div className="rounded-3xl border border-[var(--line)] bg-white p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-medium text-[var(--bond-blue)]">
                    Contracts
                  </h2>
                  <p className="text-xs text-[var(--muted)]">
                    Compact contract cards
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {totalContractPages > 1 && (
                    <>
                      <button
                        type="button"
                        onClick={() => setContractPage((p) => Math.max(0, p - 1))}
                        disabled={contractPage === 0}
                        className="flex h-7 w-7 items-center justify-center rounded-full border border-[var(--line-strong)] bg-white text-[var(--bond-blue)] transition hover:bg-[var(--baby-blue)]/40 disabled:cursor-not-allowed disabled:opacity-40"
                        aria-label="Previous contracts"
                      >
                        ‹
                      </button>
                      <span className="min-w-[2.5rem] text-center text-xs text-[var(--muted)]">
                        {contractPage + 1} / {totalContractPages}
                      </span>
                      <button
                        type="button"
                        onClick={() => setContractPage((p) => Math.min(totalContractPages - 1, p + 1))}
                        disabled={contractPage === totalContractPages - 1}
                        className="flex h-7 w-7 items-center justify-center rounded-full border border-[var(--line-strong)] bg-white text-[var(--bond-blue)] transition hover:bg-[var(--baby-blue)]/40 disabled:cursor-not-allowed disabled:opacity-40"
                        aria-label="Next contracts"
                      >
                        ›
                      </button>
                    </>
                  )}
                  <button
                    type="button"
                    onClick={handleDownloadLiveContracts}
                    className={`rounded-full border px-3 py-1 text-xs font-medium transition hover:border-[var(--bond-blue)]/35 hover:bg-[var(--baby-blue)]/22 ${isLiveData ? "border-green-200 bg-green-50 text-green-700" : contractsUnavailable ? "border-[var(--line)] bg-white text-[var(--muted)]" : "border-[var(--line-strong)] bg-[var(--prasad-purple)]/18 text-[var(--bond-blue)]"}`}
                  >
                    {isLiveData ? "Live" : contractsUnavailable ? "N/A" : "Delayed demo"}
                  </button>
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                {pagedContracts.map((contract, index) => (
                  <article
                    key={contractsUnavailable ? `na-${contractPage * contractsPerPage + index}` : contract.symbol}
                    className={`flex h-full flex-col rounded-2xl border p-3.5 ${
                      index === 0
                        ? "border-[var(--bond-blue)]/16 bg-[var(--bond-blue)] text-white shadow-[0_14px_30px_rgba(32,44,102,0.18)]"
                        : "border-[var(--line)] bg-[var(--page-bg)]"
                    }`}
                  >
                    <div className="flex min-h-[3.25rem] items-start justify-between gap-3">
                      <div>
                        <div
                          className={`text-[10px] uppercase tracking-[0.18em] ${
                            index === 0
                              ? "text-white/70"
                              : "text-[var(--muted)]"
                          }`}
                        >
                          {contract.symbol}
                        </div>
                        <h3
                          className={`mt-1 text-sm font-medium ${
                            index === 0 ? "text-white" : "text-[var(--ink)]"
                          }`}
                        >
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

                    <div
                      className={`mt-4 text-3xl font-semibold tracking-tight tabular-nums ${
                        index === 0
                          ? "text-white"
                          : "text-[var(--bond-blue)]"
                      }`}
                    >
                      {contract.price}
                    </div>
                    <div
                      className={`mt-1 text-xs ${
                        index === 0 ? "text-white/70" : "text-[var(--muted)]"
                      }`}
                    >
                      US¢/lb settlement
                    </div>

                    <div className="mt-4 grid grid-cols-2 gap-x-3 gap-y-2 border-t border-current/10 pt-3 text-xs tabular-nums">
                      <div className="space-y-0.5">
                        <div
                          className={
                            index === 0
                              ? "text-white/65"
                              : "text-[var(--muted)]"
                          }
                        >
                          Change
                        </div>
                        <div
                          className={
                            index === 0
                              ? "text-white"
                              : "text-[var(--bond-blue)]"
                          }
                        >
                          {contract.change}
                        </div>
                      </div>
                      <div className="space-y-0.5">
                        <div
                          className={
                            index === 0
                              ? "text-white/65"
                              : "text-[var(--muted)]"
                          }
                        >
                          Volume
                        </div>
                        <div
                          className={
                            index === 0
                              ? "text-white/85"
                              : "text-[var(--ink)]"
                          }
                        >
                          {contract.volume}
                        </div>
                      </div>
                      <div className="col-span-2 space-y-0.5">
                        <div
                          className={
                            index === 0
                              ? "text-white/65"
                              : "text-[var(--muted)]"
                          }
                        >
                          Open interest
                        </div>
                        <div
                          className={
                            index === 0
                              ? "text-white/85"
                              : "text-[var(--ink)]"
                          }
                        >
                          {contract.openInterest}
                        </div>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </div>

            <div className="rounded-3xl border border-[var(--line)] bg-[linear-gradient(180deg,rgba(123,159,188,0.18),rgba(197,174,203,0.14))] p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-medium text-[var(--bond-blue)]">
                    Market snapshot
                  </h2>
                  <p className="text-xs text-[var(--muted)]">
                    Core futures metrics at a glance
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleDownloadLiveSnapshot}
                  className="rounded-full border border-[var(--line-strong)] bg-white px-3 py-1 text-xs font-medium text-[var(--bond-blue)] transition hover:border-[var(--bond-blue)]/35 hover:bg-[var(--baby-blue)]/22"
                >
                  Live summary
                </button>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3">
                {displayStats.map((stat) => (
                  <div
                    key={stat.label}
                    className={`min-w-0 rounded-2xl border p-4 ${
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
                    <div
                      className={`mt-2 text-xl font-semibold leading-tight tracking-tight sm:text-2xl ${
                        stat.label === "Shape" ? "whitespace-nowrap text-lg sm:text-xl" : ""
                      } ${
                        stat.featured ? "text-white" : "text-[var(--bond-blue)]"
                      }`}
                    >
                      {stat.value}
                    </div>
                    <div
                      className={`mt-1 text-xs ${
                        stat.featured ? "text-white/70" : "text-[var(--muted)]"
                      }`}
                    >
                      {stat.sub}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <footer className="mt-5 rounded-2xl border border-[var(--line)] bg-white px-4 py-3 text-center text-sm text-[var(--muted)]">
          If you have any further questions, please feel free to contact us bondexecutives@umich.edu.
        </footer>
      </div>

      <button
        type="button"
        onClick={() => setIsGlossaryOpen(true)}
        className="fixed bottom-5 right-5 z-30 flex h-12 w-12 items-center justify-center rounded-full border border-[var(--line-strong)] bg-[var(--bond-blue)] text-2xl font-semibold text-white shadow-[0_14px_32px_rgba(32,44,102,0.28)] transition hover:scale-[1.03] hover:bg-[#1a2454] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--gameday-blue)]"
        aria-label="Open glossary"
      >
        ?
      </button>

      {isGlossaryOpen && (
        <div
          className="fixed inset-0 z-40 flex items-end justify-center bg-[rgba(18,25,44,0.42)] p-4 sm:items-center"
          onClick={() => setIsGlossaryOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-label="Coffee futures glossary"
        >
          <div
            className="w-full max-w-2xl rounded-3xl border border-[var(--line)] bg-white p-4 shadow-[0_28px_80px_rgba(15,23,42,0.3)] sm:p-5"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="mb-3 flex items-start justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold text-[var(--bond-blue)]">Market Glossary</h2>
                <p className="mt-1 text-sm text-[var(--muted)]">
                  Quick definitions for commonly used terms on this dashboard.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setIsGlossaryOpen(false)}
                className="rounded-full border border-[var(--line)] px-2.5 py-1 text-xs font-medium text-[var(--bond-blue)] transition hover:bg-[var(--baby-blue)]/28"
                aria-label="Close glossary"
              >
                Close
              </button>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              {glossaryTerms.map((item) => (
                <article
                  key={item.term}
                  className="rounded-2xl border border-[var(--line)] bg-[var(--page-bg)] px-3 py-2.5"
                >
                  <h3 className="text-sm font-semibold text-[var(--bond-blue)]">{item.term}</h3>
                  <p className="mt-1 text-sm leading-5 text-[var(--muted)]">{item.definition}</p>
                </article>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}