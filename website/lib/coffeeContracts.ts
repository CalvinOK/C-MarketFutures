import { getCoffeeMarketData } from '@/lib/coffeeMarketDataService'

export type CoffeeContractsResult = {
  contracts: Array<Record<string, unknown>>
  metadata: {
    source: 'database-cache' | 'database-stale' | 'ice-delayed'
    tradeDate: string | null
    fetchedAt: string | null
    isStale: boolean
  }
}

function newYorkDate(now = new Date()): string {
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(now)
}

export function getExpectedCoffeeTradingDate(now = new Date()): string {
  const date = new Date(`${newYorkDate(now)}T12:00:00Z`)
  const weekday = date.getUTCDay()
  if (weekday === 0) date.setUTCDate(date.getUTCDate() - 2)
  if (weekday === 6) date.setUTCDate(date.getUTCDate() - 1)
  return date.toISOString().slice(0, 10)
}

export function toApiContract(row: {
  symbol: string
  contract_name: string | null
  month_code?: string | null
  expiration_date: string | null
  last_price: number | string | null
  price_change: number | string | null
  percent_change: number | string | null
  volume: number | string | null
  open_interest: number | string | null
  source: string
  trade_date: string
  fetched_at: string
  provider_timestamp: string | null
  settlement?: number | string | null
}): Record<string, unknown> {
  return {
    symbol: row.symbol,
    label: row.contract_name ?? row.symbol,
    expiryDate: row.expiration_date,
    lastPrice: row.last_price == null ? null : Number(row.last_price),
    priceChange: row.price_change == null ? null : Number(row.price_change),
    priceChangePct: row.percent_change == null ? null : Number(row.percent_change),
    volume: row.volume == null ? null : Number(row.volume),
    openInterest: row.open_interest == null ? null : Number(row.open_interest),
    settlementPrice: row.settlement == null ? null : Number(row.settlement),
    source: row.source,
    tradeDate: row.trade_date,
    fetchedAt: row.fetched_at,
    providerTimestamp: row.provider_timestamp,
  }
}

export async function getCoffeeContracts(): Promise<CoffeeContractsResult> {
  const result = await getCoffeeMarketData()
  return {
    contracts: result.contracts,
    metadata: {
      source: result.metadata.source === 'ice-delayed'
        ? 'ice-delayed'
        : result.metadata.source === 'supabase-stale' ? 'database-stale' : 'database-cache',
      tradeDate: result.snapshot.marketDate,
      fetchedAt: result.metadata.retrievedAt,
      isStale: result.metadata.isStale,
    },
  }
}
