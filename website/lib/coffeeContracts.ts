import { fetchCoffeeContracts, getExpiryDate, type ContractData } from '@/lib/external/databento'
import {
  completeRefresh,
  getRefreshState,
  isRefreshCooldownActive,
  tryClaimRefresh,
  waitForRefresh,
} from '@/lib/refreshState'

const DATASET = 'contracts_daily'

type ContractRow = {
  symbol: string
  contract_name: string | null
  month_code: string | null
  expiration_date: string | null
  settlement: number | string | null
  last_price: number | string | null
  price_change: number | string | null
  percent_change: number | string | null
  volume: number | string | null
  open_interest: number | string | null
  provider_timestamp: string | null
  source: string
  trade_date: string
  fetched_at: string
}

export type CoffeeContractsResult = {
  contracts: Array<Record<string, unknown>>
  metadata: {
    source: 'database-cache' | 'database-stale' | 'databento'
    tradeDate: string | null
    fetchedAt: string | null
    isStale: boolean
  }
}

function supabaseConfig(): { url: string; key: string } {
  const url = process.env.SUPABASE_URL?.replace(/\/$/, '')
  const key = process.env.SUPABASE_SECRET_KEY
  if (!url || !key) throw new Error('SUPABASE_URL and SUPABASE_SECRET_KEY are required')
  return { url, key }
}

function supabaseHeaders(prefer?: string): HeadersInit {
  const { key } = supabaseConfig()
  return { Accept: 'application/json', apikey: key, Authorization: `Bearer ${key}`, ...(prefer ? { Prefer: prefer } : {}) }
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

function isWeekend(now = new Date()): boolean {
  const weekday = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', weekday: 'short' }).format(now)
  return weekday === 'Sat' || weekday === 'Sun'
}

async function getContractsForDate(tradeDate: string): Promise<ContractRow[]> {
  const { url } = supabaseConfig()
  const params = new URLSearchParams({ select: '*', trade_date: `eq.${tradeDate}`, order: 'expiration_date.asc,symbol.asc' })
  const response = await fetch(`${url}/rest/v1/coffee_contract_snapshots?${params}`, { headers: supabaseHeaders(), cache: 'no-store' })
  if (!response.ok) throw new Error(`Supabase contract cache read returned HTTP ${response.status}`)
  return (await response.json()) as ContractRow[]
}

async function getLatestContracts(): Promise<ContractRow[]> {
  const { url } = supabaseConfig()
  const params = new URLSearchParams({ select: '*', order: 'trade_date.desc,expiration_date.asc,symbol.asc', limit: '100' })
  const response = await fetch(`${url}/rest/v1/coffee_contract_snapshots?${params}`, { headers: supabaseHeaders(), cache: 'no-store' })
  if (!response.ok) throw new Error(`Supabase contract cache read returned HTTP ${response.status}`)
  const rows = (await response.json()) as ContractRow[]
  const symbols = new Set<string>()
  return rows.filter((row) => !symbols.has(row.symbol) && symbols.add(row.symbol))
}

export function toApiContract(row: ContractRow): Record<string, unknown> {
  const lastPrice = row.last_price == null ? null : Number(row.last_price)
  const priceChange = row.price_change == null ? null : Number(row.price_change)
  return {
    symbol: row.symbol,
    label: row.contract_name ?? row.symbol,
    expiryDate: row.expiration_date,
    lastPrice,
    priceChange,
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

function result(rows: ContractRow[], source: CoffeeContractsResult['metadata']['source'], expectedDate: string): CoffeeContractsResult {
  return {
    contracts: rows.map(toApiContract),
    metadata: {
      source,
      tradeDate: rows[0]?.trade_date ?? expectedDate,
      fetchedAt: rows.reduce<string | null>((latest, row) => !latest || row.fetched_at > latest ? row.fetched_at : latest, null),
      isStale: source === 'database-stale',
    },
  }
}

async function persistContracts(tradeDate: string, contracts: ContractData[], fetchedAt: string): Promise<number> {
  const { url } = supabaseConfig()
  const rows = contracts.flatMap((contract) => {
    const expirationDate = getExpiryDate(contract.symbol)
    if (!expirationDate) return []
    const priceChange = contract.priceChange ?? 0
    const price = contract.lastPrice ?? 0
    return [{
      trade_date: tradeDate,
      symbol: contract.symbol,
      contract_name: contract.symbol,
      month_code: contract.symbol[2] ?? null,
      expiration_date: expirationDate,
      settlement: price,
      last_price: price,
      price_change: priceChange,
      percent_change: price === 0 ? 0 : (priceChange / price) * 100,
      volume: contract.volume ?? 0,
      open_interest: contract.openInterest ?? 0,
      provider_timestamp: fetchedAt,
      fetched_at: fetchedAt,
      updated_at: fetchedAt,
      source: contract.source,
      raw_metadata: { provider: 'databento' },
    }]
  })
  if (rows.length === 0) throw new Error('Databento returned no valid Coffee C contracts')
  const response = await fetch(`${url}/rest/v1/coffee_contract_snapshots?on_conflict=trade_date,symbol,source`, {
    method: 'POST',
    headers: { ...supabaseHeaders('resolution=merge-duplicates,return=minimal'), 'Content-Type': 'application/json' },
    body: JSON.stringify(rows),
    cache: 'no-store',
  })
  if (!response.ok) throw new Error(`Supabase contract cache upsert returned HTTP ${response.status}`)
  return rows.length
}

async function refreshContracts(expectedDate: string): Promise<CoffeeContractsResult> {
  const current = await getContractsForDate(expectedDate)
  if (current.length > 0) return result(current, 'database-cache', expectedDate)
  const state = await getRefreshState(DATASET)
  if (isRefreshCooldownActive(state)) return result(await getLatestContracts(), 'database-stale', expectedDate)

  const token = await tryClaimRefresh(DATASET)
  if (!token) {
    const waited = await waitForRefresh(() => getContractsForDate(expectedDate), (rows) => rows.length > 0)
    return result(waited.length > 0 ? waited : await getLatestContracts(), waited.length > 0 ? 'database-cache' : 'database-stale', expectedDate)
  }

  try {
    const rechecked = await getContractsForDate(expectedDate)
    if (rechecked.length > 0) {
      await completeRefresh(DATASET, token, true)
      return result(rechecked, 'database-cache', expectedDate)
    }
    console.log('[coffee-contracts] databento refresh started')
    const fetchedAt = new Date().toISOString()
    const count = await persistContracts(expectedDate, await fetchCoffeeContracts(), fetchedAt)
    await completeRefresh(DATASET, token, true)
    console.log(`[coffee-contracts] stored contracts count=${count}`)
    return result(await getContractsForDate(expectedDate), 'databento', expectedDate)
  } catch (error) {
    await completeRefresh(DATASET, token, false, error)
    console.error(`[coffee-contracts] databento refresh failed: ${error instanceof Error ? error.message : String(error)}`)
    const stale = await getLatestContracts()
    if (stale.length > 0) return result(stale, 'database-stale', expectedDate)
    throw error
  }
}

export async function getCoffeeContracts(): Promise<CoffeeContractsResult> {
  const expectedDate = getExpectedCoffeeTradingDate()
  const current = await getContractsForDate(expectedDate)
  if (current.length > 0) return result(current, 'database-cache', expectedDate)
  const stale = await getLatestContracts()
  if (isWeekend() && stale.length > 0) return result(stale, 'database-stale', expectedDate)
  console.log('[coffee-contracts] cache miss')
  return refreshContracts(expectedDate)
}

export type { ContractData }
