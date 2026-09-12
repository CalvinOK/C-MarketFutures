import {
  buildCoffeeMarketSnapshot,
  collectIceCoffeeContracts,
  fetchLatestCoffeeOpenInterest,
  ICE_URL,
  CFTC_URL,
} from '@/scripts/collect-coffee-market-snapshot'
import type { CoffeeMarketSnapshot, IceCoffeeContract } from '@/lib/coffeeMarketSnapshot'
import { getLatestCoffeeMarketSnapshot, upsertCoffeeMarketSnapshot } from '@/lib/supabaseServer'
import {
  completeRefresh,
  getRefreshState,
  isRefreshCooldownActive,
  tryClaimRefresh,
  waitForRefresh,
} from '@/lib/refreshState'

export const ICE_PRICE_TTL_MS = 15 * 60 * 1000
const DATASET = 'market_snapshot_hourly'

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
  raw_metadata: { sourceUrl?: string | null } | null
}

export type CoffeeMarketDataResult = {
  contracts: Array<Record<string, unknown>>
  snapshot: NonNullable<Awaited<ReturnType<typeof getLatestCoffeeMarketSnapshot>>>
  metadata: {
    source: 'supabase-cache' | 'supabase-stale' | 'ice-delayed'
    retrievedAt: string
    ageMinutes: number
    ttlMinutes: 15
    isStale: boolean
    sources: {
      ice: { name: 'ICE'; url: string; fields: string[] }
      cftc: { name: 'CFTC'; url: string; fields: string[]; asOf: string }
    }
  }
}

function config(): { url: string; key: string } {
  const url = process.env.SUPABASE_URL?.replace(/\/$/, '')
  const key = process.env.SUPABASE_SECRET_KEY
  if (!url || !key) throw new Error('SUPABASE_URL and SUPABASE_SECRET_KEY are required')
  return { url, key }
}

function headers(prefer?: string): HeadersInit {
  const { key } = config()
  return { Accept: 'application/json', apikey: key, Authorization: `Bearer ${key}`, ...(prefer ? { Prefer: prefer } : {}) }
}

function ageMinutes(retrievedAt: string): number {
  return Math.max(0, Math.floor((Date.now() - Date.parse(retrievedAt)) / 60_000))
}

function isFresh(retrievedAt: string | null | undefined): boolean {
  return Boolean(retrievedAt && Date.now() - Date.parse(retrievedAt) < ICE_PRICE_TTL_MS)
}

async function readContracts(): Promise<ContractRow[]> {
  const { url } = config()
  const params = new URLSearchParams({ select: '*', order: 'fetched_at.desc,trade_date.desc,expiration_date.asc,symbol.asc', limit: '100' })
  const response = await fetch(`${url}/rest/v1/coffee_contract_snapshots?${params}`, { headers: headers(), cache: 'no-store' })
  if (!response.ok) throw new Error(`Supabase ICE contract read returned HTTP ${response.status}`)
  const rows = (await response.json()) as ContractRow[]
  const symbols = new Set<string>()
  return rows.filter((row) => !symbols.has(row.symbol) && symbols.add(row.symbol))
}

function contractResult(rows: ContractRow[]): Array<Record<string, unknown>> {
  return rows.map((row) => ({
    symbol: row.symbol,
    label: row.contract_name ?? row.symbol,
    expiryDate: row.expiration_date,
    lastPrice: row.last_price == null ? null : Number(row.last_price),
    priceChange: row.price_change == null ? null : Number(row.price_change),
    priceChangePct: row.percent_change == null ? null : Number(row.percent_change),
    volume: row.volume == null ? null : Number(row.volume),
    openInterest: null,
    settlementPrice: null,
    source: row.source,
    tradeDate: row.trade_date,
    fetchedAt: row.fetched_at,
    providerTimestamp: row.provider_timestamp,
    sourceName: row.source === 'ice' ? 'ICE' : row.source,
    sourceUrl: row.raw_metadata?.sourceUrl ?? (row.source === 'ice' ? ICE_URL : null),
  }))
}

async function persistContracts(snapshot: CoffeeMarketSnapshot, contracts: IceCoffeeContract[]): Promise<void> {
  const { url } = config()
  const retrievedAt = snapshot.retrievedAt ?? new Date().toISOString()
  const rows = contracts.flatMap((contract) => {
    if (!contract.symbol) return []
    return [{
      trade_date: snapshot.marketDate,
      symbol: contract.symbol,
      contract_name: contract.contract,
      month_code: contract.symbol.slice(2, 3),
      expiration_date: null,
      settlement: null,
      last_price: contract.price,
      price_change: contract.priceChange ?? null,
      percent_change: contract.percentChange ?? null,
      volume: contract.volume,
      open_interest: null,
      provider_timestamp: contract.priceAsOf,
      fetched_at: retrievedAt,
      updated_at: retrievedAt,
      source: 'ice',
      raw_metadata: { provider: 'ice-delayed', label: contract.contract, sourceName: contract.sourceName ?? 'ICE', sourceUrl: contract.sourceUrl ?? null },
    }]
  })
  if (rows.length < 2) throw new Error('ICE returned fewer than two valid Coffee C contracts')
  const response = await fetch(`${url}/rest/v1/coffee_contract_snapshots?on_conflict=trade_date,symbol,source`, {
    method: 'POST',
    headers: { ...headers('resolution=merge-duplicates,return=minimal'), 'Content-Type': 'application/json' },
    body: JSON.stringify(rows),
    cache: 'no-store',
  })
  if (!response.ok) throw new Error(`Supabase ICE contract upsert returned HTTP ${response.status}`)
}

function makeResult(rows: ContractRow[], snapshot: NonNullable<Awaited<ReturnType<typeof getLatestCoffeeMarketSnapshot>>>, source: CoffeeMarketDataResult['metadata']['source'], stale: boolean): CoffeeMarketDataResult {
  const retrievedAt = snapshot.retrievedAt ?? new Date(0).toISOString()
  return {
    contracts: contractResult(rows),
    snapshot,
    metadata: {
      source,
      retrievedAt,
      ageMinutes: ageMinutes(retrievedAt),
      ttlMinutes: 15,
      isStale: stale,
      sources: {
        ice: { name: 'ICE', url: ICE_URL, fields: ['frontPrice', 'frontContract', 'nextPrice', 'nextContract', 'shape', 'volume'] },
        cftc: { name: 'CFTC', url: CFTC_URL, fields: ['openInterest'], asOf: snapshot.openInterestAsOf },
      },
    },
  }
}

async function readCachedData(): Promise<{ rows: ContractRow[]; snapshot: Awaited<ReturnType<typeof getLatestCoffeeMarketSnapshot>> }> {
  const [rows, snapshot] = await Promise.all([readContracts(), getLatestCoffeeMarketSnapshot()])
  return { rows, snapshot }
}

export async function getCoffeeMarketData(): Promise<CoffeeMarketDataResult> {
  const cached = await readCachedData()
  if (cached.snapshot && isFresh(cached.snapshot.retrievedAt) && cached.rows.length >= 2 && cached.rows.every((row) => row.source === 'ice' && isFresh(row.fetched_at))) {
    console.log(`[coffee-market] cache hit age_minutes=${ageMinutes(cached.snapshot.retrievedAt ?? '')}`)
    return makeResult(cached.rows, cached.snapshot, 'supabase-cache', false)
  }

  const state = await getRefreshState(DATASET)
  if (isRefreshCooldownActive(state)) {
    console.log('[coffee-market] cooldown active')
    if (cached.snapshot) return makeResult(cached.rows, cached.snapshot, 'supabase-stale', true)
    throw new Error('Coffee market provider is temporarily unavailable')
  }

  const token = await tryClaimRefresh(DATASET)
  if (!token) {
    const waited = await waitForRefresh(readCachedData, (data) => Boolean(data.snapshot && isFresh(data.snapshot.retrievedAt)))
    if (waited.snapshot) return makeResult(waited.rows, waited.snapshot, isFresh(waited.snapshot.retrievedAt) ? 'supabase-cache' : 'supabase-stale', !isFresh(waited.snapshot.retrievedAt))
    throw new Error('Coffee market refresh is already in progress')
  }

  try {
    const rechecked = await readCachedData()
    if (rechecked.snapshot && isFresh(rechecked.snapshot.retrievedAt) && rechecked.rows.length >= 2 && rechecked.rows.every((row) => row.source === 'ice' && isFresh(row.fetched_at))) {
      await completeRefresh(DATASET, token, true)
      return makeResult(rechecked.rows, rechecked.snapshot, 'supabase-cache', false)
    }

    console.log('[coffee-market] ICE refresh started')
    const contracts = await collectIceCoffeeContracts()
    const openInterest = rechecked.snapshot
      ? { openInterest: rechecked.snapshot.openInterest, openInterestAsOf: rechecked.snapshot.openInterestAsOf }
      : await fetchLatestCoffeeOpenInterest()
    const refreshed = buildCoffeeMarketSnapshot(contracts, openInterest)
    await persistContracts(refreshed, contracts)
    await upsertCoffeeMarketSnapshot(refreshed)
    await completeRefresh(DATASET, token, true)
    const saved = await readCachedData()
    if (!saved.snapshot) throw new Error('ICE snapshot was not persisted')
    console.log(`[coffee-market] ICE refresh stored contracts=${contracts.length}`)
    return makeResult(saved.rows, saved.snapshot, 'ice-delayed', false)
  } catch (error) {
    await completeRefresh(DATASET, token, false, error)
    console.error('[coffee-market] ICE refresh failed; serving stale:', error instanceof Error ? error.message : String(error))
    if (cached.snapshot) return makeResult(cached.rows, cached.snapshot, 'supabase-stale', true)
    throw error
  }
}