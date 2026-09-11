import type { CoffeeMarketSnapshot } from '@/lib/coffeeMarketSnapshot'

type StoredCoffeeMarketSnapshotRow = {
  id: string
  market_date: string
  front_contract: string
  front_price: number | string
  next_contract: string
  next_price: number | string
  shape: CoffeeMarketSnapshot['shape']
  spread: number | string
  volume: number | string
  open_interest: number | string
  open_interest_as_of: string
  price_as_of: string | null
  unit: 'US¢/lb'
  retrieved_at: string
  created_at: string
}

export type CoffeeMarketHistoryPoint = {
  date: string
  price: number
}

export type CoffeeMarketHistoryRow = CoffeeMarketHistoryPoint & {
  open: number
  high: number
  low: number
  volume: string
  changePercent: string
  source?: string | null
  sourceContract?: string | null
  sourceInstrumentId?: string | null
  sourceRetrievedAt?: string | null
}

const HISTORICAL_TABLE = 'Coffee C Historical Data'
const SUPABASE_PAGE_SIZE = 1000

function getSupabaseConfig(): { url: string; key: string } {
  const url = process.env.SUPABASE_URL?.replace(/\/$/, '')
  const key = process.env.SUPABASE_SECRET_KEY
  if (!url || !key) throw new Error('SUPABASE_URL and SUPABASE_SECRET_KEY are required')
  return { url, key }
}

function headers(prefer?: string): HeadersInit {
  const { key } = getSupabaseConfig()
  return {
    Accept: 'application/json',
    apikey: key,
    Authorization: `Bearer ${key}`,
    ...(prefer ? { Prefer: prefer } : {}),
  }
}

export async function getLatestCoffeeMarketSnapshot(): Promise<CoffeeMarketSnapshot | null> {
  const { url } = getSupabaseConfig()
  const response = await fetch(
    `${url}/rest/v1/coffee_market_snapshots?select=*&order=market_date.desc,retrieved_at.desc&limit=1`,
    { headers: headers(), cache: 'no-store' },
  )
  if (!response.ok) throw new Error(`Supabase snapshot read returned HTTP ${response.status}`)
  const row = ((await response.json()) as StoredCoffeeMarketSnapshotRow[])[0]
  if (!row) return null
  return {
    front: Number(row.front_price),
    frontContract: row.front_contract,
    nextContract: row.next_contract,
    nextPrice: Number(row.next_price),
    shape: row.shape,
    spread: Number(row.spread),
    volume: Number(row.volume),
    openInterest: Number(row.open_interest),
    openInterestAsOf: row.open_interest_as_of,
    priceAsOf: row.price_as_of,
    marketDate: row.market_date,
    unit: row.unit,
    retrievedAt: row.retrieved_at,
  }
}

export async function getCoffeeMarketHistory(options?: {
  from?: string
  to?: string
  limit?: number
}): Promise<CoffeeMarketHistoryPoint[]> {
  const { url } = getSupabaseConfig()
  const rows: CoffeeMarketHistoryRow[] = []
  const requestedLimit = Math.min(Math.max(options?.limit ?? 5000, 1), 5000)
  const fetchLimit = 5000

  for (let offset = 0; offset < fetchLimit; offset += SUPABASE_PAGE_SIZE) {
    const params = new URLSearchParams({
      select: '"Date","Price"',
      limit: String(Math.min(SUPABASE_PAGE_SIZE, fetchLimit - offset)),
      offset: String(offset),
    })
    const response = await fetch(
      `${url}/rest/v1/${encodeURIComponent(HISTORICAL_TABLE)}?${params.toString()}`,
      { headers: headers(), cache: 'no-store' },
    )
    if (!response.ok) throw new Error(`Supabase history read returned HTTP ${response.status}`)

    const page = (await response.json()) as Array<Record<string, unknown>>
    rows.push(...page.flatMap(parseHistoricalPriceRow))
    if (page.length < Math.min(SUPABASE_PAGE_SIZE, fetchLimit - offset)) break
  }

  return rows
    .filter((row) => (!options?.from || row.date >= options.from) && (!options?.to || row.date <= options.to))
    .sort((left, right) => left.date.localeCompare(right.date))
    .slice(-requestedLimit)
    .map(({ date, price }) => ({ date, price }))
}

function parseHistoricalPriceRow(row: Record<string, unknown>): CoffeeMarketHistoryRow[] {
  const date = parseHistoricalDate(row.Date ?? row.date)
  const price = Number(row.Price ?? row.price)
  if (!date || !Number.isFinite(price) || price <= 0) return []
  return [{ date, price, open: price, high: price, low: price, volume: '', changePercent: '' }]
}

function parseHistoricalDate(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const text = value.trim()
  const slashMatch = text.match(/^(\d{2})\/(\d{2})\/(\d{4})$/)
  if (slashMatch) return `${slashMatch[3]}-${slashMatch[1]}-${slashMatch[2]}`
  if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return text
  return null
}

function parseHistoricalRow(row: Record<string, unknown>): CoffeeMarketHistoryRow[] {
  const date = parseHistoricalDate(row.Date ?? row.date)
  const price = Number(row.Price ?? row.price)
  const open = Number(row.Open ?? row.open)
  const high = Number(row.High ?? row.high)
  const low = Number(row.Low ?? row.low)
  if (!date || ![price, open, high, low].every((value) => Number.isFinite(value) && value > 0)) return []
  return [{
    date,
    price,
    open,
    high,
    low,
    volume: String(row['Vol.'] ?? row.volume ?? ''),
    changePercent: String(row['Change %'] ?? row.changePercent ?? ''),
  }]
}

export async function getCoffeeMarketHistoryRow(date: string): Promise<CoffeeMarketHistoryRow | null> {
  const { url } = getSupabaseConfig()
  const params = new URLSearchParams({ select: '*', market_date: `eq.${date}`, limit: '1' })
  const response = await fetch(
    `${url}/rest/v1/${encodeURIComponent(HISTORICAL_TABLE)}?${params.toString()}`,
    { headers: headers(), cache: 'no-store' },
  )
  if (!response.ok) throw new Error(`Supabase history row read returned HTTP ${response.status}`)
  const rows = (await response.json()) as Array<Record<string, unknown>>
  return rows.flatMap(parseHistoricalRow)[0] ?? null
}

export async function upsertCoffeeMarketHistoryRow(row: CoffeeMarketHistoryRow): Promise<boolean> {
  const { url } = getSupabaseConfig()
  const existing = await getCoffeeMarketHistoryRow(row.date)
  const [year, month, day] = row.date.split('-')
  const response = await fetch(
    `${url}/rest/v1/${encodeURIComponent(HISTORICAL_TABLE)}?on_conflict=market_date`,
    {
      method: 'POST',
      headers: { ...headers('resolution=merge-duplicates,return=minimal'), 'Content-Type': 'application/json' },
      body: JSON.stringify({
        market_date: row.date,
        Date: `${month}/${day}/${year}`,
        Price: row.price,
        Open: row.open,
        High: row.high,
        Low: row.low,
        'Vol.': row.volume,
        'Change %': row.changePercent,
        source: row.source ?? 'databento',
        source_contract: row.sourceContract ?? null,
        source_instrument_id: row.sourceInstrumentId ?? null,
        source_retrieved_at: row.sourceRetrievedAt ?? new Date().toISOString(),
      }),
    },
  )
  if (!response.ok) throw new Error(`Supabase history upsert returned HTTP ${response.status}: ${await response.text()}`)
  return existing !== null
}

export async function upsertCoffeeMarketSnapshot(snapshot: CoffeeMarketSnapshot): Promise<void> {
  const { url } = getSupabaseConfig()
  const response = await fetch(`${url}/rest/v1/coffee_market_snapshots?on_conflict=market_date`, {
    method: 'POST',
    headers: { ...headers('resolution=merge-duplicates,return=minimal'), 'Content-Type': 'application/json' },
    body: JSON.stringify({
      market_date: snapshot.marketDate,
      front_contract: snapshot.frontContract,
      front_price: snapshot.front,
      next_contract: snapshot.nextContract,
      next_price: snapshot.nextPrice,
      shape: snapshot.shape,
      spread: snapshot.spread,
      volume: snapshot.volume,
      open_interest: snapshot.openInterest,
      open_interest_as_of: snapshot.openInterestAsOf,
      price_as_of: snapshot.priceAsOf,
      unit: snapshot.unit,
      retrieved_at: snapshot.retrievedAt ?? new Date().toISOString(),
    }),
  })
  if (!response.ok) throw new Error(`Supabase snapshot upsert returned HTTP ${response.status}: ${await response.text()}`)
}
