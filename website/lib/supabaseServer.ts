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
