/**
 * DataBento client for ICE Coffee C futures (KC).
 * Dataset: IFUS.IMPACT (ICE Futures US)
 * Symbols use the continuous format: KC.c.0 (front), KC.c.1, KC.c.2, KC.c.3
 */

import { fetchWithTimeout } from '@/lib/http'

export type ContractData = {
  symbol: string        // exchange symbol, e.g. KCK26
  lastPrice: number     // US cents per pound
  priceChange: number
  volume: number
  openInterest: number
  source: 'databento' | 'alphavantage'
}

// ICE Coffee C expiry schedule — update annually
const KC_EXPIRY_MAP: Record<string, string> = {
  KCH26: '2026-03-20',
  KCK26: '2026-05-19',
  KCN26: '2026-07-20',
  KCU26: '2026-09-18',
  KCZ26: '2026-12-18',
  KCH27: '2027-03-19',
  KCK27: '2027-05-18',
}

const CONTINUOUS_SYMBOLS = ['KC.c.0', 'KC.c.1', 'KC.c.2', 'KC.c.3']

export async function fetchCoffeeContracts(): Promise<ContractData[]> {
  const apiKey = process.env.DATABENTO_API_KEY
  if (!apiKey) throw new Error('DATABENTO_API_KEY is not set')

  const now = new Date()
  const start = new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000).toISOString()
  const end = now.toISOString()

  const response = await fetchWithTimeout('https://hist.databento.com/v0/timeseries.get_range', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Basic ${Buffer.from(`${apiKey}:`).toString('base64')}`,
    },
    body: JSON.stringify({
      dataset: 'IFUS.IMPACT',
      symbols: CONTINUOUS_SYMBOLS,
      schema: 'ohlcv-1d',
      start,
      end,
      stype_in: 'continuous',
      encoding: 'json',
    }),
    cache: 'no-store',
  }, 10_000)

  if (!response.ok) {
    const body = await response.text()
    throw new Error(`DataBento HTTP ${response.status}: ${body.slice(0, 200)}`)
  }

  const records: Array<{
    symbol: string
    open: number
    close: number
    volume: number
    ts_event: string
  }> = await response.json()

  // Keep only the most recent bar per symbol
  const latest = new Map<string, (typeof records)[0]>()
  for (const r of records) {
    const existing = latest.get(r.symbol)
    if (!existing || r.ts_event > existing.ts_event) {
      latest.set(r.symbol, r)
    }
  }

  const results: ContractData[] = []
  let contractIndex = 0

  for (const [, bar] of latest) {
    const symbol = resolveSymbol(contractIndex)
    if (!symbol) { contractIndex++; continue }

    // DataBento prices are in units of 0.01 cents — divide by 100 to get US cents/lb
    const lastPrice = bar.close / 100
    const priceChange = (bar.close - bar.open) / 100

    results.push({
      symbol,
      lastPrice,
      priceChange,
      volume: bar.volume,
      openInterest: 0, // ohlcv schema does not include OI; use statistics schema for OI
      source: 'databento',
    })
    contractIndex++
  }

  return results
}

export function getExpiryDate(symbol: string): string | null {
  return KC_EXPIRY_MAP[symbol] ?? null
}

/** Maps continuous contract index to the nearest active exchange symbol. */
function resolveSymbol(index: number): string | null {
  const now = new Date()
  const activeSymbols = Object.entries(KC_EXPIRY_MAP)
    .filter(([, expiry]) => new Date(expiry) > now)
    .sort(([, a], [, b]) => a.localeCompare(b))
    .map(([sym]) => sym)

  return activeSymbols[index] ?? null
}
