/**
 * DataBento client for ICE Coffee C futures (KC).
 * The existing Python market API owns the IFUS.IMPACT definition/statistics
 * requests; this server-side client consumes its normalized contract payload.
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

export class DatabentoProviderError extends Error {
  constructor(message: string, readonly status?: number, readonly providerCode?: string) {
    super(message)
    this.name = 'DatabentoProviderError'
  }
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

export async function fetchCoffeeContracts(): Promise<ContractData[]> {
  const baseUrl = process.env.MARKET_API_BASE_URL?.trim().replace(/\/$/, '')
  if (!baseUrl) throw new DatabentoProviderError('market_api_not_configured')
  const token = process.env.MARKET_API_TOKEN?.trim()
  const response = await fetchWithTimeout(`${baseUrl}/api/contracts`, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    cache: 'no-store',
  }, 30_000)

  if (!response.ok) {
    const body = await response.text()
    let providerCode = ''
    let providerMessage = body.replace(/\s+/g, ' ').slice(0, 240)
    try {
      const parsed = JSON.parse(body) as { code?: string; error?: string; message?: string }
      providerCode = parsed.code ?? ''
      providerMessage = parsed.error ?? parsed.message ?? providerMessage
    } catch {
      // Preserve a short sanitized text response for server logs.
    }
    throw new DatabentoProviderError(
      `databento_http_error status=${response.status}${providerCode ? ` code=${providerCode}` : ''} message=${providerMessage}`,
      response.status,
      providerCode || undefined,
    )
  }

  const payload = await response.json() as {
    contracts?: Array<{
      displaySymbol?: string
      symbol: string
      lastPrice?: number | null
      priceChange?: number | null
      volume?: number | null
      openInterest?: number | null
    }>
  }
  const contracts = payload.contracts ?? []
  return contracts.map((contract) => ({
    symbol: contract.displaySymbol ?? contract.symbol,
    lastPrice: contract.lastPrice ?? 0,
    priceChange: contract.priceChange ?? 0,
    volume: contract.volume ?? 0,
    openInterest: contract.openInterest ?? 0,
    source: 'databento',
  }))
}

export function getExpiryDate(symbol: string): string | null {
  return KC_EXPIRY_MAP[symbol] ?? null
}
