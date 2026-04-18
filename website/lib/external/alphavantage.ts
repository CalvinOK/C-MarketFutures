/**
 * AlphaVantage client.
 * Used as a fallback data source when DataBento is unavailable.
 * The COFFEE endpoint returns monthly global physical market prices,
 * not ICE futures — but gives a reliable front-month proxy.
 */

import { fetchWithTimeout } from '@/lib/http'

export type AVCommodityRow = {
  date: string
  price: number
}

export async function fetchCoffeeSpotPrice(): Promise<AVCommodityRow[]> {
  const apiKey = process.env.ALPHAVANTAGE_API_KEY
  if (!apiKey) throw new Error('ALPHAVANTAGE_API_KEY is not set')

  const url =
    `https://www.alphavantage.co/query` +
    `?function=COFFEE&interval=monthly&apikey=${apiKey}`

  const res = await fetchWithTimeout(url, { cache: 'no-store' }, 8_000)
  if (!res.ok) throw new Error(`AlphaVantage HTTP ${res.status}`)

  const json: { data?: Array<{ date: string; value: string }>; Information?: string } =
    await res.json()

  if (json.Information) throw new Error(`AlphaVantage rate limit: ${json.Information}`)
  if (!Array.isArray(json.data)) throw new Error('AlphaVantage: unexpected response shape')

  return json.data
    .slice(0, 6)
    .map((d) => ({ date: d.date, price: parseFloat(d.value) }))
    .filter((d) => Number.isFinite(d.price))
}
