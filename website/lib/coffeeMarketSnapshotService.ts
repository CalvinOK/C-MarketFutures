import { getCoffeeMarketData, ICE_PRICE_TTL_MS } from '@/lib/coffeeMarketDataService'

export const MARKET_SNAPSHOT_TTL_MS = ICE_PRICE_TTL_MS
export type CoffeeMarketSnapshotResult = Awaited<ReturnType<typeof getCoffeeMarketData>>

export async function getCoffeeMarketSnapshot(): Promise<CoffeeMarketSnapshotResult> {
  return getCoffeeMarketData()
}
