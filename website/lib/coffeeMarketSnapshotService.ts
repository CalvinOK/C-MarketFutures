import { collectCoffeeMarketSnapshot } from '@/scripts/collect-coffee-market-snapshot'
import { getLatestCoffeeMarketSnapshot, upsertCoffeeMarketSnapshot } from '@/lib/supabaseServer'
import {
  getRefreshState,
  isRefreshCooldownActive,
  completeRefresh,
  tryClaimRefresh,
  waitForRefresh,
} from '@/lib/refreshState'

export const MARKET_SNAPSHOT_TTL_MS = 60 * 60 * 1000
const DATASET = 'market_snapshot_hourly'

export type CoffeeMarketSnapshotResult = {
  snapshot: NonNullable<Awaited<ReturnType<typeof getLatestCoffeeMarketSnapshot>>>
  metadata: {
    retrievedAt: string
    source: 'supabase-cache' | 'supabase-stale' | 'collector'
    fetchedAt: string
    ageMinutes: number
    ttlMinutes: 60
    isStale: boolean
    lastSuccessfulUpdate: string
  }
}

function makeResult(snapshot: NonNullable<Awaited<ReturnType<typeof getLatestCoffeeMarketSnapshot>>>, stale: boolean, source: CoffeeMarketSnapshotResult['metadata']['source']): CoffeeMarketSnapshotResult {
  const fetchedAt = snapshot.retrievedAt ?? new Date(0).toISOString()
  const ageMinutes = Math.max(0, Math.floor((Date.now() - Date.parse(fetchedAt)) / 60_000))
  return {
    snapshot,
    metadata: {
      retrievedAt: fetchedAt,
      source,
      fetchedAt,
      ageMinutes,
      ttlMinutes: 60,
      isStale: stale,
      lastSuccessfulUpdate: fetchedAt,
    },
  }
}

export async function getCoffeeMarketSnapshot(): Promise<CoffeeMarketSnapshotResult> {
  const latest = await getLatestCoffeeMarketSnapshot()
  if (latest) {
    const age = Date.now() - Date.parse(latest.retrievedAt ?? '')
    if (age < MARKET_SNAPSHOT_TTL_MS) {
      console.log(`[coffee-snapshot] cache hit age_minutes=${Math.floor(age / 60_000)}`)
      return makeResult(latest, false, 'supabase-cache')
    }
    console.log(`[coffee-snapshot] stale age_minutes=${Math.floor(age / 60_000)}`)
  }

  const rechecked = await getLatestCoffeeMarketSnapshot()
  if (rechecked && Date.now() - Date.parse(rechecked.retrievedAt ?? '') < MARKET_SNAPSHOT_TTL_MS) {
    return makeResult(rechecked, false, 'supabase-cache')
  }

  const state = await getRefreshState(DATASET)
  if (isRefreshCooldownActive(state)) {
    console.log('[coffee-snapshot] cooldown active')
    if (rechecked) return makeResult(rechecked, true, 'supabase-stale')
    throw new Error('Market snapshot provider is temporarily unavailable')
  }

  const token = await tryClaimRefresh(DATASET)
  if (!token) {
    const waited = await waitForRefresh(getLatestCoffeeMarketSnapshot, (snapshot) => Boolean(snapshot && Date.now() - Date.parse(snapshot.retrievedAt ?? '') < MARKET_SNAPSHOT_TTL_MS))
    if (waited) return makeResult(waited, false, 'supabase-cache')
    if (rechecked) return makeResult(rechecked, true, 'supabase-stale')
    throw new Error('Market snapshot refresh is already in progress')
  }

  try {
    const afterClaim = await getLatestCoffeeMarketSnapshot()
    if (afterClaim && Date.now() - Date.parse(afterClaim.retrievedAt ?? '') < MARKET_SNAPSHOT_TTL_MS) {
      await completeRefresh(DATASET, token, true)
      return makeResult(afterClaim, false, 'supabase-cache')
    }
    console.log('[coffee-snapshot] ICE/CFTC refresh started')
    const refreshed = await collectCoffeeMarketSnapshot()
    await upsertCoffeeMarketSnapshot(refreshed)
    await completeRefresh(DATASET, token, true)
    console.log('[coffee-snapshot] stored snapshot')
    return makeResult(refreshed, false, 'collector')
  } catch (error) {
    await completeRefresh(DATASET, token, false, error)
    console.error('[coffee-snapshot] refresh failed; serving stale:', error instanceof Error ? error.message : String(error))
    if (rechecked) return makeResult(rechecked, true, 'supabase-stale')
    throw error
  }
}
