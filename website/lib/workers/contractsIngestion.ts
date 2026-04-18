import { getDb } from '@/lib/db'
import { cacheDel } from '@/lib/redis'
import { fetchCoffeeContracts, getExpiryDate } from '@/lib/external/databento'
import { fetchCoffeeSpotPrice } from '@/lib/external/alphavantage'
import { logIngestion, withRetry } from '@/lib/workers/shared'

// Active KC contract symbols in expiry order
const ACTIVE_SYMBOLS = ['KCK26', 'KCN26', 'KCU26', 'KCZ26', 'KCH27']

export async function runContractsIngestion(): Promise<void> {
  const db = getDb()
  const start = Date.now()
  let recordsOut = 0
  let hasLock = false
  let sourceType: 'primary' | 'fallback' = 'primary'

  try {
    const lock = await db.query<{ locked: boolean }>(
      `SELECT pg_try_advisory_lock(hashtext($1)) AS locked`,
      ['contracts_ingestion'],
    )
    hasLock = lock.rows[0]?.locked === true
    if (!hasLock) {
      console.log('[contracts] Skipping run because another ingestion is in progress')
      return
    }

    let contracts: Awaited<ReturnType<typeof fetchCoffeeContracts>>

    // Primary: DataBento (ICE Futures US)
    try {
      contracts = await withRetry(() => fetchCoffeeContracts(), 2)
      console.log(`[contracts] DataBento returned ${contracts.length} records`)
    } catch (err: unknown) {
      console.warn('[contracts] DataBento failed, falling back to AlphaVantage:', (err as Error).message)
      sourceType = 'fallback'

      // Fallback: AlphaVantage spot price → synthesize a flat curve
      const spot = await withRetry(() => fetchCoffeeSpotPrice())
      if (spot.length === 0) throw new Error('AlphaVantage returned no data')

      const frontPrice = spot[0].price
      // Synthesize a modest contango (typical ~0.5% per month carry)
      contracts = ACTIVE_SYMBOLS.map((symbol, i) => ({
        symbol,
        lastPrice: parseFloat((frontPrice * (1 + 0.005 * i)).toFixed(2)),
        priceChange: 0,
        volume: 0,
        openInterest: 0,
        source: 'alphavantage' as const,
      }))
    }

    if (contracts.length === 0) {
      throw new Error('Zero contracts returned from all sources')
    }

    // Persist each contract snapshot
    for (const c of contracts) {
      const expiry = getExpiryDate(c.symbol)
      if (!expiry) {
        console.warn(`[contracts] Unknown symbol ${c.symbol}, skipping`)
        continue
      }

      const pricePct = c.lastPrice > 0 ? c.priceChange / c.lastPrice : 0

      await db.query(
        `INSERT INTO futures_snapshots
           (symbol, expiry_date, last_price, price_change, price_change_pct,
            volume, open_interest, source)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
        [
          c.symbol,
          expiry,
          c.lastPrice,
          c.priceChange,
          pricePct,
          c.volume,
          c.openInterest,
          c.source,
        ],
      )
      recordsOut++
    }

    // Refresh materialized view for O(1) reads on GET /api/contracts
    try {
      await db.query('REFRESH MATERIALIZED VIEW CONCURRENTLY contracts_latest')
    } catch (err: unknown) {
      console.warn('[contracts] Failed to refresh contracts_latest view:', (err as Error).message)
    }

    // Bust cache
    await cacheDel('contracts:live')

    if (sourceType === 'fallback') {
      console.warn('[contracts] Latest snapshot came from fallback source')
    }

    await logIngestion(db, 'contracts', 'success', contracts.length, recordsOut, Date.now() - start)
    console.log(`[contracts] Persisted ${recordsOut} snapshots`)
  } catch (err: unknown) {
    const msg = (err as Error).message
    console.error('[contracts] Ingestion failed:', msg)
    await logIngestion(db, 'contracts', 'failed', 0, recordsOut, Date.now() - start, msg)
    throw err
  } finally {
    if (hasLock) {
      await db.query(`SELECT pg_advisory_unlock(hashtext($1))`, ['contracts_ingestion'])
    }
  }
}
