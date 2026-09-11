/**
 * Standalone cron worker process.
 * Run with:  npm run worker
 *
 * Requires DATABASE_URL and at least one data-source key to be set in the environment.
 */

import cron from 'node-cron'
import { runNewsIngestion } from '@/lib/workers/newsIngestion'
import { runProjectionRefresh } from '@/lib/workers/projectionRefresh'
import { validateWorkerEnv } from '@/lib/env'

function safe(label: string, fn: () => Promise<void>): () => void {
  return () => {
    fn().catch((err: Error) => console.error(`[${label}] Unhandled error:`, err.message))
  }
}

// ─── News ─────────────────────────────────────────────────────────────────────
// Every 20 min, all hours
cron.schedule('*/20 * * * *', safe('news', runNewsIngestion))

// ─── Projection refresh ──────────────────────────────────────────────────────
// Weekly XGBoost retrain — Monday 06:00 ET, so the chart shows a projection
// anchored to the current week before traders look at it Monday morning.
cron.schedule(
  '0 6 * * 1',
  safe('projection-refresh', runProjectionRefresh),
  { timezone: 'America/New_York' },
)

// ─── Startup warm-up ─────────────────────────────────────────────────────────
validateWorkerEnv()
console.log('[cron] Starting workers...')
Promise.allSettled([runNewsIngestion()]).then((results) => {
  for (const r of results) {
    if (r.status === 'rejected') console.warn('[cron] Warm-up error:', r.reason?.message)
  }
  console.log('[cron] Warm-up complete. Scheduled jobs running.')
})
