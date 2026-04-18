/**
 * Standalone cron worker process.
 * Run with:  npm run worker
 *
 * Requires DATABASE_URL and at least one data-source key to be set in the environment.
 */

import cron from 'node-cron'
import { runNewsIngestion } from '@/lib/workers/newsIngestion'
import { runContractsIngestion } from '@/lib/workers/contractsIngestion'
import { validateWorkerEnv } from '@/lib/env'

function safe(label: string, fn: () => Promise<void>): () => void {
  return () => {
    fn().catch((err: Error) => console.error(`[${label}] Unhandled error:`, err.message))
  }
}

// ─── Contracts ───────────────────────────────────────────────────────────────
// Every 2 min during ICE market hours (Mon–Fri, 07:30–14:30 ET)
cron.schedule(
  '*/2 7-14 * * 1-5',
  safe('contracts:market', runContractsIngestion),
  { timezone: 'America/New_York' },
)

// Every 10 min outside market hours (for settlement price updates)
cron.schedule(
  '*/10 0-7,15-23 * * 1-5',
  safe('contracts:off-hours', runContractsIngestion),
  { timezone: 'America/New_York' },
)

// ─── News ─────────────────────────────────────────────────────────────────────
// Every 20 min, all hours
cron.schedule('*/20 * * * *', safe('news', runNewsIngestion))

// ─── Startup warm-up ─────────────────────────────────────────────────────────
validateWorkerEnv()
console.log('[cron] Starting workers...')
Promise.allSettled([runContractsIngestion(), runNewsIngestion()]).then((results) => {
  for (const r of results) {
    if (r.status === 'rejected') console.warn('[cron] Warm-up error:', r.reason?.message)
  }
  console.log('[cron] Warm-up complete. Scheduled jobs running.')
})
