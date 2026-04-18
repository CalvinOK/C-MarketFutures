import { Pool } from 'pg'

function getHttpStatusFromMessage(message: string): number | null {
  const match = message.match(/HTTP\s+(\d{3})/i)
  if (!match) return null
  const parsed = Number(match[1])
  return Number.isFinite(parsed) ? parsed : null
}

function isRetryableError(err: unknown): boolean {
  if (!(err instanceof Error)) return true

  const status = getHttpStatusFromMessage(err.message)
  if (status === null) return true

  // Retry on transient classes only.
  if (status === 408 || status === 429) return true
  if (status >= 500) return true
  return false
}

export async function logIngestion(
  db: Pool,
  jobName: string,
  status: 'success' | 'partial' | 'failed',
  recordsIn: number,
  recordsOut: number,
  durationMs: number,
  errorMsg?: string,
): Promise<void> {
  try {
    await db.query(
      `INSERT INTO ingestion_log (job_name, status, records_in, records_out, duration_ms, error_msg)
       VALUES ($1, $2, $3, $4, $5, $6)`,
      [jobName, status, recordsIn, recordsOut, durationMs, errorMsg ?? null],
    )
  } catch (err: unknown) {
    // Never let logging failure break the worker
    console.error('[ingestion_log] Failed to write log:', (err as Error).message)
  }
}

export async function withRetry<T>(
  fn: () => Promise<T>,
  attempts = 3,
  baseDelayMs = 1000,
): Promise<T> {
  let lastErr: unknown
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn()
    } catch (err) {
      lastErr = err
      if (!isRetryableError(err)) {
        throw err
      }
      if (i < attempts - 1) {
        const jitter = Math.floor(Math.random() * 200)
        await sleep(baseDelayMs * 2 ** i + jitter)
      }
    }
  }
  throw lastErr
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
