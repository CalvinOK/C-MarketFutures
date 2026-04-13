import { NextResponse } from 'next/server'

type RateEntry = {
  count: number
  resetAt: number
}

const RATE_STATE = new Map<string, RateEntry>()
const RATE_CLEANUP_INTERVAL_MS = 60_000
let lastCleanupAt = 0

function cleanupExpiredRateEntries(now: number): void {
  if (now - lastCleanupAt < RATE_CLEANUP_INTERVAL_MS) return
  for (const [key, entry] of RATE_STATE.entries()) {
    if (entry.resetAt <= now) {
      RATE_STATE.delete(key)
    }
  }
  lastCleanupAt = now
}

function getClientId(request: Request): string {
  const forwarded = request.headers.get('x-forwarded-for')
  if (forwarded) return forwarded.split(',')[0].trim()

  const realIp = request.headers.get('x-real-ip')
  if (realIp) return realIp.trim()

  return 'unknown'
}

export function requireInternalTokenIfConfigured(request: Request): NextResponse | null {
  const expectedToken = process.env.INTERNAL_API_TOKEN
  if (!expectedToken) return null

  const authHeader = request.headers.get('authorization')
  const token = authHeader?.startsWith('Bearer ') ? authHeader.slice(7).trim() : ''
  if (token === expectedToken) return null

  return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
}

export function enforceRateLimit(
  request: Request,
  key: string,
  limit: number,
  windowMs: number,
): NextResponse | null {
  const clientId = getClientId(request)
  const now = Date.now()
  cleanupExpiredRateEntries(now)
  const stateKey = `${key}:${clientId}`

  const current = RATE_STATE.get(stateKey)
  const entry: RateEntry =
    current && now < current.resetAt
      ? { ...current }
      : {
          count: 0,
          resetAt: now + windowMs,
        }

  entry.count += 1
  RATE_STATE.set(stateKey, entry)

  if (entry.count <= limit) return null

  const retryAfter = Math.ceil((entry.resetAt - now) / 1000)
  return NextResponse.json(
    { error: 'Rate limit exceeded', retryAfter },
    {
      status: 429,
      headers: { 'Retry-After': String(Math.max(retryAfter, 1)) },
    },
  )
}
