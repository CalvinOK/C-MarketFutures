const REFRESH_COOLDOWN_MS = 15 * 60 * 1000
const DEFAULT_LEASE_SECONDS = 120

type RefreshStateRow = {
  last_attempt_at: string | null
  last_success_at: string | null
  status: string
}

export type RefreshState = {
  lastAttemptAt: string | null
  lastSuccessAt: string | null
  status: string
}

function getSupabaseConfig(): { url: string; key: string } {
  const url = process.env.SUPABASE_URL?.replace(/\/$/, '')
  const key = process.env.SUPABASE_SECRET_KEY
  if (!url || !key) throw new Error('SUPABASE_URL and SUPABASE_SECRET_KEY are required')
  return { url, key }
}

function headers(prefer?: string): HeadersInit {
  const { key } = getSupabaseConfig()
  return {
    Accept: 'application/json',
    apikey: key,
    Authorization: `Bearer ${key}`,
    ...(prefer ? { Prefer: prefer } : {}),
  }
}

export async function getRefreshState(dataset: string): Promise<RefreshState | null> {
  const { url } = getSupabaseConfig()
  const params = new URLSearchParams({ select: 'last_attempt_at,last_success_at,status', dataset: `eq.${dataset}`, limit: '1' })
  const response = await fetch(`${url}/rest/v1/coffee_data_refresh_state?${params}`, { headers: headers(), cache: 'no-store' })
  if (!response.ok) throw new Error(`Supabase refresh state read returned HTTP ${response.status}`)
  const row = ((await response.json()) as RefreshStateRow[])[0]
  return row ? { lastAttemptAt: row.last_attempt_at, lastSuccessAt: row.last_success_at, status: row.status } : null
}

export function isRefreshCooldownActive(state: RefreshState | null, now = Date.now()): boolean {
  if (!state?.lastAttemptAt || state.status !== 'failed') return false
  return now - Date.parse(state.lastAttemptAt) < REFRESH_COOLDOWN_MS
}

export async function tryClaimRefresh(dataset: string, leaseSeconds = DEFAULT_LEASE_SECONDS): Promise<string | null> {
  const { url } = getSupabaseConfig()
  const response = await fetch(`${url}/rest/v1/rpc/try_claim_coffee_refresh`, {
    method: 'POST',
    headers: { ...headers(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ p_dataset: dataset, p_lease_seconds: leaseSeconds }),
    cache: 'no-store',
  })
  if (!response.ok) throw new Error(`Supabase refresh lease claim returned HTTP ${response.status}`)
  const rows = (await response.json()) as Array<{ claimed: boolean; lock_token: string | null }>
  return rows[0]?.claimed ? rows[0].lock_token : null
}

export async function completeRefresh(dataset: string, lockToken: string, succeeded: boolean, error?: unknown): Promise<void> {
  const { url } = getSupabaseConfig()
  const message = error instanceof Error ? error.message : error ? String(error) : null
  const code = error instanceof Error && 'status' in error ? String((error as { status?: unknown }).status ?? '') : null
  const response = await fetch(`${url}/rest/v1/rpc/complete_coffee_refresh`, {
    method: 'POST',
    headers: { ...headers(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ p_dataset: dataset, p_lock_token: lockToken, p_succeeded: succeeded, p_error_code: code, p_error_message: message }),
    cache: 'no-store',
  })
  if (!response.ok) throw new Error(`Supabase refresh lease completion returned HTTP ${response.status}`)
}

export async function waitForRefresh<T>(read: () => Promise<T>, isFresh: (value: T) => boolean): Promise<T> {
  let latest = await read()
  for (let attempt = 0; attempt < 8 && !isFresh(latest); attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 250))
    latest = await read()
  }
  return latest
}
