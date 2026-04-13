import { Pool, PoolClient } from 'pg'

let pool: Pool | null = null

export function getDb(): Pool {
  if (!pool) {
    if (!process.env.DATABASE_URL) {
      throw new Error('DATABASE_URL is not set')
    }
    pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      max: 10,
      idleTimeoutMillis: 30_000,
      connectionTimeoutMillis: 3_000,
    })
    pool.on('error', (err) => {
      console.error('[db] Pool error:', err.message)
    })
  }
  return pool
}

export async function withClient<T>(fn: (client: PoolClient) => Promise<T>): Promise<T> {
  const client = await getDb().connect()
  try {
    return await fn(client)
  } finally {
    client.release()
  }
}
