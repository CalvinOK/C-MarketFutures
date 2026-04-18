import Redis from 'ioredis'

let client: Redis | null = null

/** Returns null if REDIS_URL is not configured — callers must handle gracefully. */
export function tryGetRedis(): Redis | null {
  if (!process.env.REDIS_URL) return null
  if (!client) {
    client = new Redis(process.env.REDIS_URL, {
      lazyConnect: true,
      maxRetriesPerRequest: 1,
      retryStrategy: (times) => (times > 3 ? null : Math.min(times * 100, 1000)),
    })
    client.on('error', (err) => {
      console.warn('[redis] Connection error:', err.message)
    })
  }
  return client
}

export async function cacheGet(key: string): Promise<string | null> {
  const redis = tryGetRedis()
  if (!redis) return null
  try {
    return await redis.get(key)
  } catch {
    return null
  }
}

export async function cacheSet(key: string, ttlSeconds: number, value: string): Promise<void> {
  const redis = tryGetRedis()
  if (!redis) return
  try {
    await redis.setex(key, ttlSeconds, value)
  } catch (err: unknown) {
    console.warn('[redis] cacheSet failed:', (err as Error).message)
  }
}

export async function cacheDel(key: string): Promise<void> {
  const redis = tryGetRedis()
  if (!redis) return
  try {
    await redis.del(key)
  } catch {
    // non-fatal
  }
}
