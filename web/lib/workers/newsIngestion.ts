import { getDb } from '@/lib/db'
import { cacheDel } from '@/lib/redis'
import { fetchCoffeeNews, hashHeadline } from '@/lib/external/newsapi'
import { processArticles } from '@/lib/external/gemini'
import { logIngestion, withRetry } from '@/lib/workers/shared'

export async function runNewsIngestion(): Promise<void> {
  const db = getDb()
  const start = Date.now()
  let recordsIn = 0
  let recordsOut = 0
  let hasLock = false

  try {
    const lock = await db.query<{ locked: boolean }>(
      `SELECT pg_try_advisory_lock(hashtext($1)) AS locked`,
      ['news_ingestion'],
    )
    hasLock = lock.rows[0]?.locked === true
    if (!hasLock) {
      console.log('[news] Skipping run because another ingestion is in progress')
      return
    }

    // 1. Fetch raw articles
    const rawArticles = await withRetry(() => fetchCoffeeNews())
    recordsIn = rawArticles.length

    if (rawArticles.length === 0) {
      await logIngestion(db, 'news', 'partial', 0, 0, Date.now() - start, 'No articles returned')
      return
    }

    // 2. Filter out already-persisted headlines
    const hashes = rawArticles.map((a) => hashHeadline(a.title))
    const { rows: existing } = await db.query<{ headline_hash: string }>(
      'SELECT headline_hash FROM news_articles WHERE headline_hash = ANY($1)',
      [hashes],
    )
    const existingSet = new Set(existing.map((r) => r.headline_hash))
    const newArticles = rawArticles.filter((_, i) => !existingSet.has(hashes[i]))

    if (newArticles.length === 0) {
      console.log('[news] No new articles since last run')
      await logIngestion(db, 'news', 'success', recordsIn, 0, Date.now() - start)
      return
    }

    // 3. Summarize + categorize via Gemini
    const processed = await withRetry(() => processArticles(newArticles))

    if (processed.length === 0) {
      await logIngestion(db, 'news', 'partial', recordsIn, 0, Date.now() - start, 'Gemini returned 0 results')
      return
    }

    // 4. Persist
    for (const article of processed) {
      const hash = hashHeadline(article.title)
      const result = await db.query(
        `INSERT INTO news_articles
           (headline, summary, category, source_name, source_url, published_at, headline_hash, raw_content)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
         ON CONFLICT (headline_hash) DO NOTHING`,
        [
          article.title,
          article.summary,
          article.category,
          article.source?.name ?? 'Unknown',
          article.url ?? null,
          new Date(article.publishedAt),
          hash,
          article.description ?? null,
        ],
      )
      recordsOut += result.rowCount ?? 0
    }

    // 5. Bust Redis cache so the next GET /api/news returns fresh data
    await cacheDel('news:feed')

    await logIngestion(db, 'news', 'success', recordsIn, recordsOut, Date.now() - start)
    console.log(`[news] Persisted ${recordsOut} new articles`)
  } catch (err: unknown) {
    const msg = (err as Error).message
    console.error('[news] Ingestion failed:', msg)
    await logIngestion(db, 'news', 'failed', recordsIn, recordsOut, Date.now() - start, msg)
    throw err
  } finally {
    if (hasLock) {
      await db.query(`SELECT pg_advisory_unlock(hashtext($1))`, ['news_ingestion'])
    }
  }
}
