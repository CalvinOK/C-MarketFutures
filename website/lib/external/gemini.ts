import type { RawArticle } from './newsapi'
import { fetchWithTimeout } from '@/lib/http'

export type ProcessedArticle = RawArticle & {
  summary: string
  category: 'Market Brief' | 'Commodities Desk' | 'Trade Note'
}

const VALID_CATEGORIES = ['Market Brief', 'Commodities Desk', 'Trade Note'] as const
const DEFAULT_MODEL = 'gemini-2.5-flash-lite'

type GeminiPart = { text?: string }
type GeminiCandidate = { content?: { parts?: GeminiPart[] } }
type GeminiGenerateResponse = { candidates?: GeminiCandidate[] }

function limitWords(text: string, maxWords = 22): string {
  const words = text.trim().split(/\s+/).filter(Boolean)
  if (words.length <= maxWords) return text.trim()
  return words.slice(0, maxWords).join(' ')
}

function extractJSONArray(raw: string): string {
  const trimmed = raw.trim()
  const first = trimmed.indexOf('[')
  const last = trimmed.lastIndexOf(']')
  if (first === -1 || last === -1 || last <= first) return '[]'
  return trimmed.slice(first, last + 1)
}

async function generateWithGemini(prompt: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY
  if (!apiKey) throw new Error('GEMINI_API_KEY is not set')

  const model = process.env.GEMINI_MODEL || DEFAULT_MODEL
  const response = await fetchWithTimeout(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 1024,
          responseMimeType: 'application/json',
        },
        systemInstruction: {
          parts: [
            {
              text: `You are a commodity markets analyst specializing in coffee futures.
You will receive a list of news articles. For each article, decide if it is PRIMARILY about coffee as a commodity, futures market, supply chain, or agricultural investment.
- If an article only mentions coffee incidentally (e.g. a general economy piece that lists coffee among many goods), SKIP it — do not include it in the output.
- Only include articles whose main subject is coffee markets, coffee futures/contracts, coffee crop/supply/demand, or coffee trade/logistics.

For each INCLUDED article return an element in a JSON array with:
- "index": the article [N] number (integer)
- "summary": one precise sentence, max 22 words, factual, based only on the article text
- "category": exactly one of "Market Brief", "Commodities Desk", or "Trade Note"
Rules:
* Market Brief  -> macro price moves, fund flows, exchange data, COT positioning
* Commodities Desk -> supply/demand, crop conditions, weather impact, ICE inventory
* Trade Note -> FX impact, shipping, policy, roaster hedging, arbitrage
Return ONLY a valid JSON array. No other text. Return [] if no articles qualify.`,
            },
          ],
        },
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
      }),
      cache: 'no-store',
    },
    10_000,
  )

  if (!response.ok) {
    const body = await response.text()
    throw new Error(`Gemini HTTP ${response.status}: ${body.slice(0, 200)}`)
  }

  const payload = (await response.json()) as GeminiGenerateResponse
  const text = payload.candidates?.[0]?.content?.parts?.map((p) => p.text ?? '').join('\n') ?? '[]'
  return text
}

export async function processArticles(articles: RawArticle[]): Promise<ProcessedArticle[]> {
  const batch = articles.slice(0, 3) // process at most 3 to keep the feed focused

  const articleList = batch
    .map((a, i) => `[${i}] ${a.title}\n${a.description ?? ''}`)
    .join('\n\n')

  const text = await generateWithGemini(`Process these coffee market articles:\n\n${articleList}`)

  let parsed: Array<{ index: number; summary: string; category: string }>
  try {
    parsed = JSON.parse(extractJSONArray(text))
  } catch {
    console.error('[gemini] Failed to parse response:', text.slice(0, 200))
    return []
  }

  return parsed
    .filter((p) => Number.isInteger(p.index) && p.index >= 0 && p.index < batch.length)
    .filter((p) => VALID_CATEGORIES.includes(p.category as (typeof VALID_CATEGORIES)[number]))
    .map((p) => ({
      ...batch[p.index],
      summary: limitWords(p.summary, 22),
      category: p.category as ProcessedArticle['category'],
    }))
    .filter((p) => p.title) // guard against bad indices
    .slice(0, 3)
}