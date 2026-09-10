import { chromium } from 'playwright'
import {
  calculateCurveShape,
  calculateTotalVolume,
  normalizeMarketDate,
  parseNumeric,
  parseRenderedIceRows,
  type CoffeeMarketSnapshot,
} from '@/lib/coffeeMarketSnapshot'
import { upsertCoffeeMarketSnapshot } from '@/lib/supabaseServer'

const ICE_URL = 'https://www.ice.com/products/15/Coffee-C/data?marketId=5460931'
const CFTC_URL = 'https://publicreporting.cftc.gov/resource/6dca-aqww.json'
const CFTC_MARKET_CODE = '083731'

type CftcRow = {
  cftc_contract_market_code?: string
  report_date_as_yyyy_mm_dd?: string
  open_interest_all?: string | number
}

async function extractIceRows(): Promise<string[][]> {
  const browser = await chromium.launch({ headless: true })
  try {
    const page = await browser.newPage({ userAgent: 'CoffeeMarketSnapshot/1.0 (+GitHub Actions)' })
    await page.goto(ICE_URL, { waitUntil: 'domcontentloaded', timeout: 45_000 })
    await page.waitForFunction(
      () => /contract/i.test(document.body.innerText) && /last/i.test(document.body.innerText) && /volume/i.test(document.body.innerText),
      undefined,
      { timeout: 30_000 },
    )

    const tables = await page.locator('table').evaluateAll((elements) =>
      elements.map((table) =>
        Array.from(table.querySelectorAll('tr')).map((row) =>
          Array.from(row.querySelectorAll('th,td')).map((cell) => (cell.textContent ?? '').replace(/\s+/g, ' ').trim()),
        ),
      ),
    )
    for (const rows of tables) {
      const header = rows.find((row) => row.some((cell) => /contract/i.test(cell)) && row.some((cell) => /last/i.test(cell)) && row.some((cell) => /volume/i.test(cell)))
      if (header) {
        const headerIndex = rows.indexOf(header)
        return rows.slice(headerIndex + 1)
      }
    }
    throw new Error('ICE rendered page contained no futures table with Contract, Last, and Volume headers')
  } finally {
    await browser.close()
  }
}

async function fetchLatestCoffeeOpenInterest(): Promise<{ openInterest: number; openInterestAsOf: string }> {
  const query = new URLSearchParams({
    '$select': 'cftc_contract_market_code,report_date_as_yyyy_mm_dd,open_interest_all',
    '$where': `cftc_contract_market_code = '${CFTC_MARKET_CODE}'`,
    '$order': 'report_date_as_yyyy_mm_dd DESC',
    '$limit': '1',
  })
  const headers: HeadersInit = { Accept: 'application/json' }
  if (process.env.CFTC_APP_TOKEN) headers['X-App-Token'] = process.env.CFTC_APP_TOKEN
  const response = await fetch(`${CFTC_URL}?${query.toString()}`, { headers })
  if (!response.ok) throw new Error(`CFTC request returned HTTP ${response.status}`)
  const rows = (await response.json()) as CftcRow[]
  const row = rows[0]
  const openInterest = parseNumeric(row?.open_interest_all)
  const reportDate = row?.report_date_as_yyyy_mm_dd?.slice(0, 10)
  if (!row || row.cftc_contract_market_code !== CFTC_MARKET_CODE) throw new Error(`CFTC returned no Coffee C record for ${CFTC_MARKET_CODE}`)
  if (openInterest === null || openInterest < 0) throw new Error('CFTC open_interest_all is not numeric')
  if (!reportDate) throw new Error('CFTC report date is missing')
  return { openInterest, openInterestAsOf: reportDate }
}

export async function collectCoffeeMarketSnapshot(): Promise<CoffeeMarketSnapshot> {
  const rows = await extractIceRows()
  const contracts = parseRenderedIceRows(rows)
  console.log('[coffee-snapshot] Parsed ICE contracts:', JSON.stringify(contracts))
  const [openInterest] = await Promise.all([fetchLatestCoffeeOpenInterest()])
  const front = contracts[0]
  const next = contracts[1]
  const { spread, shape } = calculateCurveShape(front.price, next.price)
  const priceAsOf = front.priceAsOf
  const snapshot: CoffeeMarketSnapshot = {
    front: front.price,
    frontContract: front.contract,
    nextContract: next.contract,
    nextPrice: next.price,
    shape,
    spread,
    volume: calculateTotalVolume(contracts),
    ...openInterest,
    priceAsOf,
    marketDate: normalizeMarketDate(priceAsOf),
    unit: 'US¢/lb',
    contracts,
    retrievedAt: new Date().toISOString(),
  }
  if (!Number.isFinite(snapshot.front) || snapshot.front <= 0 || !Number.isFinite(snapshot.nextPrice) || snapshot.nextPrice <= 0 || !Number.isFinite(snapshot.volume) || snapshot.volume < 0) {
    throw new Error('Collected Coffee C snapshot contains invalid market values')
  }
  console.log('[coffee-snapshot] Calculated snapshot:', JSON.stringify(snapshot))
  return snapshot
}

async function main(): Promise<void> {
  const snapshot = await collectCoffeeMarketSnapshot()
  await upsertCoffeeMarketSnapshot(snapshot)
  console.log(`[coffee-snapshot] Upserted ${snapshot.marketDate}: ${snapshot.frontContract} ${snapshot.front} ${snapshot.shape} volume=${snapshot.volume} OI=${snapshot.openInterest} (${snapshot.openInterestAsOf})`)
}

main().catch((error: unknown) => {
  console.error('[coffee-snapshot] Collection failed:', error instanceof Error ? error.message : error)
  process.exitCode = 1
})
