import chromium from '@sparticuz/chromium'
import { chromium as playwrightChromium } from 'playwright-core'
import { existsSync } from 'node:fs'
import {
  calculateCurveShape,
  calculateTotalVolume,
  normalizeMarketDate,
  parseNumeric,
  parseRenderedIceRows,
  type CoffeeMarketSnapshot,
  type IceCoffeeContract,
} from '@/lib/coffeeMarketSnapshot'

const ICE_URL = 'https://www.ice.com/products/15/Coffee-C/data?marketId=5460931'
const CFTC_URL = 'https://publicreporting.cftc.gov/resource/6dca-aqww.json'
const CFTC_MARKET_CODE = '083731'

async function getBrowserExecutablePath(): Promise<string> {
  const configured = process.env.PLAYWRIGHT_EXECUTABLE_PATH?.trim()
  if (configured) {
    if (!existsSync(configured)) throw new Error(`Configured browser executable does not exist: ${configured}`)
    return configured
  }
  if (process.platform === 'darwin') {
    const candidates = [
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Chromium.app/Contents/MacOS/Chromium',
      `${process.env.HOME ?? ''}/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`,
      `${process.env.HOME ?? ''}/Applications/Chromium.app/Contents/MacOS/Chromium`,
    ]
    const executable = candidates.find((candidate) => candidate && existsSync(candidate))
    if (!executable) {
      throw new Error('No macOS Chrome/Chromium executable found; set PLAYWRIGHT_EXECUTABLE_PATH')
    }
    return executable
  }
  const executable = await chromium.executablePath()
  if (!existsSync(executable)) throw new Error(`Serverless Chromium executable does not exist: ${executable}`)
  return executable
}

type CftcRow = {
  cftc_contract_market_code?: string
  report_date_as_yyyy_mm_dd?: string
  open_interest_all?: string | number
}

async function extractIceRows(): Promise<{ headers: string[]; rows: string[][] }> {
  const executablePath = await getBrowserExecutablePath()
  const browser = await playwrightChromium.launch({
    args: process.platform === 'linux' ? chromium.args : [],
    executablePath,
    headless: true,
  })
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
        return { headers: header, rows: rows.slice(headerIndex + 1) }
      }
    }
    throw new Error('ICE rendered page contained no futures table with Contract, Last, and Volume headers')
  } finally {
    await browser.close()
  }
}

export async function fetchLatestCoffeeOpenInterest(): Promise<{ openInterest: number; openInterestAsOf: string }> {
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

export async function collectIceCoffeeContracts(): Promise<IceCoffeeContract[]> {
  const table = await extractIceRows()
  const contracts = parseRenderedIceRows(table.rows, table.headers)
  console.log('[coffee-snapshot] Parsed ICE contracts:', JSON.stringify(contracts))
  return contracts
}

export async function collectCoffeeMarketSnapshot(existingOpenInterest?: { openInterest: number; openInterestAsOf: string }): Promise<CoffeeMarketSnapshot> {
  const contracts = await collectIceCoffeeContracts()
  return buildCoffeeMarketSnapshot(contracts, existingOpenInterest ?? await fetchLatestCoffeeOpenInterest())
}

export function buildCoffeeMarketSnapshot(contracts: IceCoffeeContract[], openInterest: { openInterest: number; openInterestAsOf: string }): CoffeeMarketSnapshot {
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
