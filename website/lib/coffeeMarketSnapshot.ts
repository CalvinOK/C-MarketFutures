import * as cheerio from 'cheerio'
import { fetchWithTimeout } from '@/lib/http'

export type CurveShape = 'Contango' | 'Backwardation' | 'Flat'

export type IceCoffeeContract = {
  contract: string
  price: number
  volume: number
  priceAsOf: string | null
}

export type CoffeeMarketSnapshot = {
  front: number
  frontContract: string
  nextContract: string
  nextPrice: number
  shape: CurveShape
  spread: number
  volume: number
  openInterest: number
  openInterestAsOf: string
  priceAsOf: string | null
  unit: 'US¢/lb'
  contracts: IceCoffeeContract[]
  retrievedAt: string
}

type CftcRow = {
  cftc_contract_market_code?: string
  market_and_exchange_names?: string
  report_date_as_yyyy_mm_dd?: string
  open_interest_all?: string | number
}

const ICE_URL = 'https://www.ice.com/products/15/Coffee-C/data?marketId=5460931'
const CFTC_URL = 'https://publicreporting.cftc.gov/resource/6dca-aqww.json'
const CFTC_MARKET_CODE = '083731'
const MONTHS: Record<string, number> = {
  Mar: 3,
  May: 5,
  Jul: 7,
  Sep: 9,
  Dec: 12,
}

function parseNumeric(value: string | number | null | undefined): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (!value) return null
  const normalized = value.replace(/[$,%\s,\u00a0]/g, '').trim()
  if (!normalized || normalized === '-') return null
  const parsed = Number(normalized)
  return Number.isFinite(parsed) ? parsed : null
}

export function parseContractMonth(label: string): { year: number; month: number } | null {
  const match = label.trim().match(/^(Mar|May|Jul|Sep|Dec)(\d{2}|\d{4})$/i)
  if (!match) return null

  const month = MONTHS[`${match[1][0].toUpperCase()}${match[1].slice(1).toLowerCase()}`]
  const numericYear = Number(match[2])
  const year = match[2].length === 2 ? 2000 + numericYear : numericYear
  if (!month || !Number.isInteger(year)) return null
  return { year, month }
}

export function calculateCurveShape(frontPrice: number, nextPrice: number): { spread: number; shape: CurveShape } {
  const spread = nextPrice - frontPrice
  return {
    spread,
    shape: spread > 0 ? 'Contango' : spread < 0 ? 'Backwardation' : 'Flat',
  }
}

export function calculateTotalVolume(contracts: IceCoffeeContract[]): number {
  return contracts.reduce((total, contract) => total + contract.volume, 0)
}

function normalizeTimestamp(value: string): string | null {
  const trimmed = value.trim()
  if (!trimmed) return null
  const parsed = Date.parse(trimmed)
  return Number.isFinite(parsed) ? new Date(parsed).toISOString() : trimmed
}

function findHeaderIndex(headers: string[], patterns: RegExp[]): number {
  return headers.findIndex((header) => patterns.some((pattern) => pattern.test(header)))
}

export function parseIceCoffeeContracts(html: string): IceCoffeeContract[] {
  const $ = cheerio.load(html)
  const parsed: IceCoffeeContract[] = []

  $('table').each((_tableIndex, table) => {
    const rows = $(table).find('tr').toArray()
    for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
      const headerCells = $(rows[rowIndex])
        .find('th, td')
        .map((_index, cell) => $(cell).text().replace(/\s+/g, ' ').trim().toLowerCase())
        .get()
      const contractIndex = findHeaderIndex(headerCells, [/contract/, /month/])
      const priceIndex = findHeaderIndex(headerCells, [/^last$/, /last price/, /^price$/])
      const volumeIndex = findHeaderIndex(headerCells, [/volume/])
      if (contractIndex < 0 || priceIndex < 0 || volumeIndex < 0) continue

      const dateIndex = findHeaderIndex(headerCells, [/timestamp/, /date/, /as of/])
      for (const row of rows.slice(rowIndex + 1)) {
        const cells = $(row)
          .find('td, th')
          .map((_index, cell) => $(cell).text().replace(/\s+/g, ' ').trim())
          .get()
        const contractLabel = cells[contractIndex] ?? ''
        const month = parseContractMonth(contractLabel)
        if (!month) continue

        const price = parseNumeric(cells[priceIndex])
        const volumeText = cells[volumeIndex]
        const volume = volumeText === '-' || volumeText === '' ? 0 : parseNumeric(volumeText)
        if (price === null || price <= 0 || volume === null || volume < 0) continue

        parsed.push({
          contract: contractLabel,
          price,
          volume,
          priceAsOf: dateIndex >= 0 ? normalizeTimestamp(cells[dateIndex] ?? '') : null,
        })
      }
      if (parsed.length > 0) return false
    }
    return undefined
  })

  const unique = Array.from(new Map(parsed.map((contract) => [contract.contract, contract])).values())
  unique.sort((left, right) => {
    const leftMonth = parseContractMonth(left.contract)
    const rightMonth = parseContractMonth(right.contract)
    if (!leftMonth || !rightMonth) return 0
    return leftMonth.year - rightMonth.year || leftMonth.month - rightMonth.month
  })

  if (unique.length < 2) {
    throw new Error('ICE Coffee C table parsing failed: fewer than two valid contract rows were found')
  }
  return unique
}

async function fetchIceContracts(): Promise<IceCoffeeContract[]> {
  const response = await fetchWithTimeout(
    ICE_URL,
    {
      cache: 'no-store',
      headers: {
        Accept: 'text/html,application/xhtml+xml',
        'User-Agent': 'CoffeeMarketSnapshot/1.0',
      },
    },
    15_000,
  )
  if (!response.ok) throw new Error(`ICE Coffee C page returned HTTP ${response.status}`)
  return parseIceCoffeeContracts(await response.text())
}

export async function fetchLatestCoffeeOpenInterest(): Promise<{
  openInterest: number
  openInterestAsOf: string
}> {
  const query = new URLSearchParams({
    '$select': 'cftc_contract_market_code,market_and_exchange_names,report_date_as_yyyy_mm_dd,open_interest_all',
    '$where': `cftc_contract_market_code = '${CFTC_MARKET_CODE}'`,
    '$order': 'report_date_as_yyyy_mm_dd DESC',
    '$limit': '1',
  })
  const headers: HeadersInit = { Accept: 'application/json' }
  if (process.env.CFTC_APP_TOKEN) headers['X-App-Token'] = process.env.CFTC_APP_TOKEN

  const response = await fetchWithTimeout(`${CFTC_URL}?${query.toString()}`, { headers, cache: 'no-store' }, 15_000)
  if (!response.ok) throw new Error(`CFTC Coffee C request returned HTTP ${response.status}`)

  const rows = (await response.json()) as CftcRow[]
  const row = rows[0]
  if (!row || row.cftc_contract_market_code !== CFTC_MARKET_CODE) {
    throw new Error(`CFTC returned no Coffee C record for contract code ${CFTC_MARKET_CODE}`)
  }
  const openInterest = parseNumeric(row.open_interest_all)
  const reportDate = row.report_date_as_yyyy_mm_dd?.slice(0, 10)
  if (openInterest === null || openInterest < 0) throw new Error('CFTC Coffee C open_interest_all is not a valid number')
  if (!reportDate) throw new Error('CFTC Coffee C report date is missing')

  return { openInterest, openInterestAsOf: reportDate }
}

export async function getCoffeeMarketSnapshot(): Promise<CoffeeMarketSnapshot> {
  const [contracts, openInterest] = await Promise.all([fetchIceContracts(), fetchLatestCoffeeOpenInterest()])
  const frontContract = contracts[0]
  const nextContract = contracts[1]
  const { spread, shape } = calculateCurveShape(frontContract.price, nextContract.price)

  return {
    front: frontContract.price,
    frontContract: frontContract.contract,
    nextContract: nextContract.contract,
    nextPrice: nextContract.price,
    shape,
    spread,
    volume: calculateTotalVolume(contracts),
    ...openInterest,
    priceAsOf: frontContract.priceAsOf,
    unit: 'US¢/lb',
    contracts,
    retrievedAt: new Date().toISOString(),
  }
}
