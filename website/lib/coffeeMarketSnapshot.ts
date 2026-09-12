export type CurveShape = 'Contango' | 'Backwardation' | 'Flat'

export type IceCoffeeContract = {
  contract: string
  price: number
  volume: number
  priceAsOf: string | null
  priceChange?: number | null
  percentChange?: number | null
  symbol?: string
  sourceUrl?: string
  sourceName?: string
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
  marketDate: string
  unit: 'US¢/lb'
  contracts?: IceCoffeeContract[]
  retrievedAt?: string
}

const MONTHS: Record<string, number> = {
  Mar: 3,
  May: 5,
  Jul: 7,
  Sep: 9,
  Dec: 12,
}

const MONTH_CODES: Record<string, string> = {
  Jan: 'F', Feb: 'G', Mar: 'H', Apr: 'J', May: 'K', Jun: 'M',
  Jul: 'N', Aug: 'Q', Sep: 'U', Oct: 'V', Nov: 'X', Dec: 'Z',
}

export function parseNumeric(value: string | number | null | undefined): number | null {
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
  const titleMonth = `${match[1][0].toUpperCase()}${match[1].slice(1).toLowerCase()}`
  const month = MONTHS[titleMonth]
  const numericYear = Number(match[2])
  const year = match[2].length === 2 ? 2000 + numericYear : numericYear
  return month && Number.isInteger(year) ? { year, month } : null
}

export function normalizeIceSymbol(label: string): string | null {
  const parsed = parseContractMonth(label)
  if (!parsed) return null
  const month = Object.entries(MONTHS).find(([, value]) => value === parsed.month)?.[0]
  const code = month ? MONTH_CODES[month] : null
  return code ? `KC${code}${String(parsed.year).slice(-2)}` : null
}

export function calculateCurveShape(frontPrice: number, nextPrice: number): { spread: number; shape: CurveShape } {
  const spread = Number((nextPrice - frontPrice).toFixed(6))
  return { spread, shape: spread > 0 ? 'Contango' : spread < 0 ? 'Backwardation' : 'Flat' }
}

export function calculateTotalVolume(contracts: IceCoffeeContract[]): number {
  return contracts.reduce((total, contract) => total + contract.volume, 0)
}

export function sortCoffeeContracts(contracts: IceCoffeeContract[]): IceCoffeeContract[] {
  return [...contracts].sort((left, right) => {
    const leftMonth = parseContractMonth(left.contract)
    const rightMonth = parseContractMonth(right.contract)
    if (!leftMonth || !rightMonth) return 0
    return leftMonth.year - rightMonth.year || leftMonth.month - rightMonth.month
  })
}

type ParsedIceRow = { cells: string[]; sourceUrl?: string }

export function parseRenderedIceRows(rows: string[][] | ParsedIceRow[], headers?: string[]): IceCoffeeContract[] {
  const normalizedHeaders = headers?.map((header) => header.toLowerCase().replace(/[^a-z%]/g, ''))
  const indexOfHeader = (...names: string[]) => normalizedHeaders?.findIndex((header) => names.includes(header)) ?? -1
  const lastIndex = indexOfHeader('last', 'price')
  const changeIndex = indexOfHeader('change', 'netchange')
  const percentIndex = indexOfHeader('%change', 'percentchange', 'change%')
  const volumeIndex = indexOfHeader('volume', 'vol')
  const contracts: IceCoffeeContract[] = []
  for (const row of rows) {
    const cells = Array.isArray(row) ? row : row.cells
    const sourceUrl = Array.isArray(row) ? undefined : row.sourceUrl
    const contractIndex = cells.findIndex((cell) => parseContractMonth(cell) !== null)
    if (contractIndex < 0) continue

    const contract = cells[contractIndex].trim()
    const price = parseNumeric(cells[lastIndex >= 0 ? lastIndex : contractIndex + 1])
    const volumeCell = cells[volumeIndex >= 0 ? volumeIndex : cells.length - 1]?.trim() ?? ''
    const volume = volumeCell === '-' || volumeCell === '' ? 0 : parseNumeric(volumeCell)
    if (price === null || price <= 0 || volume === null || volume < 0) continue

    contracts.push({
      contract,
      price,
      volume,
      priceChange: changeIndex >= 0 ? parseNumeric(cells[changeIndex]) : null,
      percentChange: percentIndex >= 0 ? parseNumeric(cells[percentIndex]) : null,
      priceAsOf: normalizeIceTimestamp(cells[contractIndex + 2] ?? ''),
      symbol: normalizeIceSymbol(contract) ?? undefined,
      sourceName: 'ICE',
      sourceUrl,
    })
  }

  const unique = Array.from(new Map(contracts.map((contract) => [contract.contract, contract])).values())
  const sorted = sortCoffeeContracts(unique)
  if (sorted.length < 2) throw new Error('ICE rendered table parsing failed: fewer than two valid Coffee C contracts')
  return sorted
}

export function normalizeMarketDate(timestamp: string | null): string {
  if (!timestamp) return new Date().toISOString().slice(0, 10)
  const normalized = normalizeIceTimestamp(timestamp) ?? timestamp
  const parsed = Date.parse(normalized)
  return Number.isFinite(parsed) ? new Date(parsed).toISOString().slice(0, 10) : timestamp.slice(0, 10)
}

function normalizeIceTimestamp(timestamp: string): string | null {
  const trimmed = timestamp.trim()
  if (!trimmed) return null
  const match = trimmed.match(/^(\d{1,2}\/\d{1,2}\/\d{4})(\d{1,2}:\d{2}\s*[AP]M)$/i)
  if (match) {
    const parsed = Date.parse(`${match[1]} ${match[2]} GMT`)
    return Number.isFinite(parsed) ? new Date(parsed).toISOString() : `${match[1]} ${match[2]}`
  }
  const parsed = Date.parse(trimmed)
  return Number.isFinite(parsed) ? new Date(parsed).toISOString() : trimmed
}
