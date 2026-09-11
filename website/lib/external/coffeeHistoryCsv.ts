import { fetchWithTimeout } from '@/lib/http'
import type { CoffeeMarketHistoryRow } from '@/lib/supabaseServer'

function csvFields(line: string): string[] {
  const fields: string[] = []
  let field = ''
  let quoted = false

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index]
    if (character === '"') {
      if (quoted && line[index + 1] === '"') {
        field += '"'
        index += 1
      } else {
        quoted = !quoted
      }
    } else if (character === ',' && !quoted) {
      fields.push(field.trim())
      field = ''
    } else {
      field += character
    }
  }

  if (quoted) throw new Error('CSV contains an unterminated quoted field')
  fields.push(field.trim())
  return fields
}

function normalizeHeader(value: string): string {
  return value.toLowerCase().replace(/[%.]/g, '').replace(/\s+/g, '')
}

function parseDate(value: string): string | null {
  const slash = value.trim().match(/^(\d{2})\/(\d{2})\/(\d{4})$/)
  if (slash) return `${slash[3]}-${slash[1]}-${slash[2]}`
  if (/^\d{4}-\d{2}-\d{2}$/.test(value.trim())) return value.trim()
  return null
}

function parseNumber(value: string): number | null {
  const parsed = Number(value.replace(/,/g, '').replace(/%$/, '').trim())
  return Number.isFinite(parsed) ? parsed : null
}

export function parseCoffeeHistoryCsv(csv: string): CoffeeMarketHistoryRow[] {
  const report = parseCoffeeHistoryCsvReport(csv)
  if (report.rejected.length > 0) {
    throw new Error(`CSV row ${report.rejected[0].line} is invalid: ${report.rejected[0].reason}`)
  }
  return report.rows
}

export function parseCoffeeHistoryCsvReport(csv: string): {
  rows: CoffeeMarketHistoryRow[]
  rejected: Array<{ line: number; reason: string }>
} {
  const lines = csv.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  if (lines.length < 2) throw new Error('CSV did not contain a data row')

  const headers = csvFields(lines[0]).map(normalizeHeader)
  const indexOf = (...names: string[]) => headers.findIndex((header) => names.includes(header))
  const dateIndex = indexOf('date')
  const priceIndex = indexOf('price', 'close')
  const openIndex = indexOf('open')
  const highIndex = indexOf('high')
  const lowIndex = indexOf('low')
  const volumeIndex = indexOf('vol', 'volume')
  const changeIndex = indexOf('change', 'changepct')

  if ([dateIndex, priceIndex, openIndex, highIndex, lowIndex].some((index) => index < 0)) {
    throw new Error('CSV is missing Date, Price/Close, Open, High, or Low columns')
  }

  const rows: CoffeeMarketHistoryRow[] = []
  const rejected: Array<{ line: number; reason: string }> = []
  lines.slice(1).forEach((line, rowIndex) => {
    try {
      const fields = csvFields(line)
      const date = parseDate(fields[dateIndex] ?? '')
      const values = [priceIndex, openIndex, highIndex, lowIndex].map((index) => parseNumber(fields[index] ?? ''))
      if (!date || values.some((value) => value === null || value <= 0)) {
        throw new Error('invalid date or OHLC value')
      }
      rows.push({
        date,
        price: values[0] as number,
        open: values[1] as number,
        high: values[2] as number,
        low: values[3] as number,
        volume: volumeIndex >= 0 ? fields[volumeIndex] ?? '' : '',
        changePercent: changeIndex >= 0 ? fields[changeIndex] ?? '' : '',
      })
    } catch (error: unknown) {
      rejected.push({ line: rowIndex + 2, reason: error instanceof Error ? error.message : 'invalid row' })
    }
  })
  return { rows: rows.sort((left, right) => left.date.localeCompare(right.date)), rejected }
}

export function compareCoffeeHistoryRows(
  existing: CoffeeMarketHistoryRow[],
  incoming: CoffeeMarketHistoryRow[],
): { overlapRows: number; medianPercentDifference: number | null; maxAbsoluteDifference: number | null; compatible: boolean | null } {
  const existingByDate = new Map(existing.map((row) => [row.date, row]))
  const differences: number[] = []
  const absoluteDifferences: number[] = []
  for (const row of incoming) {
    const prior = existingByDate.get(row.date)
    if (!prior) continue
    for (const field of ['price', 'open', 'high', 'low'] as const) {
      const absolute = Math.abs(row[field] - prior[field])
      absoluteDifferences.push(absolute)
      differences.push((absolute / prior[field]) * 100)
    }
  }
  if (differences.length === 0) return { overlapRows: 0, medianPercentDifference: null, maxAbsoluteDifference: null, compatible: null }
  const sortedPercent = [...differences].sort((a, b) => a - b)
  return {
    overlapRows: differences.length / 4,
    medianPercentDifference: sortedPercent[Math.floor(sortedPercent.length / 2)],
    maxAbsoluteDifference: Math.max(...absoluteDifferences),
    compatible: sortedPercent[Math.floor(sortedPercent.length / 2)] <= 2 && Math.max(...absoluteDifferences) <= 25,
  }
}

export async function fetchLatestCoffeeHistoryCsv(): Promise<string> {
  const configuredUrl = process.env.COFFEE_HISTORY_CSV_URL?.trim()
  const baseUrl = process.env.MARKET_API_BASE_URL?.trim().replace(/\/$/, '')
    || (process.env.NODE_ENV !== 'production' ? 'http://127.0.0.1:8000' : '')
  const url = configuredUrl || (baseUrl ? `${baseUrl}/api/coffee/history/latest.csv` : '')
  if (!url) throw new Error('COFFEE_HISTORY_CSV_URL or MARKET_API_BASE_URL is required')

  const response = await fetchWithTimeout(url, {
    headers: {
      Accept: 'text/csv',
      ...(process.env.MARKET_API_TOKEN
        ? { Authorization: `Bearer ${process.env.MARKET_API_TOKEN}` }
        : {}),
    },
    cache: 'no-store',
  }, 20_000)
  if (!response.ok) throw new Error(`Coffee history provider returned HTTP ${response.status}`)
  return response.text()
}