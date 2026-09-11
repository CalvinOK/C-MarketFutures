import { readFile } from 'node:fs/promises'
import { parseCoffeeHistoryCsvReport } from '@/lib/external/coffeeHistoryCsv'
import {
  getCoffeeMarketHistoryRow,
  upsertCoffeeMarketHistoryRow,
  type CoffeeMarketHistoryRow,
} from '@/lib/supabaseServer'

function loadLocalEnv(): void {
  for (const path of ['.env.local', '.env']) {
    try {
      const text = require('node:fs').readFileSync(path, 'utf8') as string
      for (const line of text.split(/\r?\n/)) {
        const match = line.match(/^([A-Z0-9_]+)=(.*)$/)
        if (match && !process.env[match[1]]) process.env[match[1]] = match[2].replace(/^['"]|['"]$/g, '')
      }
    } catch { /* optional local environment file */ }
  }
}

function dedupeRows(rows: CoffeeMarketHistoryRow[]): CoffeeMarketHistoryRow[] {
  return [...new Map(rows.map((row) => [row.date, row])).values()].sort((a, b) => a.date.localeCompare(b.date))
}

async function main(): Promise<void> {
  loadLocalEnv()
  const args = process.argv.slice(2)
  const dryRun = args.includes('--dry-run')
  const filePaths = args.filter((arg) => !arg.startsWith('--'))
  if (filePaths.length === 0) throw new Error('Usage: npm run import:coffee-history -- [--dry-run] /path/to/file.csv [/path/to/another.csv]')

  const reports = await Promise.all(filePaths.map(async (filePath) =>
    parseCoffeeHistoryCsvReport(await readFile(filePath, 'utf8')),
  ))
  const parsedRows = reports.flatMap((report) => report.rows)
  const rejectedRows = reports.flatMap((report) => report.rejected)
  const rows = dedupeRows(parsedRows)
  const summary = {
    sourceFiles: filePaths,
    totalRowsRead: parsedRows.length + rejectedRows.length,
    accepted: parsedRows.length,
    rejected: rejectedRows.length,
    duplicateCsvDates: parsedRows.length - rows.length,
    overlapWithSupabase: 0,
    expectedInserted: 0,
    expectedUpdated: 0,
    expectedUnchanged: 0,
    inserted: 0,
    updated: 0,
    unchanged: 0,
    firstDate: rows[0]?.date ?? null,
    lastDate: rows.at(-1)?.date ?? null,
  }

  for (const row of rows) {
    const existing = await getCoffeeMarketHistoryRow(row.date)
    if (existing) {
      summary.overlapWithSupabase += 1
      if (JSON.stringify(existing) === JSON.stringify(row)) {
        summary.expectedUnchanged += 1
      } else {
        summary.expectedUpdated += 1
      }
    } else {
      summary.expectedInserted += 1
    }

    if (!dryRun) {
      if (existing && JSON.stringify(existing) === JSON.stringify(row)) {
        summary.unchanged += 1
      } else if (await upsertCoffeeMarketHistoryRow(row)) {
        summary.updated += 1
      } else {
        summary.inserted += 1
      }
    }
  }

  console.log(JSON.stringify({ dryRun, ...summary, rejectedRows }, null, 2))
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error)
  process.exitCode = 1
})