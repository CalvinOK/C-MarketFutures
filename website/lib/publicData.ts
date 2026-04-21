import { readFile } from 'node:fs/promises'
import path from 'node:path'

const PUBLIC_DATA_DIR = path.join(process.cwd(), 'public', 'data')

function resolveDataPath(fileName: string): string {
  return path.join(PUBLIC_DATA_DIR, fileName)
}

export async function readPublicDataText(fileName: string): Promise<string | null> {
  try {
    return await readFile(resolveDataPath(fileName), 'utf-8')
  } catch {
    return null
  }
}

export async function readPublicDataJson<T>(fileName: string): Promise<T | null> {
  const text = await readPublicDataText(fileName)
  if (!text) return null

  try {
    return JSON.parse(text) as T
  } catch {
    return null
  }
}

export function extractAsOfDateFromCsv(csvText: string): string | null {
  const lines = csvText.split(/\r?\n/).filter((line) => line.trim())
  if (lines.length < 2) return null

  const header = lines[0].split(',').map((value) => value.trim().toLowerCase())
  const asOfIndex = header.indexOf('as_of_date')
  if (asOfIndex < 0) return null

  const firstRow = lines[1].split(',').map((value) => value.trim())
  return firstRow[asOfIndex] || null
}
