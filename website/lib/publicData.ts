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
