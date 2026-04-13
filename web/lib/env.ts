const WORKER_REQUIRED_VARS = [
  'DATABASE_URL',
  'NEWS_API_KEY',
  'GEMINI_API_KEY',
] as const

const CONTRACTS_DATA_SOURCE_VARS = ['DATABENTO_API_KEY', 'ALPHAVANTAGE_API_KEY'] as const

export function validateWorkerEnv(): void {
  const missing = WORKER_REQUIRED_VARS.filter((key) => !process.env[key])
  if (missing.length > 0) {
    throw new Error(`Missing required env vars: ${missing.join(', ')}`)
  }

  const hasContractsSource = CONTRACTS_DATA_SOURCE_VARS.some((key) => !!process.env[key])
  if (!hasContractsSource) {
    throw new Error(
      `At least one contracts data source is required: ${CONTRACTS_DATA_SOURCE_VARS.join(' or ')}`,
    )
  }
}
