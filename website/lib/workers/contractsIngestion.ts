import { getCoffeeContracts } from '@/lib/coffeeContracts'

/** Compatibility entrypoint; the cache-aware service owns provider access. */
export async function runContractsIngestion(): Promise<void> {
  const result = await getCoffeeContracts()
  console.log(`[coffee-contracts] worker completed source=${result.metadata.source} count=${result.contracts.length}`)
}
