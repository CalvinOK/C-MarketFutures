import assert from 'node:assert/strict'
import test from 'node:test'
import { getExpectedCoffeeTradingDate, toApiContract } from '../lib/coffeeContracts'
import { isRefreshCooldownActive } from '../lib/refreshState'
import { MARKET_SNAPSHOT_TTL_MS } from '../lib/coffeeMarketSnapshotService'
import { normalizeIceSymbol } from '../lib/coffeeMarketSnapshot'

test('daily contract date rolls weekend requests back to Friday', () => {
  assert.equal(getExpectedCoffeeTradingDate(new Date('2026-09-12T15:00:00Z')), '2026-09-11')
  assert.equal(getExpectedCoffeeTradingDate(new Date('2026-09-13T15:00:00Z')), '2026-09-11')
  assert.equal(getExpectedCoffeeTradingDate(new Date('2026-09-14T15:00:00Z')), '2026-09-14')
})

test('ICE pricing TTL is rolling 15 minutes', () => {
  assert.equal(MARKET_SNAPSHOT_TTL_MS, 15 * 60 * 1000)
})

test('ICE month labels normalize to stable KC symbols', () => {
  assert.equal(normalizeIceSymbol('Sep26'), 'KCU26')
  assert.equal(normalizeIceSymbol('Dec26'), 'KCZ26')
  assert.equal(normalizeIceSymbol('Mar27'), 'KCH27')
})

test('failed refresh cooldown suppresses attempts for 15 minutes', () => {
  const attemptedAt = new Date('2026-09-11T12:00:00Z').toISOString()
  assert.equal(isRefreshCooldownActive({ lastAttemptAt: attemptedAt, lastSuccessAt: null, status: 'failed' }, Date.parse('2026-09-11T12:05:00Z')), true)
  assert.equal(isRefreshCooldownActive({ lastAttemptAt: attemptedAt, lastSuccessAt: null, status: 'failed' }, Date.parse('2026-09-11T12:16:00Z')), false)
})

test('contract normalization preserves zero price changes, volume, and open interest', () => {
  const contract = toApiContract({
    symbol: 'KCK26', contract_name: 'KCK26', month_code: 'K', expiration_date: '2026-05-19',
    settlement: 0, last_price: 0, price_change: 0, percent_change: 0,
    volume: 0, open_interest: 0, provider_timestamp: null, source: 'databento',
    trade_date: '2026-09-11', fetched_at: '2026-09-11T12:00:00Z',
  })
  assert.equal(contract.lastPrice, 0)
  assert.equal(contract.priceChange, 0)
  assert.equal(contract.volume, 0)
  assert.equal(contract.openInterest, 0)
})

test('ICE Last values do not become settlement values', () => {
  const contract = toApiContract({
    symbol: 'KCU26', contract_name: 'Sep26', expiration_date: null,
    last_price: 313.5, price_change: 1.25, percent_change: 0.4,
    volume: 24527, open_interest: null, settlement: null,
    provider_timestamp: '2026-09-11T15:33:00Z', source: 'ice',
    trade_date: '2026-09-11', fetched_at: '2026-09-11T20:00:00Z',
  })
  assert.equal(contract.lastPrice, 313.5)
  assert.equal(contract.settlementPrice, null)
  assert.equal(contract.openInterest, null)
})
