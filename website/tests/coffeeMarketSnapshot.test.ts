import assert from 'node:assert/strict'
import test from 'node:test'
import {
  calculateCurveShape,
  calculateTotalVolume,
  parseContractMonth,
  parseRenderedIceRows,
} from '../lib/coffeeMarketSnapshot'

test('orders Coffee C contract months chronologically', () => {
  const contracts = parseRenderedIceRows([
    ['Jul27', '200.00', '14:02:00', '+0.10', '1,200'],
    ['Dec26', '210.00', '14:02:00', '-0.10', '19,113'],
    ['Mar27', '205.00', '14:02:00', '0.00', '-'],
  ])

  assert.deepEqual(contracts.map((contract) => contract.contract), ['Dec26', 'Mar27', 'Jul27'])
  assert.deepEqual(parseContractMonth('May26'), { year: 2026, month: 5 })
})

test('parses comma-separated and blank volumes', () => {
  const contracts = parseRenderedIceRows([
    ['Dec26', '291.15', '14:02:00', '+0.10', '19,113'],
    ['Mar27', '282.75', '14:02:00', '0.00', '-'],
  ])

  assert.equal(contracts[0].volume, 19113)
  assert.equal(contracts[1].volume, 0)
  assert.equal(calculateTotalVolume(contracts), 19113)
})

test('calculates contango, backwardation, and flat curves', () => {
  assert.deepEqual(calculateCurveShape(100, 101), { spread: 1, shape: 'Contango' })
  assert.deepEqual(calculateCurveShape(100, 99), { spread: -1, shape: 'Backwardation' })
  assert.deepEqual(calculateCurveShape(100, 100), { spread: 0, shape: 'Flat' })
})

test('rejects malformed ICE tables instead of returning fake data', () => {
  assert.throws(
    () => parseRenderedIceRows([['Not a contract', '291.15', '14:02:00', '0.00', '100']]),
    /fewer than two valid Coffee C contracts/,
  )
})
