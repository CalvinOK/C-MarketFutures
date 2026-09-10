import assert from 'node:assert/strict'
import test from 'node:test'
import {
  calculateCurveShape,
  calculateTotalVolume,
  parseContractMonth,
  parseIceCoffeeContracts,
} from '../lib/coffeeMarketSnapshot'

test('orders Coffee C contract months chronologically', () => {
  const contracts = parseIceCoffeeContracts(`
    <table><tr><th>Contract</th><th>Last</th><th>Volume</th></tr>
    <tr><td>Jul27</td><td>200.00</td><td>1,200</td></tr>
    <tr><td>Dec26</td><td>210.00</td><td>19,113</td></tr>
    <tr><td>Mar27</td><td>205.00</td><td>-</td></tr></table>
  `)

  assert.deepEqual(contracts.map((contract) => contract.contract), ['Dec26', 'Mar27', 'Jul27'])
  assert.deepEqual(parseContractMonth('May26'), { year: 2026, month: 5 })
})

test('parses comma-separated and blank volumes', () => {
  const contracts = parseIceCoffeeContracts(`
    <table><tr><th>Contract</th><th>Last Price</th><th>Volume</th></tr>
    <tr><td>Dec26</td><td>291.15</td><td>19,113</td></tr>
    <tr><td>Mar27</td><td>282.75</td><td>-</td></tr></table>
  `)

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
    () => parseIceCoffeeContracts('<table><tr><th>Contract</th><th>Last</th><th>Volume</th></tr></table>'),
    /fewer than two valid contract rows/,
  )
})
