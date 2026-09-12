import assert from 'node:assert/strict'
import test from 'node:test'
import {
  calculateCurveShape,
  calculateTotalVolume,
  derivePriceChangeFromPercent,
  parseContractMonth,
  parseRenderedIceRows,
  sortContractRecords,
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

test('sorts the complete contract list before pagination order is applied', () => {
  const sorted = sortContractRecords([
    { label: 'Sep26', symbol: 'KCU26' },
    { label: 'Sep27', symbol: 'KCU27' },
    { label: 'Dec26', symbol: 'KCZ26' },
    { label: 'Dec27', symbol: 'KCZ27' },
    { label: 'Mar27', symbol: 'KCH27' },
  ])
  assert.deepEqual(sorted.map((contract) => contract.label), ['Sep26', 'Dec26', 'Mar27', 'Sep27', 'Dec27'])
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

test('parses ICE semantic headers for Last, Change, and Volume', () => {
  const contracts = parseRenderedIceRows([
    ['Sep26', '313.50', '+1.25', '0.40', '5'],
    ['Dec26', '284.25', '-0.50', '-0.18', '13,995'],
  ], ['Contract', 'Last', 'Change', '% Change', 'Volume'])

  assert.equal(contracts[0].price, 313.5)
  assert.equal(contracts[0].priceChange, 1.25)
  assert.equal(contracts[0].percentChange, 0.4)
  assert.equal(contracts[1].volume, 13995)
})

test('derives absolute ICE change from percent change when ICE omits Change', () => {
  const contracts = parseRenderedIceRows([
    ['Sep26', '313.50', '9/11/2026 3:33 PM', '-0.571', '5'],
    ['Dec26', '284.25', '9/11/2026 5:29 PM', '-1.353', '13995'],
  ], ['Contract', 'Last', 'Time(GMT)', '% Change', 'Volume'])
  assert.equal(contracts[0].priceChange, derivePriceChangeFromPercent(313.5, -0.571))
  assert.equal(contracts[0].priceChangeDerived, true)
  assert.equal(contracts[0].priceChange?.toFixed(2), '-1.80')
  assert.equal(derivePriceChangeFromPercent(313.5, 0)?.toFixed(2), '0.00')
  assert.equal(derivePriceChangeFromPercent(313.5, 0.75)! > 0, true)
})

test('preserves an ICE contract source href when supplied', () => {
  const contracts = parseRenderedIceRows([
    { cells: ['Sep26', '313.50', '+1.25', '0.40', '5'], sourceUrl: 'https://www.ice.com/contract/sep26' },
    { cells: ['Dec26', '284.25', '-0.50', '-0.18', '13,995'] },
  ], ['Contract', 'Last', 'Change', '% Change', 'Volume'])

  assert.equal(contracts[0].sourceUrl, 'https://www.ice.com/contract/sep26')
  assert.equal(contracts[1].sourceUrl, undefined)
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
