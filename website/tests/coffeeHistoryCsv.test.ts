import assert from 'node:assert/strict'
import test from 'node:test'
import { compareCoffeeHistoryRows, parseCoffeeHistoryCsv, parseCoffeeHistoryCsvReport } from '../lib/external/coffeeHistoryCsv'

test('parses quoted CSV and sorts dates chronologically', () => {
  const rows = parseCoffeeHistoryCsv([
    'Date,Price,Open,High,Low,"Vol.","Change %"',
    '01/02/2025,200,199,202,198,"12.5K","0.50%"',
    '12/31/2024,198,197,200,196,"10K","-0.25%"',
  ].join('\n'))

  assert.deepEqual(rows.map((row) => row.date), ['2024-12-31', '2025-01-02'])
  assert.equal(rows[1].price, 200)
})

test('rejects malformed OHLC rows', () => {
  assert.throws(
    () => parseCoffeeHistoryCsv('Date,Price,Open,High,Low\n01/02/2025,200,,202,198'),
    /invalid date or OHLC value/,
  )
})

test('reports rejected rows without discarding valid missing-period rows', () => {
  const report = parseCoffeeHistoryCsvReport([
    'Date,Price,Open,High,Low,Vol.,Change %',
    '06/10/2025,360,359,362,358,"12.1K","0.2%"',
    'not-a-date,broken,359,362,358,"bad","bad"',
  ].join('\n'))

  assert.equal(report.rows.length, 1)
  assert.equal(report.rejected.length, 1)
  assert.equal(report.rejected[0].line, 3)
})

test('flags a material provider mismatch while tolerating small differences', () => {
  const existing = parseCoffeeHistoryCsv('Date,Price,Open,High,Low\n06/10/2025,360,359,362,358')
  const close = parseCoffeeHistoryCsv('Date,Price,Open,High,Low\n06/10/2025,361,360,363,359')
  const mismatch = parseCoffeeHistoryCsv('Date,Price,Open,High,Low\n06/10/2025,900,900,901,899')
  assert.equal(compareCoffeeHistoryRows(existing, close).compatible, true)
  assert.equal(compareCoffeeHistoryRows(existing, mismatch).compatible, false)
})