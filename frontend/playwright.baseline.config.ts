import { defineConfig } from '@playwright/test';
import path from 'node:path';
import existing from './playwright.config';

if (!process.env.MIAOWU_BROWSER_EVIDENCE_DIR || !process.env.MIAOWU_WORKSPACE) {
  throw new Error('Use scripts/run_browser_baseline.py to run against an isolated workspace.');
}
export default defineConfig(existing, {
  outputDir: path.join(process.env.MIAOWU_BROWSER_EVIDENCE_DIR, 'test-results'),
  reporter: [['list'], ['json', { outputFile: path.join(process.env.MIAOWU_BROWSER_EVIDENCE_DIR, 'browser-tests.json') }]],
});
