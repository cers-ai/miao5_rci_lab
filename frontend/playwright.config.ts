import { defineConfig } from '@playwright/test';
export default defineConfig({ testDir: './tests', timeout: 180000, workers: 1, reporter: [['list'], ['json', { outputFile: '../workspace/logs/browser-tests.json' }]], use: { baseURL: 'http://localhost:3000', headless: true, viewport: { width: 1440, height: 1000 }, screenshot: 'only-on-failure', trace: 'retain-on-failure' } });
