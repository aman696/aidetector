// Set this flag only after the hosted detector has passed a real inference check.
export const DEMO_ENABLED = process.env.DEMO_ENABLED === 'true';
export const DEMO_API_ORIGIN = process.env.DEMO_API_ORIGIN || '';
