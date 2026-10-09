import { proxyDemo } from '../../server/demo-proxy.mjs';

export const onRequest = context => proxyDemo(context);
