import type { NextConfig } from "next";

const nextConfig: NextConfig = {
	serverExternalPackages: ['@sparticuz/chromium'],
	outputFileTracingIncludes: {
		'/api/coffee/market-snapshot': [
			'./node_modules/@sparticuz/chromium/bin/**/*',
		],
	},
};

export default nextConfig;
