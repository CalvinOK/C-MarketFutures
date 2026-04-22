import type { Metadata } from "next";
import "./globals.css";

const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL ??
  (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : "http://localhost:3000");

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: "C-Market Futures | ICE Arabica Coffee Dashboard",
  description:
    "Track ICE Arabica coffee futures contracts, forward curve, and market context.",
  icons: {
    icon: "/bond-small-icon.png",
    shortcut: "/bond-small-icon.png",
    apple: "/bond-small-icon.png",
  },
  openGraph: {
    title: "C-Market Futures | ICE Arabica Coffee Dashboard",
    description:
      "Track ICE Arabica coffee futures contracts, forward curve, and market context.",
    images: [
      {
        url: "/bond-logo-navy.png",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "C-Market Futures | ICE Arabica Coffee Dashboard",
    description:
      "Track ICE Arabica coffee futures contracts, forward curve, and market context.",
    images: ["/bond-logo-navy.png"],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
