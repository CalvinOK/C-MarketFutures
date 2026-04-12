import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "C-Market Futures | ICE Arabica Coffee Dashboard",
  description:
    "Track ICE Arabica coffee futures contracts, forward curve, and market context.",
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
