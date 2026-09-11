import React from "react";

export function Tag({ children, variant = "default" }: { children: React.ReactNode, variant?: "live"|"demo"|"default" }) {
  const styles = {
    live: "text-[#4DE87F] border-[#4DE87F]",
    demo: "text-[#E8B84D] border-[#E8B84D]",
    default: "text-[#5E7B96] border-[#2D5578]",
  };
  return (
    <span className={`font-mono text-[9px] px-1.5 py-0.5 border tracking-wide uppercase ${styles[variant]}`}>
      {children}
    </span>
  );
}
