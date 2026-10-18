import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = "C:/Users/cs768/Downloads/index (2).html";
const destination = resolve(root, "public/dashboard.html");

if (!existsSync(source)) {
  throw new Error(`Source HTML not found: ${source}`);
}

mkdirSync(dirname(destination), { recursive: true });
copyFileSync(source, destination);
console.log(`Copied dashboard source to ${destination}`);
