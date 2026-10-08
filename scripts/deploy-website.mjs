// Manual website deploy: push the current branch, then trigger the
// GitHub Actions workflow that publishes website/ to GitHub Pages.
import { execSync } from "node:child_process";

function run(cmd, opts = {}) {
  execSync(cmd, { stdio: "inherit", ...opts });
}

const branch = execSync("git branch --show-current").toString().trim();
if (!branch) {
  console.error("Not on a git branch — nothing to deploy.");
  process.exit(1);
}

console.log(`Pushing ${branch}...`);
run("git push origin HEAD");

console.log("Triggering the Deploy website workflow...");
run(`gh workflow run deploy-website.yml --ref ${branch}`);

console.log("\nDeploy started. Watch it with:  gh run watch");
console.log("Live at: https://zetawave.github.io/eqt/");
