import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const skillDir =
  process.env.PRESENTATIONS_SKILL_DIR ||
  "/Users/baiduli/.codex/plugins/cache/openai-primary-runtime/presentations/26.601.10930/skills/presentations";
const builder = path.join(skillDir, "scripts", "build_artifact_deck.mjs");
const deckBasename = process.env.DECK_BASENAME || "gradcam-editable";
const workspace = path.join(repoRoot, "outputs", "manual-deck-build", "presentations", deckBasename);
const slidesDir = path.join(repoRoot, "deck", "slides");
const outputDir = path.join(repoRoot, "artifacts", "deck");
const previewDir = path.join(outputDir, `preview-${deckBasename}`);
const layoutDir = path.join(outputDir, `layout-${deckBasename}`);
const out = path.join(outputDir, `${deckBasename}.pptx`);
const contactSheet = path.join(outputDir, `${deckBasename}-contact-sheet.png`);
const manifest = path.join(outputDir, `${deckBasename}-artifact-build-manifest.json`);
const python = process.env.PYTHON || path.join(repoRoot, ".venv", "bin", "python");

const requiredFigurePaths = [
  "artifacts/figures/method_grid.png",
  "artifacts/figures/class_discriminability.png",
  "artifacts/figures/notgradcam_decomposition.png",
  "artifacts/figures/gradcam_decomposition.png",
  "artifacts/figures/guided_gradcam_decomposition.png",
  "artifacts/figures/layercam_decomposition.png",
  "artifacts/figures/occlusion_decomposition.png",
  "artifacts/figures/integrated_gradients_decomposition.png",
  "artifacts/figures/integrated_gradcam_decomposition.png",
];

function assertExperimentFigures() {
  const figureManifestPath = path.join(repoRoot, "artifacts", "figures", "manifest.json");
  if (!fs.existsSync(figureManifestPath)) {
    throw new Error(
      `Missing figure provenance manifest: ${figureManifestPath}\n` +
        "Run `uv run gradcam-repro all` before building the deck.",
    );
  }
  const figureManifest = JSON.parse(fs.readFileSync(figureManifestPath, "utf8"));
  const listedPaths = new Set((figureManifest.figures || []).map((figure) => figure.path));
  for (const relativePath of requiredFigurePaths) {
    const absolutePath = path.join(repoRoot, relativePath);
    if (!fs.existsSync(absolutePath)) {
      throw new Error(`Missing generated experiment figure: ${absolutePath}`);
    }
    if (!listedPaths.has(relativePath)) {
      throw new Error(
        `Deck figure is not listed in experiment manifest: ${relativePath}\n` +
          "Regenerate figures with `uv run gradcam-repro demo` or `uv run gradcam-repro all`.",
      );
    }
  }
}

assertExperimentFigures();

fs.mkdirSync(outputDir, { recursive: true });

const result = spawnSync(
  process.execPath,
  [
    builder,
    "--workspace",
    workspace,
    "--slides-dir",
    slidesDir,
    "--out",
    out,
    "--preview-dir",
    previewDir,
    "--layout-dir",
    layoutDir,
    "--contact-sheet",
    contactSheet,
    "--manifest",
    manifest,
    "--slide-count",
    "19",
    "--slide-size",
    "1280x720",
  ],
  {
    cwd: repoRoot,
    encoding: "utf8",
    stdio: "inherit",
    env: { ...process.env, PYTHON: python },
  },
);

if (result.status !== 0) {
  process.exit(result.status ?? 1);
}
