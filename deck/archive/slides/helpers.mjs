import fs from "node:fs";
import path from "node:path";

import { references, slides } from "./content.mjs";

const W = 1280;
const H = 720;
const MX = 64; // page margin

// Palette — clean white content with dark bookends ("sandwich").
// Accent semantics: hot = attribution heat / signal, blue = analysis,
// green = evidence. The square chip echoes the heatmap-cell motif.
const C = {
  bg: "#FFFFFF",
  surface: "#F1F4F8",
  surfaceAlt: "#E7ECF2",
  ink: "#13181E",
  muted: "#5E6873",
  faint: "#9AA6B2",
  rule: "#E3E8EE",
  hot: "#E8590C",
  hotSoft: "#FBE6D6",
  blue: "#1F6FA8",
  blueSoft: "#DCEAF3",
  green: "#2E7D52",
  greenSoft: "#DCEDE2",
  // dark slides
  dark: "#0E141B",
  darkSurf: "#19222C",
  darkRule: "#2A3540",
  white: "#FFFFFF",
  ice: "#C7D2DD",
  iceFaint: "#7E8A96",
};

function addShape(slide, position, options = {}) {
  return slide.shapes.add({
    geometry: options.geometry || "rect",
    position,
    fill: { type: "solid", color: options.fill || C.bg },
    line: {
      style: "solid",
      fill: options.line || options.fill || C.bg,
      width: options.lineWidth ?? 0,
    },
  });
}

function addText(slide, text, position, style = {}) {
  const shape = addShape(slide, position, {
    fill: style.fill || "transparent",
    line: style.line || "transparent",
    lineWidth: style.lineWidth ?? 0,
  });
  shape.text.style = {
    typeface: style.typeface || "Aptos",
    fontSize: style.fontSize || 22,
    color: style.color || C.ink,
    bold: Boolean(style.bold),
    italic: Boolean(style.italic),
    alignment: style.alignment || "left",
    verticalAlignment: style.verticalAlignment || "top",
  };
  shape.text = text;
  return shape;
}

function addBackground(slide, dark = false) {
  addShape(slide, { left: 0, top: 0, width: W, height: H }, {
    fill: dark ? C.dark : C.bg,
    line: dark ? C.dark : C.bg,
  });
}

// The motif: a small filled square (one "heatmap cell").
function chip(slide, x, y, color, size = 13) {
  addShape(slide, { left: x, top: y, width: size, height: size }, { fill: color, line: color });
}

// Tinted surface card with a thin accent edge on the left.
function card(slide, pos, accent) {
  addShape(slide, pos, { fill: pos.fill || C.surface, line: pos.line || C.rule, lineWidth: 1 });
  if (accent) {
    addShape(slide, { left: pos.left, top: pos.top, width: 4, height: pos.height }, { fill: accent, line: accent });
  }
}

function bulletsText(bullets) {
  return bullets.map((b) => `•  ${b}`).join("\n\n");
}

function compactBulletsText(bullets) {
  return bullets.map((b) => `• ${b}`).join("\n");
}

function numberedText(items) {
  return items.map((s, i) => `${i + 1}. ${s}`).join("\n");
}

function sectionLabel(slide, label, x, y, color = C.faint, width = 240) {
  chip(slide, x, y + 3, color, 8);
  addText(slide, label, { left: x + 16, top: y, width, height: 16 }, {
    color,
    fontSize: 9.5,
    bold: true,
  });
}

function equationFontSize(text, base = 15) {
  const len = [...String(text)].length;
  if (len > 96) return base - 4;
  if (len > 72) return base - 3;
  if (len > 52) return base - 1.5;
  return base;
}

function addHeader(slide, item, number, dark = false) {
  const ink = dark ? C.white : C.ink;
  const muted = dark ? C.ice : C.muted;
  const faint = dark ? C.iceFaint : C.faint;
  chip(slide, MX, 38, C.hot);
  addText(slide, item.kicker, { left: MX + 22, top: 35, width: 700, height: 22 }, {
    color: dark ? C.ice : C.muted,
    fontSize: 12,
    bold: true,
  });
  addText(slide, `${String(number).padStart(2, "0")} / ${String(slides.length).padStart(2, "0")}`,
    { left: W - MX - 150, top: 35, width: 150, height: 20 }, {
      color: faint,
      fontSize: 12,
      bold: true,
      alignment: "right",
    });
  return { ink, muted, faint };
}

function addTitle(slide, item, y = 92, dark = false) {
  const { ink, muted } = addHeader(slide, item, slides.indexOf(item) + 1, dark);
  addText(slide, item.title, { left: MX, top: y, width: 1080, height: 104 }, {
    color: ink,
    fontSize: 38,
    bold: true,
  });
  if (item.subtitle) {
    addText(slide, item.subtitle, { left: MX, top: y + 108, width: 860, height: 72 }, {
      color: muted,
      fontSize: 21,
    });
  }
  return { ink, muted };
}

function addFooter(slide, text, dark = false) {
  if (!text) return;
  addText(slide, text, { left: MX, top: 684, width: 1000, height: 20 }, {
    color: dark ? C.iceFaint : C.faint,
    fontSize: 11,
  });
}

function addMetric(slide, label, value, x, y, accent = C.blue, w = 128) {
  addShape(slide, { left: x, top: y, width: w, height: 76 }, { fill: C.surface, line: C.rule, lineWidth: 1 });
  addShape(slide, { left: x, top: y, width: 4, height: 76 }, { fill: accent, line: accent });
  // scale the value down when it is a phrase rather than a short number
  const vlen = String(value).length;
  const vsize = vlen > 7 ? 15 : vlen > 4 ? 19 : 25;
  addText(slide, value, { left: x + 12, top: y + 10, width: w - 22, height: 36 }, {
    color: accent,
    fontSize: vsize,
    bold: true,
    alignment: "center",
    verticalAlignment: "middle",
  });
  addText(slide, label, { left: x + 12, top: y + 50, width: w - 20, height: 18 }, {
    color: C.muted,
    fontSize: 12,
    alignment: "center",
  });
}

function layoutCover(slide, item) {
  addBackground(slide, true);
  // motif strip: a row of heatmap cells, hot fading to surface
  const cellColors = [C.hot, "#C24C12", "#8A4A2A", C.darkSurf, C.darkSurf];
  cellColors.forEach((c, i) => chip(slide, MX + i * 22, 70, c, 14));
  addText(slide, item.kicker, { left: MX, top: 110, width: 500, height: 22 }, {
    color: C.hot,
    fontSize: 13,
    bold: true,
  });
  addText(slide, item.title, { left: MX, top: 184, width: 770, height: 170 }, {
    color: C.white,
    fontSize: 56,
    bold: true,
  });
  addText(slide, item.subtitle, { left: MX, top: 386, width: 760, height: 96 }, {
    color: C.ice,
    fontSize: 23,
  });
  // stat panel, right
  const px = 880;
  addShape(slide, { left: px, top: 150, width: 320, height: 300 }, { fill: C.darkSurf, line: C.darkRule, lineWidth: 1 });
  addShape(slide, { left: px, top: 150, width: 320, height: 5 }, { fill: C.hot, line: C.hot });
  const stats = [
    ["32³", "real CT patch"],
    ["1 3D CNN", "stage2 CAM tap"],
    ["7 methods", "one tradeoff sheet"],
  ];
  stats.forEach((s, i) => {
    const y = 188 + i * 88;
    addText(slide, s[0], { left: px + 28, top: y, width: 270, height: 38 }, {
      color: C.white,
      fontSize: 30,
      bold: true,
    });
    addText(slide, s[1], { left: px + 28, top: y + 42, width: 270, height: 22 }, {
      color: C.iceFaint,
      fontSize: 15,
    });
  });
  addFooter(slide, item.footer, true);
}

function layoutTwoColumn(slide, item) {
  addBackground(slide);
  addTitle(slide, item);
  const accents = [C.blue, C.green];
  const soft = [C.blueSoft, C.greenSoft];
  item.columns.forEach((col, i) => {
    const x = i === 0 ? MX : 660;
    const pos = { left: x, top: 210, width: 516, height: 300 };
    card(slide, pos, accents[i]);
    addShape(slide, { left: x + 24, top: 234, width: 30, height: 30 }, { fill: soft[i], line: soft[i] });
    chip(slide, x + 35, 245, accents[i], 8);
    addText(slide, col.title, { left: x + 68, top: 232, width: 420, height: 34 }, {
      color: accents[i],
      fontSize: 22,
      bold: true,
    });
    addText(slide, bulletsText(col.bullets), { left: x + 28, top: 290, width: 462, height: 200 }, {
      color: C.ink,
      fontSize: 17,
    });
  });
  addText(slide, item.note, { left: MX, top: 548, width: 1152, height: 60 }, {
    color: C.muted,
    fontSize: 17,
    italic: true,
  });
}

function layoutRoadmap(slide, item) {
  addBackground(slide);
  addTitle(slide, item);
  const colors = [C.hot, C.blue, C.green, C.dark];
  const soft = [C.hotSoft, C.blueSoft, C.greenSoft, C.surfaceAlt];
  const w = 252;
  const gap = 36;
  const y = 234;
  item.steps.forEach((step, i) => {
    const x = MX + i * (w + gap);
    addShape(slide, { left: x, top: y, width: w, height: 230 }, {
      fill: C.surface,
      line: C.rule,
      lineWidth: 1,
    });
    addShape(slide, { left: x, top: y, width: w, height: 5 }, {
      fill: colors[i],
      line: colors[i],
    });
    addShape(slide, { left: x + 22, top: y + 26, width: 44, height: 44 }, {
      fill: soft[i],
      line: soft[i],
    });
    addText(slide, step.label, { left: x + 22, top: y + 34, width: 44, height: 26 }, {
      color: colors[i],
      fontSize: 18,
      bold: true,
      alignment: "center",
    });
    addText(slide, step.title, { left: x + 22, top: y + 90, width: w - 44, height: 34 }, {
      color: colors[i],
      fontSize: 20,
      bold: true,
    });
    addText(slide, step.question, { left: x + 22, top: y + 132, width: w - 44, height: 48 }, {
      color: C.ink,
      fontSize: 17,
      bold: true,
    });
    addText(slide, step.pain, { left: x + 22, top: y + 188, width: w - 44, height: 28 }, {
      color: C.muted,
      fontSize: 13,
    });
    if (i < item.steps.length - 1) {
      addText(slide, "›", { left: x + w + 4, top: y + 92, width: gap - 8, height: 48 }, {
        color: C.faint,
        fontSize: 34,
        bold: true,
        alignment: "center",
      });
    }
  });
  addShape(slide, { left: MX, top: 520, width: 1152, height: 70 }, {
    fill: C.dark,
    line: C.dark,
  });
  addShape(slide, { left: MX, top: 520, width: 5, height: 70 }, {
    fill: C.hot,
    line: C.hot,
  });
  addText(slide, item.note, { left: MX + 28, top: 528, width: 1098, height: 52 }, {
    color: C.white,
    fontSize: 18,
    bold: true,
    verticalAlignment: "middle",
  });
}

function layoutComparison(slide, item) {
  addBackground(slide);
  addTitle(slide, item);
  const blocks = [item.left, item.right];
  const accents = [C.blue, C.hot];
  blocks.forEach((block, i) => {
    const x = i === 0 ? MX : 660;
    const accent = accents[i];
    const pos = { left: x, top: 210, width: 516, height: 332 };
    card(slide, pos, accent);
    addText(slide, block.title, { left: x + 26, top: 230, width: 460, height: 38 }, {
      color: accent,
      fontSize: 26,
      bold: true,
    });
    addShape(slide, { left: x + 26, top: 282, width: 464, height: 56 }, { fill: C.bg, line: C.rule, lineWidth: 1 });
    addText(slide, block.formula, { left: x + 36, top: 296, width: 444, height: 30 }, {
      color: C.ink,
      fontSize: equationFontSize(block.formula, 20),
      alignment: "center",
      typeface: "Cambria Math",
      italic: true,
    });
    if (block.symbols) {
      sectionLabel(slide, "SYMBOLS", x + 26, 354, C.faint);
      addText(slide, compactBulletsText(block.symbols), { left: x + 26, top: 378, width: 464, height: 76 }, {
        color: C.muted,
        fontSize: 11.3,
      });
    }
    addText(slide, block.body, { left: x + 26, top: block.symbols ? 462 : 356, width: 464, height: block.symbols ? 62 : 130 }, {
      color: C.muted,
      fontSize: block.symbols ? 13.6 : 18,
    });
  });
  // connector arrow
  addText(slide, "›", { left: 588, top: 320, width: 72, height: 60 }, {
    color: C.faint,
    fontSize: 44,
    bold: true,
    alignment: "center",
  });
  addText(slide, item.note, { left: MX, top: 566, width: 1152, height: 42 }, {
    color: C.ink,
    fontSize: 17,
    italic: true,
  });
}

function layoutSetup(slide, item) {
  addBackground(slide);
  addTitle(slide, item);
  card(slide, { left: MX, top: 210, width: 600, height: 360 }, C.blue);
  addText(slide, bulletsText(item.bullets), { left: MX + 28, top: 232, width: 548, height: 320 }, {
    fontSize: 15.4,
    color: C.ink,
  });
  // pipeline diagram, right
  const stages = [
    { name: "input", tensor: "C=1 | 32³", channels: 1, size: 92, kind: "map", ops: "CT patch" },
    { name: "stage1", tensor: "C=8 | 16³", channels: 8, size: 70, kind: "map", ops: "2 Conv3D + Pool", modules: "5 modules" },
    { name: "stage2", tensor: "C=16 | 8³", channels: 16, size: 52, kind: "map", tap: true, ops: "2 Conv3D + Pool", modules: "5 modules" },
    { name: "stage3", tensor: "C=32 | 8³", channels: 32, size: 52, kind: "map", ops: "3x3x3 Conv", modules: "2 modules" },
    { name: "head", tensor: "2 logits", channels: 2, size: 40, kind: "vector", ops: "global pool + FC", modules: "classifier" },
  ];
  addText(slide, "ARCHITECTURE", { left: 700, top: 214, width: 300, height: 18 }, {
    color: C.muted,
    fontSize: 12,
    bold: true,
  });
  addText(slide, "3D volume blocks; labels carry D x H x W size", { left: 700, top: 232, width: 470, height: 16 }, {
    color: C.faint,
    fontSize: 10,
  });
  const centers = [738, 840, 942, 1040, 1135];
  const centerY = 286;
  function depthSheets(channels) {
    if (channels <= 1) return 1;
    if (channels <= 8) return 3;
    if (channels <= 16) return 4;
    return 5;
  }
  stages.forEach((stage, i) => {
    const tap = Boolean(stage.tap);
    const square = stage.kind === "vector"
      ? { width: 46, height: 28 }
      : { width: stage.size, height: stage.size };
    const x = centers[i] - square.width / 2;
    const y = centerY - square.height / 2;
    if (stage.kind === "map") {
      const sheets = depthSheets(stage.channels);
      for (let layer = sheets - 1; layer >= 1; layer -= 1) {
        const offset = layer * 4;
        addShape(slide, { left: x + offset, top: y - offset, width: square.width, height: square.height }, {
          geometry: "cube",
          fill: tap ? "#FFE9DC" : "#F7FAFD",
          line: tap ? C.hot : C.rule,
          lineWidth: 1,
        });
      }
    }
    addShape(slide, { left: x, top: y, width: square.width, height: square.height }, {
      geometry: stage.kind === "map" ? "cube" : "rect",
      fill: tap ? C.hotSoft : C.surface,
      line: tap ? C.hot : C.rule,
      lineWidth: tap ? 2 : 1,
    });
    if (stage.kind === "map" && stage.size >= 52) {
      addShape(slide, { left: x + square.width - 16, top: y + 8, width: 6, height: 6 }, {
        fill: tap ? C.hot : C.faint,
        line: tap ? C.hot : C.faint,
      });
      addShape(slide, { left: x + square.width - 26, top: y + 18, width: 6, height: 6 }, {
        fill: tap ? C.hot : C.faint,
        line: tap ? C.hot : C.faint,
      });
    }
    if (stage.modules) {
      addText(slide, stage.modules, { left: centers[i] - 42, top: 326, width: 84, height: 16 }, {
        fontSize: 9.2,
        color: tap ? C.hot : C.blue,
        bold: tap,
        alignment: "center",
      });
    }
    addText(slide, stage.name, { left: centers[i] - 46, top: 348, width: 92, height: 18 }, {
      fontSize: 11.5,
      color: tap ? C.hot : C.muted,
      bold: tap,
      alignment: "center",
    });
    addText(slide, stage.tensor, { left: centers[i] - 58, top: 368, width: 116, height: 18 }, {
      fontSize: 10.8,
      color: C.ink,
      bold: true,
      alignment: "center",
    });
    addText(slide, stage.ops, { left: centers[i] - 58, top: 388, width: 116, height: 20 }, {
      fontSize: 8.8,
      color: tap ? C.hot : C.muted,
      alignment: "center",
    });
    if (i < stages.length - 1) {
      const arrowX = (centers[i] + centers[i + 1]) / 2 - 10;
      addText(slide, "›", { left: arrowX, top: 271, width: 20, height: 32 }, {
        fontSize: 22,
        color: C.faint,
        bold: true,
        alignment: "center",
      });
    }
  });
  addText(slide, "CAM tap point", { left: 886, top: 410, width: 112, height: 18 }, {
    color: C.hot,
    fontSize: 11,
    bold: true,
    alignment: "center",
  });
  item.metrics.forEach((m, i) => addMetric(slide, m[0], m[1], 700 + i * 124, 450, i === 3 ? C.hot : C.blue, 110));
}

function layoutMethod(slide, item) {
  addBackground(slide);
  addTitle(slide, item);
  const m = item.method;
  const top = 206;

  // Method identity and source card.
  const cx = MX;
  card(slide, { left: cx, top, width: 326, height: 470 }, C.hot);
  addText(slide, m.name, { left: cx + 24, top: top + 22, width: 278, height: 36 }, {
    color: C.hot,
    fontSize: m.name.length > 18 ? 22 : 25,
    bold: true,
  });
  addText(slide, m.paper, { left: cx + 24, top: top + 66, width: 278, height: 54 }, {
    color: C.muted,
    fontSize: 10.5,
  });
  sectionLabel(slide, "PAIN SOLVED", cx + 24, top + 136, C.hot);
  addText(slide, m.pain, { left: cx + 24, top: top + 160, width: 278, height: 72 }, {
    color: C.ink,
    fontSize: 12.6,
  });
  sectionLabel(slide, "HANDLE", cx + 24, top + 248, C.blue);
  addText(slide, m.handle, { left: cx + 24, top: top + 272, width: 278, height: 50 }, {
    color: C.ink,
    fontSize: 12.4,
  });
  sectionLabel(slide, "USE WHEN", cx + 24, top + 338, C.green);
  addText(slide, compactBulletsText(m.pros), { left: cx + 24, top: top + 360, width: 278, height: 42 }, {
    color: C.ink,
    fontSize: 10.5,
  });
  sectionLabel(slide, "WATCH", cx + 24, top + 404, C.hot);
  addText(slide, compactBulletsText(m.cons), { left: cx + 24, top: top + 426, width: 278, height: 42 }, {
    color: C.muted,
    fontSize: 10.2,
  });

  // Formula and notation card.
  const rx = 414;
  card(slide, { left: rx, top, width: 802, height: 146 }, C.blue);
  sectionLabel(slide, "FORMULA", rx + 24, top + 20, C.blue);
  addText(slide, m.formula, { left: rx + 24, top: top + 46, width: 440, height: 72 }, {
    color: C.ink,
    fontSize: equationFontSize(m.formula, 14),
    typeface: "Cambria Math",
    italic: true,
    verticalAlignment: "middle",
  });
  addShape(slide, { left: rx + 488, top: top + 22, width: 1, height: 104 }, {
    fill: C.rule,
    line: C.rule,
  });
  sectionLabel(slide, "SYMBOLS", rx + 510, top + 20, C.faint);
  addText(slide, compactBulletsText(m.symbols), { left: rx + 510, top: top + 48, width: 270, height: 86 }, {
    color: C.muted,
    fontSize: 8.8,
  });

  // Code and visualization flow.
  const codeY = 370;
  if (m.figure) {
    card(slide, { left: rx, top: codeY, width: 802, height: 286 }, C.green);
    sectionLabel(slide, m.figureTitle || "EXPERIMENT VISUALIZATION", rx + 24, codeY + 18, C.green, 260);
    const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..");
    const imagePath = path.join(repoRoot, m.figure);
    const imageRegion = { left: rx + 24, top: codeY + 42, width: 754, height: 218 };
    if (fs.existsSync(imagePath)) {
      const buffer = fs.readFileSync(imagePath);
      const size = pngSize(buffer) || { width: imageRegion.width, height: imageRegion.height };
      const scale = Math.min(imageRegion.width / size.width, imageRegion.height / size.height);
      const drawW = Math.round(size.width * scale);
      const drawH = Math.round(size.height * scale);
      const left = Math.round(imageRegion.left + (imageRegion.width - drawW) / 2);
      const topImage = Math.round(imageRegion.top + (imageRegion.height - drawH) / 2);
      addShape(slide, { left, top: topImage, width: drawW, height: drawH }, {
        fill: C.bg,
        line: C.rule,
        lineWidth: 1,
      });
      slide.images.add({
        dataUrl: `data:image/png;base64,${buffer.toString("base64")}`,
        position: { left, top: topImage, width: drawW, height: drawH },
      });
    } else {
      addText(slide, `Missing generated figure: ${m.figure}`, imageRegion, {
        color: C.hot,
        fontSize: 14,
        bold: true,
        alignment: "center",
        verticalAlignment: "middle",
      });
    }
    addText(slide, m.figureCaption || "", { left: rx + 24, top: codeY + 262, width: 754, height: 18 }, {
      color: C.muted,
      fontSize: 8.8,
    });
    return;
  }
  card(slide, { left: rx, top: codeY, width: 390, height: 250 }, C.dark);
  sectionLabel(slide, "MINIMAL CODE", rx + 24, codeY + 20, C.dark);
  addShape(slide, { left: rx + 24, top: codeY + 48, width: 342, height: 176 }, {
    fill: C.dark,
    line: C.darkRule,
    lineWidth: 1,
  });
  addText(slide, m.code.join("\n"), { left: rx + 36, top: codeY + 60, width: 318, height: 154 }, {
    color: C.white,
    fontSize: 8.8,
    typeface: "Consolas",
  });
  addText(slide, "All snippets map to src/gradcam_repro/attribution.py.", { left: rx + 24, top: codeY + 228, width: 342, height: 14 }, {
    color: C.faint,
    fontSize: 8.5,
  });

  const sx = rx + 414;
  card(slide, { left: sx, top: codeY, width: 388, height: 250 }, C.green);
  sectionLabel(slide, "VISUALIZATION STEPS", sx + 24, codeY + 20, C.green);
  addText(slide, numberedText(m.steps), { left: sx + 24, top: codeY + 54, width: 340, height: 150 }, {
    color: C.ink,
    fontSize: 10.8,
  });
  addShape(slide, { left: sx + 24, top: codeY + 212, width: 340, height: 1 }, {
    fill: C.rule,
    line: C.rule,
  });
  addText(slide, m.mechanism, { left: sx + 24, top: codeY + 224, width: 340, height: 22 }, {
    color: C.muted,
    fontSize: 8.8,
  });
}

function layoutFailure(slide, item) {
  addBackground(slide);
  addTitle(slide, item);
  card(slide, { left: MX, top: 210, width: 382, height: 250 }, C.hot);
  addText(slide, bulletsText(item.bullets), { left: MX + 28, top: 234, width: 326, height: 210 }, {
    color: C.ink,
    fontSize: 14.3,
  });

  // Mechanism schematic, right: an editable shape diagram rather than a static screenshot.
  const dx = 478;
  const dy = 210;
  addShape(slide, { left: dx, top: dy, width: 738, height: 250 }, { fill: C.surface, line: C.rule, lineWidth: 1 });
  sectionLabel(slide, "WHY THE MAP CAN DISAPPEAR", dx + 24, dy + 18, C.hot, 260);

  function miniHeatmap(x, y, cells, cellSize = 10) {
    cells.forEach((row, r) => {
      row.forEach((color, c) => {
        addShape(slide, { left: x + c * cellSize, top: y + r * cellSize, width: cellSize, height: cellSize }, {
          fill: color,
          line: "#FFFFFF",
          lineWidth: 0.5,
        });
      });
    });
  }

  function arrow(x, y, w = 34) {
    addText(slide, "›", { left: x, top: y, width: w, height: 34 }, {
      color: C.faint,
      fontSize: 24,
      bold: true,
      alignment: "center",
      verticalAlignment: "middle",
    });
  }

  const neg = "#315BB7";
  const negSoft = "#9CB7EA";
  const zero = "#F0F4F8";
  const posSoft = "#F4B095";
  const pos = "#D9480F";
  const blank = "#FFFFFF";
  const gradCells = [
    [neg, negSoft, zero, posSoft, pos],
    [neg, negSoft, zero, posSoft, pos],
    [neg, neg, negSoft, pos, pos],
    [neg, negSoft, zero, posSoft, pos],
    [neg, neg, negSoft, posSoft, pos],
  ];
  const weightedCells = [
    [negSoft, negSoft, neg, neg, neg],
    [negSoft, neg, neg, neg, neg],
    [negSoft, neg, neg, neg, neg],
    [negSoft, negSoft, neg, neg, neg],
    [negSoft, negSoft, negSoft, neg, neg],
  ];
  const blankCells = Array.from({ length: 5 }, () => Array.from({ length: 5 }, () => blank));

  const columns = [
    { x: dx + 34, title: "local gradients", caption: "∂yᶜ/∂Aᵏ has mixed signs", cells: gradCells },
    { x: dx + 205, title: "channel weight", caption: "αₖᶜ = meanᵢⱼ(∂yᶜ/∂Aᵏᵢⱼ) < 0" },
    { x: dx + 382, title: "weighted feature", caption: "αₖᶜAᵏ becomes negative evidence", cells: weightedCells },
    { x: dx + 574, title: "after ReLU", caption: "ReLU(Σₖ αₖᶜAᵏ) = blank", cells: blankCells, blank: true },
  ];
  columns.forEach((col) => {
    addText(slide, col.title, { left: col.x, top: dy + 54, width: 120, height: 18 }, {
      color: col.blank ? C.hot : C.ink,
      fontSize: 11.5,
      bold: true,
      alignment: "center",
    });
    if (col.cells) {
      addShape(slide, { left: col.x + 25, top: dy + 82, width: 70, height: 70 }, {
        fill: C.bg,
        line: col.blank ? C.hot : C.rule,
        lineWidth: col.blank ? 1.5 : 1,
      });
      miniHeatmap(col.x + 35, dy + 92, col.cells);
    } else {
      addShape(slide, { left: col.x + 8, top: dy + 82, width: 104, height: 70 }, {
        fill: C.bg,
        line: C.hot,
        lineWidth: 1.2,
      });
      addText(slide, "αₖᶜ < 0", { left: col.x + 12, top: dy + 92, width: 96, height: 28 }, {
        color: C.hot,
        fontSize: 20,
        bold: true,
        alignment: "center",
        verticalAlignment: "middle",
      });
      addText(slide, "one number\nfor all voxels", { left: col.x + 12, top: dy + 124, width: 96, height: 24 }, {
        color: C.muted,
        fontSize: 8.8,
        alignment: "center",
        verticalAlignment: "middle",
      });
    }
    addText(slide, col.caption, { left: col.x - 6, top: dy + 162, width: 132, height: 34 }, {
      color: C.muted,
      fontSize: 8.8,
      alignment: "center",
    });
  });
  arrow(dx + 160, dy + 100);
  arrow(dx + 335, dy + 100);
  arrow(dx + 524, dy + 100);

  addShape(slide, { left: dx + 34, top: dy + 210, width: 670, height: 1 }, {
    fill: C.rule,
    line: C.rule,
  });
  addText(slide, "Repair idea: keep spatial sign before pooling", { left: dx + 34, top: dy + 218, width: 244, height: 18 }, {
    color: C.green,
    fontSize: 11,
    bold: true,
  });
  addText(slide, "LayerCAM: ReLU(∂yᶜ/∂Aᵏᵢⱼ) ⊙ Aᵏᵢⱼ", { left: dx + 300, top: dy + 216, width: 210, height: 22 }, {
    color: C.ink,
    fontSize: 9.8,
    alignment: "center",
  });
  addText(slide, "IGC: integrate gradients along the path", { left: dx + 520, top: dy + 216, width: 178, height: 22 }, {
    color: C.ink,
    fontSize: 9.8,
    alignment: "center",
  });

  // callout band
  addShape(slide, { left: MX, top: 498, width: 1152, height: 66 }, { fill: C.dark, line: C.dark });
  addShape(slide, { left: MX, top: 498, width: 5, height: 66 }, { fill: C.hot, line: C.hot });
  addText(slide, item.callout, { left: MX + 32, top: 498, width: 1100, height: 66 }, {
    color: C.white,
    fontSize: 19,
    bold: true,
    verticalAlignment: "middle",
  });
}

// Read a PNG's pixel dimensions from its IHDR header.
function pngSize(buffer) {
  if (buffer.length < 24) return null;
  return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

function layoutImageProof(slide, item) {
  addBackground(slide);
  addTitle(slide, item, 86);
  const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..", "..");
  const imagePath = path.join(repoRoot, item.image);
  // Available content region for the figure.
  const region = { left: MX, top: 192, width: 1152, height: 430 };
  const pad = 14;
  if (fs.existsSync(imagePath)) {
    const buffer = fs.readFileSync(imagePath);
    const size = pngSize(buffer) || { width: region.width, height: region.height };
    const maxW = region.width - 2 * pad;
    const maxH = region.height - 2 * pad;
    const scale = Math.min(maxW / size.width, maxH / size.height);
    const drawW = Math.round(size.width * scale);
    const drawH = Math.round(size.height * scale);
    // Frame hugs the drawn image so wide and tall figures both look intentional.
    const frameW = drawW + 2 * pad;
    const frameH = drawH + 2 * pad;
    const frameLeft = Math.round(region.left + (region.width - frameW) / 2);
    const frameTop = Math.round(region.top + (region.height - frameH) / 2);
    addShape(slide, { left: frameLeft, top: frameTop, width: frameW, height: frameH }, {
      fill: C.surface,
      line: C.rule,
      lineWidth: 1,
    });
    const dataUrl = `data:image/png;base64,${buffer.toString("base64")}`;
    slide.images.add({
      dataUrl,
      alt: item.title,
      position: { left: frameLeft + pad, top: frameTop + pad, width: drawW, height: drawH },
      fit: "contain",
    });
  } else {
    addShape(slide, region, { fill: C.surface, line: C.rule, lineWidth: 1 });
    addText(slide, `Missing generated image:\n${item.image}`, { left: 100, top: 360, width: 1080, height: 80 }, {
      color: C.hot,
      fontSize: 24,
      bold: true,
      alignment: "center",
    });
  }
  chip(slide, MX, 648, C.hot, 11);
  addText(slide, item.note, { left: MX + 20, top: 644, width: 1132, height: 40 }, {
    color: C.muted,
    fontSize: 14,
  });
}

function layoutTable(slide, item) {
  addBackground(slide);
  addTitle(slide, item, 86);
  const left = MX;
  const top = 206;
  const colW = [248, 84, 104, 132, 116, 122, 152];
  const rowH = 44;
  let x = left;
  item.headers.forEach((h, i) => {
    addShape(slide, { left: x, top, width: colW[i], height: 40 }, { fill: C.dark, line: C.dark });
    addText(slide, h, { left: x + 12, top: top + 12, width: colW[i] - 20, height: 16 }, {
      color: C.white,
      fontSize: 12,
      bold: true,
      alignment: i === 0 ? "left" : "center",
    });
    x += colW[i];
  });
  item.rows.forEach((row, r) => {
    x = left;
    row.forEach((cell, i) => {
      const y = top + 40 + r * rowH;
      addShape(slide, { left: x, top: y, width: colW[i], height: rowH }, {
        fill: r % 2 ? C.surface : C.bg,
        line: C.rule,
        lineWidth: 1,
      });
      if (i === 0) addShape(slide, { left: x, top: y, width: 4, height: rowH }, { fill: C.hot, line: C.hot });
      addText(slide, cell, { left: x + 12, top: y + 13, width: colW[i] - 20, height: 16 }, {
        color: i === 0 ? C.ink : C.muted,
        fontSize: 12,
        alignment: i === 0 ? "left" : "center",
        bold: i === 0,
      });
      x += colW[i];
    });
  });
  addFooter(slide, item.footnote);
}

function layoutNotesGrid(slide, item) {
  addBackground(slide);
  addTitle(slide, item, 86);
  item.notes.forEach((note, i) => {
    const x = MX + (i % 3) * 384;
    const y = 210 + Math.floor(i / 3) * 178;
    const pos = { left: x, top: y, width: 360, height: 158 };
    card(slide, pos, C.blue);
    chip(slide, x + 24, y + 26, C.hot, 10);
    addText(slide, note[0], { left: x + 44, top: y + 22, width: 300, height: 28 }, {
      color: C.ink,
      fontSize: 18,
      bold: true,
    });
    addText(slide, note[1], { left: x + 24, top: y + 62, width: 318, height: 84 }, {
      color: C.muted,
      fontSize: 15,
    });
  });
}

function layoutPlaybook(slide, item) {
  addBackground(slide);
  addTitle(slide, item, 86);
  const n = item.plays.length;
  const gap = 16;
  const cw = Math.floor((1152 - gap * (n - 1)) / n);
  item.plays.forEach((play, i) => {
    const x = MX + i * (cw + gap);
    const pos = { left: x, top: 220, width: cw, height: 300 };
    card(slide, pos, C.hot);
    addText(slide, `0${i + 1}`, { left: x + 22, top: 244, width: 60, height: 38 }, {
      color: C.hot,
      fontSize: 30,
      bold: true,
    });
    addShape(slide, { left: x + 22, top: 292, width: 36, height: 3 }, { fill: C.rule, line: C.rule });
    addText(slide, play[0], { left: x + 22, top: 312, width: cw - 40, height: 90 }, {
      color: C.ink,
      fontSize: 19,
      bold: true,
    });
    addText(slide, play[1], { left: x + 22, top: 432, width: cw - 40, height: 76 }, {
      color: C.blue,
      fontSize: 15,
      bold: true,
    });
  });
}

function layoutReferences(slide, item) {
  addBackground(slide);
  addTitle(slide, item, 86);
  references.forEach((ref, i) => {
    const x = i < 4 ? MX : 660;
    const y = 214 + (i % 4) * 92;
    addShape(slide, { left: x, top: y, width: 512, height: 76 }, { fill: C.surface, line: C.rule, lineWidth: 1 });
    addText(slide, `[${i + 1}]`, { left: x + 18, top: y + 14, width: 44, height: 22 }, {
      color: C.hot,
      fontSize: 15,
      bold: true,
    });
    addText(slide, ref, { left: x + 62, top: y + 14, width: 434, height: 52 }, {
      color: C.ink,
      fontSize: 15,
    });
  });
}

function layoutDiscussion(slide, item) {
  addBackground(slide, true);
  addTitle(slide, item, 92, true);
  item.questions.forEach((q, i) => {
    const x = i % 2 === 0 ? MX : 660;
    const y = 250 + Math.floor(i / 2) * 150;
    const pos = { left: x, top: y, width: 516, height: 126, fill: C.darkSurf, line: C.darkRule };
    card(slide, pos, C.hot);
    addText(slide, `Q${String(i + 1).padStart(2, "0")}`, { left: x + 26, top: y + 22, width: 70, height: 26 }, {
      color: C.hot,
      fontSize: 17,
      bold: true,
    });
    addText(slide, q, { left: x + 26, top: y + 54, width: 466, height: 64 }, {
      color: C.white,
      fontSize: 19,
    });
  });
}

export async function renderSlideByNumber(presentation, number) {
  const item = slides[number - 1];
  const slide = presentation.slides.add();
  switch (item.layout) {
    case "cover":
      layoutCover(slide, item);
      break;
    case "twoColumn":
      layoutTwoColumn(slide, item);
      break;
    case "roadmap":
      layoutRoadmap(slide, item);
      break;
    case "comparison":
      layoutComparison(slide, item);
      break;
    case "setup":
      layoutSetup(slide, item);
      break;
    case "method":
      layoutMethod(slide, item);
      break;
    case "failure":
      layoutFailure(slide, item);
      break;
    case "imageProof":
      layoutImageProof(slide, item);
      break;
    case "table":
      layoutTable(slide, item);
      break;
    case "notesGrid":
      layoutNotesGrid(slide, item);
      break;
    case "playbook":
      layoutPlaybook(slide, item);
      break;
    case "references":
      layoutReferences(slide, item);
      break;
    case "discussion":
      layoutDiscussion(slide, item);
      break;
    default:
      addBackground(slide);
      addTitle(slide, item);
  }
  return slide;
}
