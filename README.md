# New Paradigm — ejlok1.github.io

Personal blog by **Eu Jin Lok** (Kaggle Grandmaster, Senior AI Engineer at H2O.ai).

Covering Agentic AI, LLMs in production, enterprise ML, and the lineage from classical NLP to modern AI systems.

🌐 **Live site**: https://ejlok1.github.io

---

## Quick Setup (First Time)

### 1. Create the GitHub repo

Go to GitHub and create a new repository named exactly:
```
ejlok1.github.io
```
This special name tells GitHub Pages to host it at `https://ejlok1.github.io`.

### 2. Install Hugo (Extended)

**Mac:**
```bash
brew install hugo
```

**Windows:**
```bash
winget install Hugo.Hugo.Extended
```

**Linux:**
```bash
sudo snap install hugo
```

### 3. Clone this repo and add the PaperMod theme

```bash
git clone https://github.com/ejlok1/ejlok1.github.io.git
cd ejlok1.github.io

# Add PaperMod theme as a git submodule
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
git submodule update --init --recursive
```

### 4. Enable GitHub Pages in repo settings

1. Go to your repo on GitHub → **Settings** → **Pages**
2. Under **Source**, select **GitHub Actions**
3. That's it — the workflow in `.github/workflows/hugo.yml` handles everything

### 5. Push and go live

```bash
git add .
git commit -m "Initial site"
git push origin main
```

Your site will be live at **https://ejlok1.github.io** within ~60 seconds. ✅

---

## Writing a New Post

Posts live in `content/posts/`. Create a new `.md` file:

```bash
# Option A: Hugo CLI (scaffolds frontmatter automatically)
hugo new posts/my-new-post.md

# Option B: Just copy an existing post and edit it
cp content/posts/2026-mcp-servers-explained.md content/posts/2026-my-new-post.md
```

Each post starts with **frontmatter** — metadata between the `---` markers:

```markdown
---
title: "Your Post Title"
date: 2026-05-01
description: "A one-sentence summary shown in post lists."
tags: ["agentic-ai", "LLM"]
categories: ["New Series"]
showToc: true
---

Your content here in Markdown...
```

Then push:

```bash
git add content/posts/my-new-post.md
git commit -m "Add post: your post title"
git push
```

Site rebuilds automatically in ~60 seconds.

---

## Previewing Locally

```bash
hugo server -D
```

Visit `http://localhost:1313` — live reloads as you save files.

---

## Site Structure

```
ejlok1.github.io/
├── hugo.toml                    # Site config (title, theme, menus, social links)
├── content/
│   ├── about.md                 # About page
│   ├── archives.md              # Auto-generated post archive
│   └── posts/
│       ├── episode-1-tfidf.md          # 2017 classic (migrated)
│       ├── episode-2-sentiment.md      # 2017 classic (migrated)
│       ├── episode-3-word-embeddings.md # 2018 classic (migrated)
│       ├── episode-4-cnn-text.md       # 2018 classic (migrated)
│       ├── episode-5-lstm.md           # 2018 classic (migrated)
│       ├── episode-6-gru.md            # 2018 classic (migrated)
│       ├── 2026-return-new-paradigm.md # Comeback post
│       ├── 2026-mcp-servers-explained.md
│       └── 2026-rag-in-production.md
├── themes/
│   └── PaperMod/               # Theme (git submodule — don't edit)
└── .github/
    └── workflows/
        └── hugo.yml             # Auto-deploy on push to main
```

---

## Updating the Config

Edit `hugo.toml` to change:
- Site title/description
- Social media links
- Homepage intro text
- Menu items

---

## Tips

- **Dark mode** is the default — users can toggle to light
- **Reading time** is shown automatically
- **Table of contents** appears on posts with `showToc: true`
- **Code blocks** have copy buttons built in
- **Tags** are auto-linked — use consistent tag names (lowercase, hyphenated)
