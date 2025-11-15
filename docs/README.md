# GitHub Pages Site for cslrtools2

This directory contains the documentation website for `cslrtools2`, hosted on GitHub Pages.

## Structure

```
docs/
├── index.md              # Homepage
├── installation.md       # Installation guide
├── _config.yml          # Jekyll configuration
├── api/                 # API reference
│   ├── index.md
│   ├── lmpipe.md
│   ├── convsize.md
│   └── sldataset.md
└── examples/            # Usage examples
    └── index.md
```

## Building Locally

### Prerequisites

Install Jekyll:

```bash
# Ubuntu/Debian
sudo apt-get install ruby-full build-essential zlib1g-dev
gem install jekyll bundler

# macOS
brew install ruby
gem install jekyll bundler

# Windows
# Install Ruby from https://rubyinstaller.org/
gem install jekyll bundler
```

### Serve Locally

```bash
cd docs
bundle install
bundle exec jekyll serve
```

Visit `http://localhost:4000/python-cslrtools2/`

## GitHub Pages Setup

1. Go to repository Settings
2. Navigate to Pages section
3. Source: Deploy from a branch
4. Branch: `main` (or your default branch)
5. Folder: `/docs`
6. Save

The site will be available at: `https://ikegami-yukino.github.io/python-cslrtools2/`

## Theme

This site uses the [Cayman theme](https://github.com/pages-themes/cayman) for GitHub Pages.

To change the theme, edit `_config.yml`:

```yaml
theme: jekyll-theme-minimal  # or another theme
```

Available themes:
- jekyll-theme-cayman
- jekyll-theme-minimal
- jekyll-theme-slate
- jekyll-theme-architect
- jekyll-theme-dinky

## Navigation

Add navigation links by editing `_config.yml`:

```yaml
navigation:
  - title: Home
    url: /
  - title: Installation
    url: /installation
  - title: API Reference
    url: /api/
  - title: Examples
    url: /examples/
```

## SEO

The site includes:
- `jekyll-seo-tag` plugin for meta tags
- `jekyll-sitemap` for sitemap.xml
- `jekyll-feed` for RSS feed

Configure in `_config.yml`:

```yaml
title: cslrtools2
description: Comprehensive toolkit for Continuous Sign Language Recognition research
url: "https://ikegami-yukino.github.io"
baseurl: "/python-cslrtools2"
```

## Contributing

To add or update documentation:

1. Edit Markdown files in `docs/`
2. Test locally with `bundle exec jekyll serve`
3. Commit and push to GitHub
4. GitHub Pages will automatically rebuild the site

## File Formats

- Use Markdown (`.md`) for content
- Front matter for page metadata:

```markdown
---
layout: default
title: Page Title
---

# Content here
```

## Code Highlighting

Supports syntax highlighting for code blocks:

```python
from cslrtools2 import convsize
```

## Math Equations

Supports LaTeX math via KaTeX or MathJax (add to `_config.yml`):

```markdown
$$
\text{output} = \left\lfloor \frac{\text{input} + 2 \times \text{padding}}{\text{stride}} \right\rfloor + 1
$$
```

## Links

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Markdown Guide](https://www.markdownguide.org/)
