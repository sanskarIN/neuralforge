# Durable Social-Link Policy

## Publication rule

NeuralForge publication files intentionally omit X/Twitter URLs and usernames.

Social handles can be renamed, transferred, or removed after a reader purchases an immutable ebook copy. A stale handle would make the book contain incorrect information.

## Preferred durable destinations

Use stable project- or publisher-controlled destinations instead:

- Canonical companion repository: `https://github.com/sanskarIN/neuralforge`
- Official Ram Sandesh Gumroad storefront: `https://ramsandesh.gumroad.com`
- Repository issues/discussions for project-specific updates when enabled
- Stable support or business contact channels maintained by the publisher

## Gumroad storefront rule

For GitHub-facing NeuralForge pages that mention purchasing, storefront releases, publication downloads, or the official store, use this exact canonical URL:

`https://ramsandesh.gumroad.com`

Do not replace it with short-lived campaign, redirect, tracking, or product-specific URLs when a durable storefront destination is sufficient.

The repository may display `assets/gumroad-storefront.svg` as a clickable storefront badge. It is a custom NeuralForge repository graphic, not an official Gumroad corporate logo.

## Storefront-only exception for social profiles

If a temporary social link is useful on a storefront profile or marketing page, keep it outside the ebook manuscript so it can be changed without rebuilding purchased copies.

## Release check

Every publication release should scan PDF, EPUB, DOCX, metadata, descriptions, and package documentation for `x.com`, `twitter.com`, `Twitter`, and known historical usernames before publication.

GitHub-facing release QA should additionally verify that the canonical storefront URL remains exactly:

`https://ramsandesh.gumroad.com`
