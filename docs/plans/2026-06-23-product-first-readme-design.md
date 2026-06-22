# Product-First README Design

## Goal

Replace the upstream-oriented README with a repository-specific landing page for the production Qwen3-TTS fine-tuning and inference service.

## Audience and positioning

The README should help an engineer quickly understand what this repository adds around Alibaba's Qwen3-TTS models, decide whether it fits their deployment, and start the service. It should credit the upstream Qwen team without presenting this fork as the upstream model repository.

## Structure

1. Product identity, Apache 2.0 badge, and concise value proposition.
2. Supported production workflows and operational features.
3. Architecture and fine-tuning lifecycle diagrams.
4. Docker-first and local/TIR quick starts.
5. A small set of representative API requests, linking to `API_DOCS.md` for the complete contract.
6. GPU, storage, translation, observability, and event-driven inference guidance.
7. Repository map, requirements, security caveats, attribution, citation, and Apache 2.0 license.

## Content boundaries

- Use only capabilities evidenced by the repository, `API_DOCS.md`, `.env.example`, Docker Compose, and specialist docs.
- Keep detailed endpoint tables and upstream model internals out of the landing page.
- Preserve links to upstream model documentation and the Qwen3-TTS technical report.
- Make Docker Compose the shortest path while retaining the existing TIR/local scripts.

## Verification

Check local Markdown links, heading anchors, shell commands, endpoint names, environment variables, Apache 2.0 metadata, and the final diff.
