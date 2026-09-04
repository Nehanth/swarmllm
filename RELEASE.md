# Releases

SwarmLLM deploys continuously from `main` to [swarmllm.ai](https://swarmllm.ai); the staging site tracks the active feature branch. Git tags mark milestones people can cite.

## Cutting a release

1. All GPU tests green on the maintainer's hardware (`npm run test:gpu`, `npm run test:q38`).
2. [docs/bench-log.md](docs/bench-log.md) has rows for every performance change since the last tag.
3. Update [CHANGELOG.md](CHANGELOG.md): move *Unreleased* into a dated section.
4. Tag: `git tag -a v0.X.0 -m "..." && git push --tags`.
5. Deploy: `npx vercel deploy --prod` from `main`.

Versioning is `0.MINOR.PATCH` until the peer protocol is declared stable; a protocol change bumps MINOR.
