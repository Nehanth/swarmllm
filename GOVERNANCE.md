# Governance

SwarmLLM is a small open-source project with a single maintainer. This document describes how decisions are made today and how that changes as the project grows.

## Roles

- **Maintainer** (currently [Nehanth Narendrula](https://github.com/Nehanth)): sets direction, reviews and merges pull requests, cuts releases, and has final say on disputes.
- **Collaborators**: contributors with a track record of merged work who are granted triage rights (labeling, closing duplicates, reviewing). Collaborators are added by the maintainer and listed in [AUTHORS](AUTHORS).
- **Contributors**: anyone who opens a pull request or issue. See [CONTRIBUTING.md](CONTRIBUTING.md).

## How decisions are made

- **Small changes** (bug fixes, docs, tests, kernel micro-optimizations that keep output bit-exact) go through a normal pull request and one review.
- **Substantial changes** need a short design note before code is written. Open an issue using the *Research / design proposal* template. Substantial means any of:
  - a change to the peer-to-peer protocol (message types, frame format, room lifecycle);
  - a new model architecture or quantization format in the engine;
  - anything that changes model output (a new kernel is only "done" when the golden tests still pass);
  - a new dependency or a build step;
  - anything touching what peers can learn about each other (see [SECURITY.md](SECURITY.md)).
- Disagreements are settled by discussion in the issue; the maintainer decides if consensus is not reached.

## Non-negotiables

These are the project's design commitments. Proposals that conflict with them will be declined regardless of implementation quality:

1. **Easy to use.** Opening a room link in a browser must always work with nothing installed, and it stays first-class. Native and headless peers (servers, Jetsons, gaming PCs, the CLI) are equally welcome; they extend the room, they never become a requirement for it.
2. **No accounts, no tokens, no ads.** SwarmLLM has no counterparty and will not add one.
3. **Honest claims.** Speed numbers come with the commit and hardware that produced them ([docs/bench-log.md](docs/bench-log.md)); privacy claims match [SECURITY.md](SECURITY.md).
4. **Output correctness is bit-exact by default.** Optimizations must reproduce the reference output; approximations need an explicit, documented switch.

## Releases

See [RELEASE.md](RELEASE.md).

## Changes to this document

Governance changes are proposed as pull requests to this file and require the maintainer's approval.
