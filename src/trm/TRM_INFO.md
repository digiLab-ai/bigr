# TRM backbone source

`src/trm/` is a trimmed fork of nano-trm (https://github.com/olivkoch/nano-trm),
commit `e5659b1f5b309883e936b8b829ccaa09b3df5eb0`, imported on 2026-03-10. The training
entry point is `python -m trm.nn.train`; guided inference reaches the model only through
the adapters in `bigr.backbone`. The licence, upstream attributions and the list of
modifications are in `NOTICES.md` at the repository root; the upstream licence and README
are alongside this file.
