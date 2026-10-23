# Data sources and rights

The MIT license applies to the author's original software, not to third-party
annual reports, financial disclosures, market observations, dependencies, or
third-party rights in research data derived from those sources. Dependencies
remain subject to their respective licenses.

The two included research snapshots contain corporate-year financial values,
report-derived controls and calculated variables. Available producer code
identifies OpenDART annual-report financial endpoints and official annual-report
text. Fiscal-year market controls are derived from cached price series; the
available acquisition code contains Daum, Naver-fallback and legacy pykrx routes.
The final snapshot records cache categories rather than a complete per-series
original-provider ledger. Source-provider redistribution terms have not been
independently verified in the local audit. This notice does not grant additional
third-party rights or assert that every historical acquisition branch was used.

The author authorized sharing the snapshots. To avoid publishing historical
credential-bearing exception text and private local paths, the public files
preserve their original headers, row order and all other cell strings while
clearing only these non-analytic fields:

- `bundle/finance/expanded_panel.csv`: `opendart_error_note` is blank. Public
  SHA-256: `c1100096d259f92b889078618412896cb46ae54ad1f0a394d2627eb2c9a1edfc`.
- `bundle/finance/frozen_augmented_panel.csv`: `text_cache_path` is blank. Public
  SHA-256: `fc8bd3fc69d0196ad14f4c14e6424eea0be85b3a857e91e9cc53ca6403fd74bc`.

Both snapshots contain22,654 unique corporate-year rows covering2,402 firms in
2015–2025. Historical event-related columns may be present in the source snapshot;
their presence does not make capital-market analyses part of the current paper
or the selected model workflow.

The original private snapshots remain unchanged and are not included. Their
hashes are provenance identities, not the hashes of these public derivatives.
Raw report ZIPs, raw webpage bodies, private collection receipts, API keys,
machine environments and wheel archives are not included in this staging set.
Users must obtain required source material through the applicable providers or
other authorized channels, retain the exact locked source identities, and follow
the relevant terms.

The15 public viewer source descriptors are path-independent projections of
historical evidence. They preserve locked request identities, order and response
body hashes and identify original private receipt hashes for lineage. They are
not newly performed HTTP captures, and do not establish historical whole-text
byte identity where that evidence is absent.

No live-redownload, publication, completed multi-round reproduction, or peer-review
approval claim is made by assembling these files. Source and execution gates
must be reported separately using their actual verification evidence.
